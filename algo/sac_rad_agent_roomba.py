from algo.rl_agent import BaseLearner, BasePerformer
from algo.models import ActorModel, CriticModel
import copy
import torch
import utils
import numpy as np
import torch.multiprocessing as mp
from algo.sac_rad_buffer import AsyncRadReplayBuffer, RadReplayBuffer
import queue

class SACRADPerformer(BasePerformer):
    '''
    avoid duplicating the performer weights
    '''
    def __init__(self, args) -> None:
        self._args = args
        self._args.device = torch.device(args.device)

        if not 'conv' in self._args.net_params:  # no image
            self._args.image_shape = (0, 0, 0)

        self._actor = ActorModel(self._args.image_shape,
                                 self._args.proprioception_shape,
                                 self._args.action_shape[0],
                                 self._args.net_params,
                                 self._args.rad_offset).to(self._args.device)

        self._critic = CriticModel(self._args.image_shape,
                                   self._args.proprioception_shape,
                                   self._args.action_shape[0],
                                   self._args.net_params,
                                   self._args.rad_offset).to(self._args.device)
        self._critic_target = copy.deepcopy(self._critic) # also copies the encoder instance
        if hasattr(self._actor.encoder, 'convs'):
            self._actor.encoder.convs = self._critic.encoder.convs

        self.train()

    def train(self, is_training=True):
        self._actor.train(is_training)
        self._critic.train(is_training)
        self._critic_target.train(is_training)
        self.is_training = is_training

    def load_policy_from_file(self, model_dir, step):
        self._actor.load_state_dict(
            torch.load('%s/actor_%s.pt' % (model_dir, step))
        )
        self._critic.load_state_dict(
            torch.load('%s/critic_%s.pt' % (model_dir, step))
        )

    def load_policy(self, policy):
        actor_weights = policy['actor']
        for key in actor_weights:
            actor_weights[key] = torch.from_numpy(actor_weights[key]).to(self._args.device)

        critic_weights = policy['critic']
        for key in critic_weights:
            critic_weights[key] = torch.from_numpy(critic_weights[key]).to(self._args.device)

        self._actor.load_state_dict(actor_weights)
        self._critic.load_state_dict(critic_weights)

    def sample_action(self, ob, step):
        # sample action for data collection
        if step < self._args.init_steps:
            action = self._args.env_action_space.sample()
        else:
            with utils.eval_mode(self):
                (image, propri) = ob

                with torch.no_grad():
                    if image is not None:
                        image = torch.FloatTensor(image).to(self._args.device)
                        image.unsqueeze_(0)

                    if propri is not None:
                        propri = torch.FloatTensor(propri).to(self._args.device)
                        propri.unsqueeze_(0)

                    _, pi, _, _ = self._actor(
                        image, propri, random_rad=False, compute_pi=True, compute_log_pi=False,
                    )

                    action = pi.cpu().data.numpy().flatten()

        return action

    def close(self):
        del self

class SACRADLearner(BaseLearner):
    def __init__(self, args) -> None:
        self._args = args
        self._args.device = torch.device(args.device)

        if not 'conv' in self._args.net_params: # no image
            self._args.image_shape = (0, 0, 0)

        if self._args.async_mode:
            # initialize processes in 'spawn' mode, required by CUDA runtime
            ctx = mp.get_context('spawn')
            episode_length_step = int(self._args.episode_length_time / self._args.dt)
            self._sample_queue = ctx.Queue(episode_length_step+100)
            self._minibatch_queue = ctx.Queue(100)

            # initialize data augmentation process
            self._replay_buffer_process = ctx.Process(target=AsyncRadReplayBuffer,
                                    args=(
                                        self._args.image_shape,
                                        self._args.proprioception_shape,
                                        self._args.action_shape,
                                        self._args.replay_buffer_capacity,
                                        self._args.batch_size,
                                        self._sample_queue,
                                        self._minibatch_queue,
                                        self._args.init_steps,
                                        self._args.max_updates_per_step,
                                        )
                                )
            self._replay_buffer_process.start()
        else:
            self._replay_buffer = RadReplayBuffer(
                image_shape=self._args.image_shape,
                proprioception_shape=self._args.proprioception_shape,
                action_shape=self._args.action_shape,
                capacity=self._args.replay_buffer_capacity,
                batch_size=self._args.batch_size)

        if self._args.async_mode:
            self._update_queue = ctx.Queue(2)
            self._update_process = ctx.Process(target=self._async_update)
            self._update_process.start()

    def set_performer(self, performer):
        self._performer = performer
        
    def get_policy(self):
        actor_weights = self._actor.state_dict()
        for key in actor_weights:
            actor_weights[key] = actor_weights[key].cpu().numpy()

        critic_weights = self._critic.state_dict()
        for key in critic_weights:
            critic_weights[key] = critic_weights[key].cpu().numpy()

        return {
                'actor': actor_weights,
                'critic': critic_weights
            }

    def _init_optimizers(self):
        self._actor_optimizer = torch.optim.Adam(
            self._actor.parameters(), lr=self._args.actor_lr, betas=(0.9, 0.999)
        )

        self._critic_optimizer = torch.optim.Adam(
            self._critic.parameters(), lr=self._args.critic_lr, betas=(0.9, 0.999)
        )

        self._log_alpha_optimizer = torch.optim.Adam(
            [self._log_alpha], lr=self._args.alpha_lr, betas=(0.5, 0.999)
        )

    def _share_memory(self):
        self._actor.share_memory()
        self._critic.share_memory()
        self._critic_target.share_memory()
        self._log_alpha.share_memory_()

    def update_policy(self, step):
        if self._args.async_mode:
            try:
                (stat, new_policy) = self._update_queue.get_nowait()
                self._performer.load_policy(new_policy)
            except queue.Empty:
                return None

            return stat
        
        if step > self._args.init_steps and (step % self._args.update_every == 0):
            for _ in range(self._args.update_epochs):
                stat = self._update(*self._replay_buffer.sample())
            
            return stat
        
        return None
    
    def _update_critic(self, images, proprioceptions, actions, rewards, next_images, next_proprioceptions, dones):
        with torch.no_grad():
            _, policy_actions, log_pis, _ = self._actor(next_images, next_proprioceptions)
            target_Q1, target_Q2 = self._critic_target(next_images, next_proprioceptions, policy_actions)
            target_V = torch.min(target_Q1, target_Q2) - self._alpha.detach() * log_pis
            if self._args.bootstrap_terminal:
                # enable infinite bootstrap
                target_Q = rewards + (self._args.discount * target_V)
            else:
                target_Q = rewards + ((1.0 - dones) * self._args.discount * target_V)

        # get current Q estimates
        current_Q1, current_Q2 = self._critic(images, proprioceptions, actions, detach_encoder=False)

        critic_loss = torch.mean((current_Q1 - target_Q) ** 2 + (current_Q2 - target_Q) ** 2)

        # Optimize the critic
        self._critic_optimizer.zero_grad()
        critic_loss.backward()
        #torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1)
        self._critic_optimizer.step()

        critic_stats = {
            'train_critic/loss': critic_loss.item()
        }

        return critic_stats

    def _update_actor_and_alpha(self, images, proprioceptions):
        # detach encoder, so we don't update it with the actor loss
        _, pis, log_pis, log_stds = self._actor(images, proprioceptions ,detach_encoder=True)
        actor_Q1, actor_Q2 = self._critic(images, proprioceptions, pis, detach_encoder=True)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self._alpha.detach() * log_pis - actor_Q).mean()

        entropy = 0.5 * log_stds.shape[1] * (1.0 + np.log(2 * np.pi)
                                            ) + log_stds.sum(dim=-1)

        # optimize the actor
        self._actor_optimizer.zero_grad()
        actor_loss.backward()
        self._actor_optimizer.step()

        self._log_alpha_optimizer.zero_grad()
        alpha_loss = (self._alpha *
                      (-log_pis - self._target_entropy).detach()).mean()
        alpha_loss.backward()
        self._log_alpha_optimizer.step()

        actor_stats = {
            'train_actor/loss': actor_loss.item(),
            'train_actor/target_entropy': self._target_entropy.item(),
            'train_actor/entropy': entropy.mean().item(),
            'train_alpha/loss': alpha_loss.item(),
            'train_alpha/value': self._alpha.item(),
            'train/entropy': entropy.mean().item(),
        }
        return actor_stats

    def _soft_update_target(self):
        utils.soft_update_params(
            self._critic.Q1, self._critic_target.Q1, self._args.critic_tau
        )
        utils.soft_update_params(
            self._critic.Q2, self._critic_target.Q2, self._args.critic_tau
        )
        utils.soft_update_params(
            self._critic.encoder, self._critic_target.encoder,
            self._args.encoder_tau
        )

    def _update(self, images, propris, actions, rewards, next_images, next_propris, dones):
        # regular update of SAC_RAD, sequentially augment data and train
        if images is not None:
            images = torch.as_tensor(images, device=self._args.device).float()
            next_images = torch.as_tensor(next_images, device=self._args.device).float()
        if propris is not None:
            propris = torch.as_tensor(propris, device=self._args.device).float()
            next_propris = torch.as_tensor(next_propris, device=self._args.device).float()
        actions = torch.as_tensor(actions, device=self._args.device)
        rewards = torch.as_tensor(rewards, device=self._args.device)
        dones = torch.as_tensor(dones, device=self._args.device)
        
        stats = self._update_critic(images, propris, actions, rewards, next_images, next_propris, dones)
        if self._num_updates % self._args.actor_update_freq == 0:
            actor_stats = self._update_actor_and_alpha(images, propris)
            stats = {**stats, **actor_stats}
        if self._num_updates % self._args.critic_target_update_freq == 0:
            self._soft_update_target()
        stats['train/batch_reward'] = rewards.mean().item()
        stats['train/num_updates'] = self._num_updates
        self._num_updates += 1
        
        new_policy = self.get_policy()
        return (stats, new_policy)
        
    def _async_update(self):
        self._performer = SACRADPerformer(self._args)
        self._performer.train()
        self._actor = self._performer._actor
        self._critic = self._performer._critic
        self._critic_target = self._performer._critic_target

        self._log_alpha = torch.tensor(np.log(self._args.init_temperature)).to(self._args.device)
        self._log_alpha.requires_grad = True
        # set target entropy to -|A|
        self._target_entropy = -np.prod(self._args.action_shape)

        self._num_updates = 0

        # optimizers
        self._init_optimizers()

        while True:
            try:
                self._update_queue.put_nowait(self._update(*self._minibatch_queue.get()))
            except queue.Full:
                pass

    def push_sample(self, ob, action, reward, next_ob, done):
        (image, propri) = ob
        (next_image, next_propri) = next_ob

        if self._args.async_mode:
            self._sample_queue.put((image, propri, action, reward, next_image, next_propri, done))
        else:
            self._replay_buffer.add(image, propri, action, reward, next_image, next_propri, done)

    def pause_update(self):
        if self._args.async_mode:
            self._sample_queue.put('pause')
    
    def resume_update(self):
        if self._args.async_mode:
            self._sample_queue.put('resume')

    def save_policy_to_file(self, model_dir, step):
        torch.save(
            self._actor.state_dict(), '%s/actor_%s.pt' % (model_dir, step)
        )
        torch.save(
            self._critic.state_dict(), '%s/critic_%s.pt' % (model_dir, step)
        )

    def load_policy_from_file(self, model_dir, step):
        self._performer.load_policy_from_file(model_dir, step)

    def close(self):
        if self._args.async_mode:
            self._replay_buffer_process.terminate()
            self._update_process.terminate()
            self._replay_buffer_process.join()
            self._update_process.join()

        del self

    @property
    def _alpha(self):
        return self._log_alpha.exp()

