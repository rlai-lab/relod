
import time
import torch
import utils

import numpy as np
import torch.multiprocessing as mp

from algo.rl_agent import BaseLearner, BasePerformer
from algo.ppo_rad_buffer import VisuomotorReplayBuffer 
from algo.ppo_models import ActorModel, CriticModel
from torch.optim import Adam
from torch import nn

class PPORADPerformer(BasePerformer):
    def __init__(self, args) -> None:
        self._args = args
        self.device = torch.device(args.device)

        if not 'conv' in self._args.net_params:  # no image
            self._args.image_shape = (0, 0, 0)

        self._actor = ActorModel(args.image_shape, args.proprioception_shape, args.action_shape[0], args.net_params,
                                args.rad_offset, args.freeze_cnn).to(self.device)

        self._critic = CriticModel(args.image_shape, args.proprioception_shape, args.net_params,
                                  args.rad_offset, args.freeze_cnn).to(self.device)
        
        if hasattr(self._actor.encoder, 'convs'):
            self._actor.encoder.convs = self._critic.encoder.convs

        self.train()
    
    def train(self, is_training=True):
        self._actor.train(is_training)
        self._critic.train(is_training)
        self.is_training = is_training

    def sample_action(self, ob):
        img, prop = ob
        img = img.to(self.device)
        prop = prop.to(self.device)

        with torch.no_grad():
            mu, action, lprob = self._actor(img, prop, random_rad=False, detach_encoder=True)

        return (action.cpu().view(-1), lprob.cpu().view(-1))

    def load_policy(self, policy):
        actor_weights = policy['actor']
        for key in actor_weights:
            actor_weights[key] = torch.from_numpy(actor_weights[key]).to(self._args.device)

        critic_weights = policy['critic']
        for key in critic_weights:
            critic_weights[key] = torch.from_numpy(critic_weights[key]).to(self._args.device)

        self._actor.load_state_dict(actor_weights)
        self._critic.load_state_dict(critic_weights)

    def load_policy_from_file(self, model_dir, step):
        self._actor.load_state_dict(
            torch.load('%s/actor_%s.pt' % (model_dir, step))
        )
        self._critic.load_state_dict(
            torch.load('%s/critic_%s.pt' % (model_dir, step))
        )

    def close(self):
        del self

class PPORADLearner(BaseLearner):
    def __init__(self, args, performer=None) -> None:
        self._args = args
        self.gamma = args.gamma
        self.batch_size = args.batch_size
        self.bootstrap_terminal = args.bootstrap_terminal
        self.device = torch.device(args.device)
        self.n_updates = 0
        self.lmbda = args.lmbda

        # TODO: Hack of +1000 to account for episode not being done when we have batch_size samples 
        self.buffer = VisuomotorReplayBuffer(args.image_shape, args.proprioception_shape, args.action_shape,
                                            args.batch_size + 1000, store_lprob=True)
        

        if performer == None:
            performer = PPORADPerformer(args)

        self._performer = performer
        self._actor = performer._actor
        self._critic = performer._critic

        # optimizers
        self._init_optimizers()
        
        self._performer.train()
    
    def _init_optimizers(self):
        self._actor_opt = Adam(self._actor.parameters(), lr=self._args.actor_lr, weight_decay=self._args.l2_reg)
        self._critic_loss = nn.MSELoss()
        self._critic_opt = Adam(self._critic.parameters(), lr=self._args.critic_lr, weight_decay=self._args.l2_reg)
  
    def estimate_returns_advantages(self, rewards, dones, vals):
        """ len(rewards) = len(dones) = len(vals)-1
        Args:
            rewards:
            dones:
            vals:
        Returns:
        """
        advs = torch.as_tensor(np.zeros(len(vals), dtype=np.float32), device=self.device)

        for t in reversed(range(len(rewards))):
            if self.bootstrap_terminal:
                delta = rewards[t] + self.gamma * vals[t+1] - vals[t]
                advs[t] = delta + self.gamma * self.lmbda * advs[t+1]
            else:
                delta = rewards[t] + (1 - dones[t]) * self.gamma * vals[t + 1] - vals[t]
                advs[t] = delta + (1 - dones[t]) * self.gamma * self.lmbda * advs[t + 1]

        rets = advs[:-1] + vals[:-1]
        return rets, advs[:-1]

    def update(self, next_imgs, next_propris):
        images, propris, actions, rewards, dones, old_lprobs = self.buffer.sample(len(self.buffer))
        images = torch.cat([images, next_imgs])
        propris = torch.cat([propris, next_propris])
        vals = []
        with torch.no_grad():
            end = len(images)
            for ind in range(0, end, 256):
                inds = np.arange(ind, min(end, ind+256))
                img = images[inds].to(self.device)
                prop = propris[inds].to(self.device)
                v = self._critic(images=img, proprioceptions=prop, random_rad=True, detach_encoder=False)
                vals.append(v)

        vals = torch.cat(vals)
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)
        old_lprobs = old_lprobs.to(self.device)
        rets, advs = self.estimate_returns_advantages(rewards=rewards, dones=dones, vals=vals)

        # Normalize advantages
        norm_advs = (advs - advs.mean()) / advs.std()

        inds = np.arange(len(rewards))
        for itr in range(self._args.n_epochs):
            np.random.shuffle(inds)
            for i_start in range(0, len(self.buffer), self._args.opt_batch_size):
                opt_inds = inds[i_start: min(i_start+self._args.opt_batch_size, len(inds)-1)]
                img = images[opt_inds].to(self.device)
                prop = propris[opt_inds].to(self.device)
                a = actions[opt_inds].to(self.device)

                # Policy update preparation
                new_lprobs = self._actor.lprob(img, prop, a, random_rad=True, detach_encoder=True)
                new_vals = self._critic(img, prop, random_rad=True, detach_encoder=False)
                ratio = torch.exp(new_lprobs - old_lprobs[opt_inds])
                p_loss = ratio * norm_advs[opt_inds]
                clipped_p_loss = torch.clamp(ratio, 1 - self._args.clip_epsilon, 1 + self._args.clip_epsilon) * norm_advs[opt_inds]
                actor_loss = -(torch.min(p_loss, clipped_p_loss)).mean()
                critic_loss = self._critic_loss(new_vals, rets[opt_inds])
                loss = actor_loss + critic_loss

                # Apply gradients
                self._actor_opt.zero_grad()
                self._critic_opt.zero_grad()
                loss.backward()
                self._actor_opt.step()
                self._critic_opt.step()
    
    def push_sample(self, ob, action, reward, next_ob, done, lprob):
        (image, propri) = ob
        self.buffer.push(image, propri, action, reward, done, lprob)

    def save_policy_to_file(self, model_dir, step):
        torch.save(
            self._actor.state_dict(), '%s/actor_%s.pt' % (model_dir, step)
        )
        torch.save(
            self._critic.state_dict(), '%s/critic_%s.pt' % (model_dir, step)
        )

    def load_policy_from_file(self, model_dir, step):
        self._performer.load_policy_from_file(model_dir, step)

    def update_policy(self, done, next_imgs, next_propris):
        if len(self.buffer) >= self.batch_size and done:
            tic = time.time()
            self.update(next_imgs, next_propris)
            self.buffer.reset()
            self.n_updates += 1
            print("Update {} took {}s".format(self.n_updates, time.time()-tic))
            return True
            
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
