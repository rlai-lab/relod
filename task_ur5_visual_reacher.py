import torch
import argparse
import time
import cv2
import os
import wandb
import multiprocessing

import numpy as np
import relod.utils as utils
import matplotlib.pyplot as plt

from relod.logger import Logger
from relod.algo.comm import MODE
from relod.algo.local_wrapper import LocalWrapper
from relod.algo.sac_rad_agent import SACRADLearner, SACRADPerformer
from relod.envs.visual_ur5_min_time_reacher.env import VisualReacherEnv, VisualReacherMinTimeEnv, MonitorTarget
from relod.envs.visual_ur5_min_time_reacher.env_vid import VideoPlayer
from tqdm import tqdm


config = {
    
    'conv': [
        # in_channel, out_channel, kernel_size, stride
        [-1, 32, 3, 2],
        [32, 32, 3, 2],
        [32, 32, 3, 2],
        [32, 32, 3, 1],
    ],
    
    'latent': 50,

    'mlp': [
        [-1, 1024], # first hidden layer
        [1024, 1024], 
        [1024, -1] # output layer
    ],
}


def parse_args():
    parser = argparse.ArgumentParser(description='Local remote visual UR5 Reacher')
    # environment
    parser.add_argument('--setup', default='Visual-UR5-min-time')  # it says min-time here because the screen is setup closer. Actually using dense rewards in this file
    parser.add_argument('--env', default='ur5', type=str)
    parser.add_argument('--ur5_ip', default='129.128.159.210', type=str)
    parser.add_argument('--camera_id', default=0, type=int)
    parser.add_argument('--image_width', default=160, type=int)
    parser.add_argument('--image_height', default=90, type=int)
    parser.add_argument('--target_type', default='size', type=str)
    parser.add_argument('--random_action_repeat', default=1, type=int)
    parser.add_argument('--agent_action_repeat', default=1, type=int)
    parser.add_argument('--image_history', default=3, type=int)
    parser.add_argument('--joint_history', default=1, type=int)
    parser.add_argument('--ignore_joint', default=False, action='store_true')
    parser.add_argument('--episode_length_time', default=6.0, type=float)
    parser.add_argument('--dt', default=0.04, type=float)
    parser.add_argument('--size_tol', default=0.015, type=float)
    parser.add_argument('--center_tol', default=0.1, type=float)
    parser.add_argument('--reward_tol', default=2.0, type=float)
    parser.add_argument('--reset_penalty_steps', default=70, type=int)
    parser.add_argument('--reward', default=-1, type=float)
    parser.add_argument('--background_color', default='white', type=str)
    parser.add_argument('--eval_env_mode', default='video_easy_5', type=str)
    parser.add_argument('--train_env_mode', default='clean', type=str, help="Mode in ['clean', 'video_easy_5']")
    parser.add_argument('--use_sparse_reward', default=False, action='store_true')
    # replay buffer
    parser.add_argument('--replay_buffer_capacity', default=100000, type=int)
    parser.add_argument('--rad_offset', default=0.01, type=float, help="Offset for RAD. Default is 0.01. Will be set to 0 when running SAC")
    # train
    parser.add_argument('--algorithm', default='rad', type=str, help="Algorithms in ['rad', 'madi', 'svea']")
    parser.add_argument('--init_steps', default=1000, type=int) 
    parser.add_argument('--env_steps', default=100050, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--sync_mode', default=False, action='store_true')
    parser.add_argument('--max_updates_per_step', default=0.6, type=float)
    parser.add_argument('--update_every', default=50, type=int)
    parser.add_argument('--update_epochs', default=50, type=int)
    # critic
    parser.add_argument('--critic_lr', default=3e-4, type=float)
    parser.add_argument('--critic_tau', default=0.005, type=float)
    parser.add_argument('--critic_target_update_freq', default=1, type=int)
    parser.add_argument('--bootstrap_terminal', default=0, type=int)
    # actor
    parser.add_argument('--actor_lr', default=3e-4, type=float)
    parser.add_argument('--actor_update_freq', default=1, type=int)
    # encoder
    parser.add_argument('--encoder_tau', default=0.005, type=float)
    # sac
    parser.add_argument('--discount', default=0.99, type=float)
    parser.add_argument('--init_temperature', default=0.1, type=float)
    parser.add_argument('--alpha_lr', default=3e-4, type=float)
    # madi
    parser.add_argument('--masker_lr', default=3e-4, type=float)  # was 1e-3 in MaDi work, but 3e-4 is standard here. Can try 1e-3 later
    parser.add_argument('--save_mask', default=False, action='store_true')
    parser.add_argument('--save_mask_freq', default=1000, type=int)
    parser.add_argument('--save_augm', default=False, action='store_true')
    parser.add_argument('--save_augm_freq', default=1000, type=int)
    parser.add_argument('--strong_augment', default='none', type=str, help="Augmentations in ['none', 'conv', 'overlay']")
    parser.add_argument('--anneal_masker_lr', default='none', type=str, help="['none', 'cosine', 'cosine10k']")
    parser.add_argument('--when_augm', default='before', type=str,
                        help="When to augment. Before or after masking (or both)", choices=['before', 'after', 'both'])
    # agent
    parser.add_argument('--remote_ip', default='localhost', type=str)
    parser.add_argument('--port', default=9876, type=int)
    parser.add_argument('--mode', default='l', type=str, help="Modes in ['r', 'l', 'rl', 'e'] ")
    # misc
    parser.add_argument('--run_type', default='experiment', type=str)
    parser.add_argument('--description', default='', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--work_dir', default='results/', type=str)
    parser.add_argument('--save_tb', default=False, action='store_true')
    parser.add_argument('--save_model', default=False, action='store_true')
    parser.add_argument('--plot_learning_curve', default=False, action='store_true')
    parser.add_argument('--xtick', default=1200, type=int)
    parser.add_argument('--display_image',  default=False, action='store_true')
    parser.add_argument('--save_image',  default=False, action='store_true')
    parser.add_argument('--save_model_freq', default=10000, type=int)
    parser.add_argument('--load_model', default=-1, type=int)
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--lock', default=False, action='store_true')
    parser.add_argument('--wandb_mode', default='online', type=str, help="Either online, offline, or disabled")

    args = parser.parse_args()
    assert args.mode in ['r', 'l', 'rl', 'e']
    assert args.reward < 0 and args.reset_penalty_steps >= 0
    args.async_mode = not args.sync_mode
    return args

def set_one_thread():
  '''
  N.B: Pytorch over-allocates resources and hogs CPU, which makes experiments very slow!
  Set number of threads for pytorch to 1 to avoid this issue. This is a temporary workaround.
  '''
  os.environ['OMP_NUM_THREADS'] = '1'
  os.environ['MKL_NUM_THREADS'] = '1'
  torch.set_num_threads(1)

def main():
    args = parse_args()

    if args.mode == 'r':
        mode = MODE.REMOTE_ONLY
    elif args.mode == 'l':
        mode = MODE.LOCAL_ONLY
    elif args.mode == 'rl':
        mode = MODE.REMOTE_LOCAL
    elif args.mode == 'e':
        mode = MODE.EVALUATION
    else:
        raise NotImplementedError()

    if args.device is '':
        args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    args.work_dir += f'/{args.env}/timeout={args.episode_length_time:.0f}/seed={args.seed}'
    args.model_dir = args.work_dir+'/models'
    args.return_dir = args.work_dir+'/returns'
    if mode != MODE.EVALUATION:
        os.makedirs(args.model_dir, exist_ok=False)
        os.makedirs(args.return_dir, exist_ok=False)
    if mode == MODE.LOCAL_ONLY:
        L = Logger(args.return_dir, use_tb=args.save_tb)

    if args.save_image:
        args.image_dir = args.work_dir + f'/eval_images_{args.train_env_mode}'
        if mode == MODE.LOCAL_ONLY or mode == MODE.EVALUATION:
            os.makedirs(args.image_dir, exist_ok=False)

    env = VisualReacherEnv(
        setup = args.setup,
        ip = args.ur5_ip,
        seed = args.seed,
        camera_id = args.camera_id,
        image_width = args.image_width,
        image_height = args.image_height,
        target_type = args.target_type,
        image_history = args.image_history,
        joint_history = args.joint_history,
        episode_length = args.episode_length_time,
        dt = args.dt,
        size_tol = args.size_tol,
        center_tol = args.center_tol,
        reward_tol = args.reward_tol,
        background_color = args.background_color,
        sparse_reward=args.use_sparse_reward,
    )

    utils.set_seed_everywhere(args.seed, None)

    if args.train_env_mode == "clean":
        mt = MonitorTarget(args.background_color)
        mt.reset_plot()
    elif args.train_env_mode == "video_easy_5":
        player = VideoPlayer(mode='video_easy_5')
        video_process = multiprocessing.Process(target=player.run)
        video_process.start()
        env.bgr_lower = [0, 0, 120]
        env.bgr_upper = [50, 0, 255]

    image, prop = env.reset()
    if args.display_image:
        image_to_show = np.transpose(image, [1, 2, 0])
        image_to_show = image_to_show[:,:,-3:]
        cv2.imshow('raw', image_to_show)
        cv2.waitKey(1)
    args.image_shape = env.image_space.shape
    args.proprioception_shape = env.proprioception_space.shape
    print(env.proprioception_space.shape)
    args.action_shape = env.action_space.shape
    args.env_action_space = env.action_space
    args.net_params = config

    # start a new wandb run to track this script
    wandb.init(
        project="avg",
        config=vars(args),
        name=f"UR5-{args.algorithm}-seed{args.seed}-batch{args.batch_size}-{args.description}",
        entity="gauthamv",
        mode=args.wandb_mode,
    )

    episode_length_step = int(args.episode_length_time / args.dt)
    agent = LocalWrapper(episode_length_step, mode, remote_ip=args.remote_ip, port=args.port)
    agent.send_data(args)
    print(f"Algorithm: {args.algorithm}")
    mask_rec = None
    augm_rec = None

    if args.save_augm:
        args.augm_dir = args.work_dir + '/augmentations'
        os.makedirs(args.augm_dir, exist_ok=False)
        augm_rec = AugmentationRecorder(args.augm_dir, args)

    if args.algorithm == 'rad':
        agent.init_performer(SACRADPerformer, args)
        agent.init_learner(SACRADLearner, args, agent.performer)
    elif args.algorithm == 'sac':
        args.rad_offset = 0.0
        print("Running SAC: rad_offset is now set to 0 (overridden if it was not 0 already)")
        agent.init_performer(SACRADPerformer, args)
        agent.init_learner(SACRADLearner, args, agent.performer)
    elif args.algorithm == 'madi':
        agent.init_performer(MaDiPerformer, args)
        agent.init_learner(MaDiLearner, args, agent.performer, augm_rec)
        if args.save_mask:
            args.mask_dir = args.work_dir + '/madi_masks'
            os.makedirs(args.mask_dir, exist_ok=False)
            mask_rec = MaskRecorder(args.mask_dir, args)
    elif args.algorithm == 'svea':
        assert args.strong_augment != 'none', 'must specify strong_augment when running svea'
        agent.init_performer(SVEAPerformer, args)
        agent.init_learner(SVEALearner, args, agent.performer, augm_rec)
    else:
        raise NotImplementedError()

    # For the overlay augmentation
    if args.strong_augment == 'overlay':
        assert os.environ['DMCGB_DATASETS'] == "/home/gautham/madi/relod/datasets_augmentation/dmcgb", "Check your location of the Places365 dataset for the overlay augmentation"

    # input('go?')

    # sync initial weights with remote
    agent.apply_remote_policy(block=True)

    if args.load_model > -1:
        agent.load_policy_from_file(args.model_dir, args.load_model)
    
    # First inference took a while (~1 min), do it before the agent-env interaction loop
    if mode != MODE.REMOTE_ONLY:
        agent.performer.sample_action((image, prop))
        agent.performer.sample_action((image, prop))
        agent.performer.sample_action((image, prop))


    # Experiment block starts
    experiment_done = False
    total_steps = 0
    sub_epi = 1
    returns = []
    epi_lens = []
    start_time = time.time()
    print(f'Experiment starts at: {start_time}')
    
    # Eval
    if args.eval_env_mode != 'none':
        eval(total_steps, args, agent, env, mask_rec, L)
    
    # Save all rewards
    train_rewards = np.zeros((args.env_steps//150, 150))

    while not experiment_done:
        # start a new episode
        if mode == MODE.EVALUATION:
            if args.train_env_mode == "clean":
                mt.reset_plot()
            elif args.train_env_mode == "video_easy_5":
                player.switch.value = 1
            image, prop = env.reset()
        else:
            if args.train_env_mode == "clean":
                mt.reset_plot()
            elif args.train_env_mode == "video_easy_5":
                player.switch.value = 1

            image, prop = env.reset() 
        agent.send_init_ob((image, prop))
        ret = 0
        epi_steps = 0
        sub_steps = 0
        epi_done = 0
        if (mode == MODE.LOCAL_ONLY or mode == MODE.EVALUATION) and args.save_image:
            episode_image_dir = args.image_dir+f'/episode={len(returns)+1}/'
            os.makedirs(episode_image_dir, exist_ok=False)

        epi_start_time = time.time()
        while not experiment_done and not epi_done:
            if args.display_image or ((mode == MODE.LOCAL_ONLY or mode == MODE.EVALUATION) and args.save_image):

                if args.algorithm == 'madi':
                    image_torch = torch.as_tensor(image, device=args.device).float().unsqueeze(0)
                    mask = agent.performer._masker(image_torch.chunk(3, dim=1)[-1])
                    mask = mask.squeeze(0).detach().cpu().numpy()
                    mask = np.transpose(mask, [1, 2, 0])
                    mask = mask[:,:,0]
                    if args.display_image:
                        cv2.imshow('mask', mask)
                        cv2.waitKey(1)
                    if (mode == MODE.LOCAL_ONLY or mode == MODE.EVALUATION) and args.save_image:
                        cv2.imwrite(episode_image_dir+f'sub_epi={sub_epi}-epi_step={epi_steps}-mask.png', mask*255)
                    masked_obs = agent.performer.apply_mask(image_torch)
                    masked_obs = masked_obs.squeeze(0).detach().cpu().numpy()
                    masked_obs = np.transpose(masked_obs, [1, 2, 0])
                    masked_obs = masked_obs[:,:,-3:]
                    if args.display_image:
                        cv2.imshow('masked_obs', masked_obs)
                        cv2.waitKey(1)
                    if (mode == MODE.LOCAL_ONLY or mode == MODE.EVALUATION) and args.save_image:
                        cv2.imwrite(episode_image_dir+f'sub_epi={sub_epi}-epi_step={epi_steps}-masked.png', masked_obs)

                image_to_show = np.transpose(image, [1, 2, 0])
                image_to_show = image_to_show[:,:,-3:]
                if (mode == MODE.LOCAL_ONLY or mode == MODE.EVALUATION) and args.save_image:
                    cv2.imwrite(episode_image_dir+f'sub_epi={sub_epi}-epi_step={epi_steps}.png', image_to_show)
                if args.display_image:
                    cv2.imshow('raw', image_to_show)
                    cv2.waitKey(1)

            # log the mask
            if args.algorithm == 'madi' and args.save_mask and total_steps % args.save_mask_freq == 0:
                mask_rec.record(image, agent, total_steps, test_env=False, test_mode=None)

            # select an action
            action = agent.sample_action((image, prop))

            # step in the environment
            next_image, next_prop, reward, epi_done, _ = env.step(action)

            # store
            agent.push_sample((image, prop), action, reward, (next_image, next_prop), epi_done)

            stat = agent.update_policy(total_steps)
            if mode == MODE.LOCAL_ONLY and stat is not None:
                for k, v in stat.items():
                    L.log(k, v, total_steps)
            
            #### 
            image = next_image
            prop = next_prop

            # Log
            total_steps += 1
            ret += reward
            train_rewards[len(returns), epi_steps] = reward  # Save all rewards
            epi_steps += 1
            sub_steps += 1
            
            if args.save_model and total_steps % args.save_model_freq == 0:
                agent.save_policy_to_file(args.model_dir, total_steps)

            if not epi_done and sub_steps >= episode_length_step: # set timeout here
                sub_steps = 0
                ret += args.reset_penalty_steps * args.reward
                total_steps += args.reset_penalty_steps
                print(f'Sub episode {sub_epi} done.')

                image, prop = env.reset()
                agent.send_init_ob((image, prop))
                sub_epi += 1

            experiment_done = total_steps >= args.env_steps

        # save the last image
        if (mode == MODE.LOCAL_ONLY or mode == MODE.EVALUATION) and args.save_image:
            image_to_show = np.transpose(image, [1, 2, 0])
            image_to_show = image_to_show[:,:,-3:]
            cv2.imwrite(episode_image_dir+f'sub_epi={sub_epi}-epi_step={epi_steps}.png', image_to_show)

        if epi_done:  # episode done, save result
            returns.append(ret)
            epi_lens.append(epi_steps)
            if mode != MODE.EVALUATION:
                utils.save_returns(args.return_dir+'/return.txt', returns, epi_lens)

            if mode == MODE.LOCAL_ONLY:
                L.log('train/duration', time.time() - epi_start_time, total_steps)
                L.log('train/episode_reward', ret, total_steps)
                L.log('train/episode', len(returns), total_steps)
                if stat is not None:
                    for k, v in stat.items():
                        L.log(k, v, total_steps)
                L.dump(total_steps)
                if args.plot_learning_curve:
                    utils.show_learning_curve(args.return_dir+'/learning curve.png', returns, epi_lens, xtick=args.xtick)
            
            sub_epi += 1
        
        # Eval 
        if total_steps % 4500 == 0 and args.eval_env_mode != 'none':
            eval(total_steps, args, agent, env, mask_rec, L)

    duration = time.time() - start_time
    agent.save_policy_to_file(args.model_dir, total_steps)

    # Eval
    if args.eval_env_mode != 'none':
        eval(total_steps, args, agent, env, mask_rec, L)

    # Save train rewards
    np.savetxt(args.return_dir+"/train_rewards.txt", train_rewards)

    env.reset()
    agent.close()
    env.close()

    # always show a learning curve at the end
    if mode == MODE.LOCAL_ONLY:
        utils.show_learning_curve(args.return_dir+'/learning curve.png', returns, epi_lens, xtick=args.xtick)
    print(f"Finished in {duration}s")


def eval(total_steps, args, agent, env, mask_rec, L):
    eval_rewards = np.zeros((10, 150))
    agent.learner.pause_update()
    player = VideoPlayer(mode=args.eval_env_mode)
    video_process = multiprocessing.Process(target=player.run)
    video_process.start()
    env.bgr_lower = [0, 0, 120]
    env.bgr_upper = [50, 0, 255]
    EP = 10
    eval_ep_ind = np.random.randint(0, EP)
    eval_ep_step = np.random.randint(0, 150)
    eval_rets = []
    for ep in range(EP):
        image, prop = env.reset()
        eval_done = False
        eval_ret = 0
        episode_step = 0
        while not eval_done:
            action = agent.sample_action((image, prop))
            next_image, next_prop, reward, eval_done, _ = env.step(action)
            image = next_image
            prop = next_prop
            eval_ret += reward
            eval_rewards[ep, episode_step] = reward
            episode_step += 1
            # log the mask
            if args.algorithm == 'madi' and args.save_mask and ep == eval_ep_ind and episode_step == eval_ep_step:
                mask_rec.record(image, agent, total_steps, test_env=True, test_mode=args.eval_env_mode)
        player.switch.value = 1
        eval_rets.append(eval_ret)
    L.log('eval/step', total_steps, total_steps)
    L.log('eval/episode_reward', np.mean(eval_rets), total_steps)
    L.dump(total_steps)

    # Save eval rewards
    np.savetxt(args.return_dir+f"/eval_rewards_{total_steps}.txt", eval_rewards)

    env.bgr_lower = [0, 0, 120]
    env.bgr_upper = [50, 50, 255]
    player.switch.value = 2
    video_process.join()
    agent.learner.resume_update()


if __name__ == '__main__':
    # set_one_thread()
    main()
