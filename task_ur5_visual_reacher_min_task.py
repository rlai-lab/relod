import torch
import argparse
import relod.utils as utils
import time
import os

from relod.logger import Logger
from relod.algo.comm import MODE
from relod.algo.local_wrapper import LocalWrapper
from relod.algo.sac_rad_agent import SACRADLearner, SACRADPerformer
from relod.envs.visual_ur5_reacher.configs.ur5_config import config
from relod.envs.visual_ur5_min_time_reacher.env import VisualReacherMinTimeEnv
from remote_learner_ur5 import MonitorTarget
import numpy as np
import cv2

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
    parser.add_argument('--setup', default='Visual-UR5-min-time')
    parser.add_argument('--env', default='Visual-UR5-min-time', type=str)
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
    parser.add_argument('--episode_length_time', default=30.0, type=float)
    parser.add_argument('--dt', default=0.04, type=float)
    parser.add_argument('--size_tol', default=0.015, type=float)
    parser.add_argument('--center_tol', default=0.1, type=float)
    parser.add_argument('--reward_tol', default=2.0, type=float)
    # replay buffer
    parser.add_argument('--replay_buffer_capacity', default=100000, type=int)
    parser.add_argument('--rad_offset', default=0.01, type=float)
    # train
    parser.add_argument('--init_steps', default=5000, type=int) 
    parser.add_argument('--env_steps', default=120000, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--async_mode', default=True, action='store_true')
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
    # agent
    parser.add_argument('--remote_ip', default='localhost', type=str)
    parser.add_argument('--port', default=9876, type=int)
    parser.add_argument('--mode', default='rl', type=str, help="Modes in ['r', 'l', 'rl', 'e'] ")
    # misc
    parser.add_argument('--description', default='size_margin=20', type=str)
    parser.add_argument('--seed', default=4, type=int)
    parser.add_argument('--work_dir', default='.', type=str)
    parser.add_argument('--save_tb', default=False, action='store_true')
    parser.add_argument('--save_model', default=True, action='store_true')
    #parser.add_argument('--save_buffer', default=False, action='store_true')
    parser.add_argument('--save_model_freq', default=10000, type=int)
    parser.add_argument('--load_model', default=-1, type=int)
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--lock', default=False, action='store_true')

    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    if args.mode == 'r':
        mode = MODE.REMOTE_ONLY
    elif args.mode == 'l':
        mode = MODE.LOCAL_ONLY
        mt = MonitorTarget()
        
    elif args.mode == 'rl':
        mode = MODE.REMOTE_LOCAL
    elif args.mode == 'e':
        mt = MonitorTarget()
        mt.reset_plot()
        mode = MODE.EVALUATION
    else:
        raise  NotImplementedError()

    if args.device is '':
        args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    args.work_dir += f'/results/{args.env}_' \
                     f'dt={args.dt}_bs={args.batch_size}_' \
                     f'target_type={args.target_type}_'\
                     f'dim={args.image_width}*{args.image_height}_{args.seed}_'+args.description

    args.model_dir = args.work_dir+'/model'

    if mode == MODE.LOCAL_ONLY:
        utils.make_dir(args.work_dir)
        utils.make_dir(args.model_dir)
        L = Logger(args.work_dir, use_tb=args.save_tb)

    if mode == MODE.EVALUATION:
        args.image_dir = args.work_dir+'image'
        utils.make_dir(args.image_dir)

    env = VisualReacherMinTimeEnv(
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
    )

    utils.set_seed_everywhere(args.seed, None)
    mt.reset_plot()
    mt.reset_plot()
    mt.reset_plot()
    mt.reset_plot()
    input('go?')
    image, prop = env.reset()
    image_to_show = np.transpose(image, [1, 2, 0])
    image_to_show = image_to_show[:,:,-3:]
    cv2.imshow('raw', image_to_show)
    cv2.waitKey(0)
    args.image_shape = env.image_space.shape
    args.proprioception_shape = env.proprioception_space.shape
    args.action_shape = env.action_space.shape
    args.env_action_space = env.action_space
    args.net_params = config

    episode_length_step = int(args.episode_length_time / args.dt)
    agent = LocalWrapper(episode_length_step, mode, remote_ip=args.remote_ip, port=args.port)
    agent.send_data(args)
    agent.init_performer(SACRADPerformer, args)
    agent.init_learner(SACRADLearner, args, agent.performer)

    # sync initial weights with remote
    agent.apply_remote_policy(block=True)

    if args.load_model > -1:
        agent.load_policy_from_file(args.model_dir, args.load_model)
    
    episode, episode_reward, episode_step, done = 0, 0, 0, True
    if mode == MODE.EVALUATION:
        episode_image_dir = utils.make_dir(os.path.join(args.image_dir, str(episode)))
    # First inference took a while (~1 min), do it before the agent-env interaction loop
    if mode != MODE.REMOTE_ONLY:
        agent.performer.sample_action((image, prop), args.init_steps+1)

    if mode == MODE.EVALUATION and args.load_model > -1:
        args.init_steps = 0
    
    agent.send_init_ob((image, prop))
    success = 0
    start_time = time.time()
    for step in range(args.env_steps + args.init_steps):
        image_to_show = np.transpose(image, [1, 2, 0])
        image_to_show = image_to_show[:,:,-3:]
        cv2.imshow('raw', image_to_show)
        cv2.waitKey(1)

        action = agent.sample_action((image, prop), step)
        # step in the environment
        next_image, next_prop, reward, done, _ = env.step(action)

        episode_reward += reward
        episode_step += 1
        
        agent.push_sample((image, prop), action, reward, (next_image, next_prop), done)

        if done or (episode_step == episode_length_step): # set time out here
            if done:
                success += 1

            if mode == MODE.LOCAL_ONLY:
                L.log('train/duration', time.time() - start_time, step)
                L.log('train/episode_reward', episode_reward, step)
                L.log('train/episode', episode+1, step)
                L.dump(step)
                mt.reset_plot()

            next_image, next_prop = env.reset()
            agent.send_init_ob((next_image, next_prop))
            episode_reward = 0
            episode_step = 0
            episode += 1
            if mode == MODE.EVALUATION:
                episode_image_dir = utils.make_dir(os.path.join(args.image_dir, str(episode)))
                mt.reset_plot()
            success_rate = success / episode
            print('success rate:', success_rate)
            start_time = time.time()

        stat = agent.update_policy(step)
        if stat is not None:
            for k, v in stat.items():
                L.log(k, v, step)
        
        image = next_image
        prop = next_prop

        if args.save_model and (step+1) % args.save_model_freq == 0:
            agent.save_policy_to_file(args.model_dir, step)

    if args.save_model:
        agent.save_policy_to_file(args.model_dir, step)
        
    # Clean up
    agent.close()
    env.close()
    print('Train finished')

if __name__ == '__main__':
    main()
