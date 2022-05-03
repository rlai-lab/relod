import torch
import argparse
import utils
import time
import os

from logger import Logger
from algo.comm import MODE
from algo.onboard_wrapper import OnboardWrapper
from algo.sac_rad_agent import SACRADLearner, SACRADPerformer
from envs.visual_ur5_reacher.configs.ur5_config import config
from envs.visual_ur5_reacher.ur5_wrapper import UR5Wrapper
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
    parser.add_argument('--setup', default='Visual-UR5')
    parser.add_argument('--env_name', default='Visual-UR5', type=str)
    parser.add_argument('--ur5_ip', default='129.128.159.210', type=str)
    parser.add_argument('--camera_id', default=0, type=int)
    parser.add_argument('--image_width', default=160, type=int)
    parser.add_argument('--image_height', default=90, type=int)
    parser.add_argument('--target_type', default='reaching', type=str)
    parser.add_argument('--random_action_repeat', default=1, type=int)
    parser.add_argument('--agent_action_repeat', default=1, type=int)
    parser.add_argument('--image_history', default=3, type=int)
    parser.add_argument('--joint_history', default=1, type=int)
    parser.add_argument('--ignore_joint', default=False, action='store_true')
    parser.add_argument('--episode_length_time', default=4.0, type=float)
    parser.add_argument('--dt', default=0.04, type=float)
    # replay buffer
    parser.add_argument('--replay_buffer_capacity', default=100000, type=int)
    parser.add_argument('--rad_offset', default=0.01, type=float)
    # train
    parser.add_argument('--init_steps', default=1000, type=int) 
    parser.add_argument('--env_steps', default=100000, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--async_mode', default=True, action='store_true')
    parser.add_argument('--max_updates_per_step', default=2, type=float)
    parser.add_argument('--update_every', default=50, type=int)
    parser.add_argument('--update_epochs', default=50, type=int)
    # critic
    parser.add_argument('--critic_lr', default=3e-4, type=float)
    parser.add_argument('--critic_tau', default=0.01, type=float)
    parser.add_argument('--critic_target_update_freq', default=2, type=int)
    parser.add_argument('--bootstrap_terminal', default=1, type=int)
    # actor
    parser.add_argument('--actor_lr', default=3e-4, type=float)
    parser.add_argument('--actor_update_freq', default=2, type=int)
    # encoder
    parser.add_argument('--encoder_tau', default=0.05, type=float)
    # sac
    parser.add_argument('--discount', default=0.99, type=float)
    parser.add_argument('--init_temperature', default=0.1, type=float)
    parser.add_argument('--alpha_lr', default=1e-4, type=float)
    # agent
    parser.add_argument('--remote_ip', default='localhost', type=str)
    parser.add_argument('--port', default=9876, type=int)
    parser.add_argument('--mode', default='ro', type=str, help="Modes in ['r', 'o', 'ro', 'e'] ")
    # misc
    parser.add_argument('--seed', default=0, type=int)
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
    elif args.mode == 'o':
        mode = MODE.LOCAL_ONLY
        mt = MonitorTarget()
        mt.reset_plot()
    elif args.mode == 'ro':
        mode = MODE.ONBOARD_REMOTE
    elif args.mode == 'e':
        mt = MonitorTarget()
        mt.reset_plot()
        mode = MODE.EVALUATION
    else:
        raise  NotImplementedError()

    if args.device is '':
        args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    args.work_dir += f'/results/{args.env_name}_{args.target_type}_' \
                     f'dt={args.dt}_bs={args.batch_size}_' \
                     f'dim={args.image_width}*{args.image_height}_{args.seed}/'

    args.model_dir = args.work_dir+'model'

    if mode == MODE.LOCAL_ONLY:
        utils.make_dir(args.work_dir)
        utils.make_dir(args.model_dir)
        L = Logger(args.work_dir, use_tb=args.save_tb)

    if mode == MODE.EVALUATION:
        args.image_dir = args.work_dir+'image'
        utils.make_dir(args.image_dir)

    env = UR5Wrapper(
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
        ignore_joint = args.ignore_joint,
    )

    utils.set_seed_everywhere(args.seed, None)
    
    obs, state = env.reset()
    args.image_shape = env.observation_space.shape
    args.proprioception_shape = env.state_space.shape
    args.action_shape = env.action_space.shape
    args.net_params = config
    args.env_action_space = env.action_space

    episode_length_step = int(args.episode_length_time / args.dt)
    agent = OnboardWrapper(episode_length_step, mode, remote_ip=args.remote_ip, port=args.port)
    agent.send_data(args)
    agent.init_performer(SACRADPerformer, args)
    agent.init_learner(SACRADLearner, args, agent.performer)

    # sync initial weights with remote
    agent.apply_remote_policy(block=True)

    if args.load_model > -1:
        agent.load_policy_from_file(args.model_dir, args.load_model)
    
    # TODO: Fix this hack. This gives us enough time to toggle target in the monitor
    
    time.sleep(10)
    episode, episode_reward, episode_step, done = 0, 0, 0, True
    if mode == MODE.EVALUATION:
        episode_image_dir = utils.make_dir(os.path.join(args.image_dir, str(episode)))
    # First inference took a while (~1 min), do it before the agent-env interaction loop
    if mode != MODE.REMOTE_ONLY:
        agent.performer.sample_action((obs, state), args.init_steps+1)

    if mode == MODE.EVALUATION and args.load_model > -1:
        args.init_steps = 0
    
    agent.send_init_ob((obs, state))
    start_time = time.time()
    for step in range(args.env_steps + args.init_steps):
        if mode == MODE.EVALUATION:
            image_to_save = np.transpose(obs, [1, 2, 0])
            image_to_save = image_to_save[:,:,0:3]
            cv2.imwrite(episode_image_dir+'/'+str(step)+'.png', image_to_save)

        action = agent.sample_action((obs, state), step)

        # step in the environment
        next_obs, next_state, reward, done, _ = env.step(action)

        episode_reward += reward
        episode_step += 1
        
        agent.push_sample((obs, state), action, reward, (next_obs, next_state), done)

        if done and step > 0:
            if mode == MODE.LOCAL_ONLY:
                L.log('train/duration', time.time() - start_time, step)
                L.log('train/episode_reward', episode_reward, step)
                L.log('train/episode', episode+1, step)
                L.dump(step)
                mt.reset_plot()

            next_obs, next_state = env.reset()
            agent.send_init_ob((next_obs, next_state))
            episode_reward = 0
            episode_step = 0
            episode += 1
            if mode == MODE.EVALUATION:
                episode_image_dir = utils.make_dir(os.path.join(args.image_dir, str(episode)))
                mt.reset_plot()
            start_time = time.time()

        stat = agent.update_policy(step)
        if stat is not None:
            for k, v in stat.items():
                L.log(k, v, step)
        
        obs = next_obs
        state = next_state

        if args.save_model and (step+1) % args.save_model_freq == 0:
            agent.save_policy_to_file(args.model_dir, step)

    if args.save_model:
        agent.save_policy_to_file(args.model_dir, step)
        
    # Clean up
    agent.close()
    env.terminate()
    print('Train finished')

if __name__ == '__main__':
    main()
