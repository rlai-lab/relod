import torch
import argparse
import os
from algo.onboard_wrapper import OnboardWrapper
from algo.sac_rad_agent import SACRADAgent
import utils
from senseact.utils import NormalizedEnv
from envs.create2_visual_reacher.env import Create2VisualReacherEnv


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
        [-1, 1024],
        [1024, 1024],
        [1024, -1]
    ],
}

def parse_args():
    parser = argparse.ArgumentParser()
    # environment
    parser.add_argument('--target_type', default='create2_visual_reacher', type=str)
    parser.add_argument('--episode_length_time', default=30.0, type=float)
    parser.add_argument('--dt', default=0.045, type=float)
    parser.add_argument('--image_height', default=120, type=int)
    parser.add_argument('--image_width', default=160, type=int)
    parser.add_argument('--stack_frames', default=3, type=int)
    parser.add_argument('--camera_id', default=0, type=int)
    parser.add_argument('--min_target_size', default=0.12, type=float)
    # replay buffer
    parser.add_argument('--replay_buffer_capacity', default=100000, type=int)
    parser.add_argument('--rad_offset', default=0.01, type=float)
    # train
    parser.add_argument('--init_steps', default=1000, type=int)
    parser.add_argument('--env_steps', default=160000, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--async_mode', default=True, action='store_true')
    parser.add_argument('--max_updates_per_step', default=1.0, type=float)
    parser.add_argument('--update_every', default=50, type=int)
    parser.add_argument('--update_epochs', default=50, type=int)
    # critic
    parser.add_argument('--critic_lr', default=1e-3, type=float)
    parser.add_argument('--critic_tau', default=0.01, type=float)
    parser.add_argument('--critic_target_update_freq', default=1, type=int)
    # actor
    parser.add_argument('--actor_lr', default=1e-3, type=float)
    parser.add_argument('--actor_update_freq', default=1, type=int)
    # encoder
    parser.add_argument('--encoder_tau', default=0.05, type=float)
    # sac
    parser.add_argument('--discount', default=1., type=float)
    parser.add_argument('--init_temperature', default=0.1, type=float)
    parser.add_argument('--alpha_lr', default=1e-4, type=float)
    parser.add_argument('--bootstrap_terminal', default=0, type=int)
    # agent
    parser.add_argument('--remote_ip', default='192.168.0.103', type=str)
    parser.add_argument('--port', default=9876, type=int)
    # misc
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--work_dir', default='.', type=str)
    parser.add_argument('--save_tb', default=False, action='store_true')
    parser.add_argument('--save_model', default=True, action='store_true')
    parser.add_argument('--save_model_freq', default=10000, type=int)
    parser.add_argument('--load_model', default=-1, type=int)
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--lock', default=False, action='store_true')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    if args.device is '':
        args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    if not args.async_mode:
        version = 'SAC_sync'
    elif args.async_mode and args.lock:
        version = 'SACv1'
    elif args.async_mode:
        version = 'SAC_async'
    else:
        raise NotImplementedError('Not a supported mode!')

    args.work_dir += f'/results/{version}_{args.target_type}_' \
                     f'dt={args.dt}_' \
                     f'seed={args.seed}_' \
                     f'target_size={args.min_target_size}/'

    if not 'conv' in config:
        image_shape = (0, 0, 0)
    else: 
        image_shape = (3*args.stack_frames, args.image_height, args.image_width)

    env = Create2VisualReacherEnv(
        episode_length_time=args.episode_length_time, 
        dt=args.dt,
        image_shape=image_shape,
        camera_id=args.camera_id,
        min_target_size=args.min_target_size
        )
    
    env = NormalizedEnv(env)
    utils.set_seed_everywhere(args.seed, env)
    env.start()

    args.image_shape = env.image_space.shape
    args.proprioception_shape = env.proprioception_space.shape
    args.action_shape = env.action_space.shape
    args.net_params = config
    args.env_action_space = env.action_space

    episode_length_step = int(args.episode_length_time / args.dt)
    agent = OnboardWrapper(episode_length_step,
                           remote_ip=args.remote_ip,
                           rl_agent_class=None, 
                           rl_agent_args=args)

    episode, episode_reward, episode_step, done = 0, 0, 0, True
    (image, propri) = env.reset()
    agent.send_init_ob((image, propri))
    for step in range(args.env_steps + args.init_steps):
        action = agent.sample_action((image, propri))

        # step in the environment
        (next_image, next_propri), reward, done, _ = env.step(action)

        episode_reward += reward
        episode_step += 1

        done = False if episode_step == episode_length_step else done

        agent.push_sample((image, propri), action, reward, (next_image, next_propri), done)

        if done or (episode_step == episode_length_step): # set time out here
            (next_image, next_propri) = env.reset()
            agent.send_init_ob((next_image, next_propri))
            done = False
            episode_reward = 0
            episode_step = 0
            episode += 1
        
        agent.update_policy()
        
        image = next_image
        propri = next_propri
        
    # Clean up
    agent.close()
    env.close()

if __name__ == '__main__':
    main()
