import torch
import argparse
from algo.onboard_wrapper import OnboardWrapper
from algo.sac_rad_agent import SACRADAgent
import utils
from envs.visual_ur5_reacher.configs.ur5_config import config
from envs.visual_ur5_reacher.ur5_wrapper import UR5Wrapper
import time

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
    # actor
    parser.add_argument('--actor_lr', default=3e-4, type=float)
    parser.add_argument('--actor_update_freq', default=2, type=int)
    # encoder
    parser.add_argument('--encoder_tau', default=0.05, type=float)
    # sac
    parser.add_argument('--discount', default=0.99, type=float)
    parser.add_argument('--init_temperature', default=0.1, type=float)
    parser.add_argument('--alpha_lr', default=1e-4, type=float)
    parser.add_argument('--bootstrap_terminal', default=1, type=int)
    # agent
    parser.add_argument('--remote_ip', default='localhost', type=str)
    parser.add_argument('--port', default=9876, type=int)
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

    if args.device is '':
        args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    args.work_dir += f'/results/{args.env_name}_{args.target_type}_' \
                     f'dt={args.dt}_bs={args.batch_size}_' \
                     f'dim={args.image_width}*{args.image_height}_{args.seed}/'

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

    # TODO:
    #  rl_agent_class=SACRADAgent and remote_ip=<remote_ip> => remote-onboard
    #  rl_agent_class=None => remote-only
    agent = OnboardWrapper(episode_length_step,
                           remote_ip=args.remote_ip,
                           rl_agent_class=SACRADAgent, 
                           rl_agent_args=args)
    go = input('press any key to go')
    episode, episode_reward, episode_step, done = 0, 0, 0, True
    agent.send_init_ob((obs, state))
    for step in range(args.env_steps + args.init_steps):
        action = agent.sample_action((obs, state))

        # step in the environment
        next_obs, next_state, reward, done, _ = env.step(action)

        episode_reward += reward
        episode_step += 1
        
        agent.push_sample((obs, state), action, reward, (next_obs, next_state), done)

        if done and step > 0:
            next_obs, next_state = env.reset()
            agent.send_init_ob((next_obs, next_state))
            done = False
            episode_reward = 0
            episode_step = 0
            episode += 1
        
        agent.update_policy()
        
        obs = next_obs
        state = next_state

    # Clean up
    agent.close()
    env.terminate()
    print('Training finished')

if __name__ == '__main__':
    main()
