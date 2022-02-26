import torch
import argparse
import time
import utils
import os
import numpy as np

from algo.onboard_wrapper import OnboardWrapper
from algo.ppo_rad_agent import PPORADPerformer, PPORADLearner
from envs.mujoco_visual_reacher.env import ReacherWrapper
from algo.comm import MODE
from logger import Logger


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
    parser.add_argument('--target_type', default='visual_reacher', type=str)
    parser.add_argument('--image_height', default=125, type=int)
    parser.add_argument('--image_width', default=200, type=int)
    parser.add_argument('--stack_frames', default=3, type=int)
    parser.add_argument('--tol', default=0.036, type=float)
    parser.add_argument('--image_period', default=1, type=int)
    parser.add_argument('--episode_length_time', default=50, type=int)
    parser.add_argument('--dt', default=1, type=int)
    parser.add_argument('--env_steps', default=500000, type=int, help="# timesteps for the run")
    # RAD
    parser.add_argument('--freeze_cnn', default=0, type=int)
    parser.add_argument('--rad_offset', default=0.01, type=float)
    # PPO
    parser.add_argument('--batch_size', default=4096, type=int)
    parser.add_argument('--opt_batch_size', default=256, type=int, help="Optimizer batch size")
    parser.add_argument('--n_epochs', default=10, type=int, help="Number of learning epochs per PPO update")
    parser.add_argument('--actor_lr', default=0.0003, type=float)
    parser.add_argument('--critic_lr', default=0.001, type=float)
    parser.add_argument('--gamma', default=0.99, type=float, help="Discount factor")
    parser.add_argument('--lmbda', default=0.97, type=float, help="Lambda return coefficient")
    parser.add_argument('--clip_epsilon', default=0.2, type=float, help="Clip epsilon for KL divergence in PPO actor loss")
    parser.add_argument('--l2_reg', default=1e-4, type=float, help="L2 regularization coefficient")
    parser.add_argument('--bootstrap_terminal', default=0, type=int, help="Bootstrap on terminal state")
    # agent
    parser.add_argument('--remote_ip', default='localhost', type=str)
    parser.add_argument('--port', default=9876, type=int)
    parser.add_argument('--mode', default='ro', type=str, help="Modes in ['r', 'o', 'ro'] ")
    # misc
    parser.add_argument('--args_port', default=9630, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--work_dir', default='.', type=str)
    parser.add_argument('--save_tb', default=False, action='store_true')
    parser.add_argument('--save_model', default=False, action='store_true')
    parser.add_argument('--save_model_freq', default=1000, type=int)
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
    elif args.mode == 'ro':
        mode = MODE.ONBOARD_REMOTE
    else:
        raise  NotImplementedError()

    if args.device is '':
        args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    if not 'conv' in config:
        image_shape = (0, 0, 0)
    else: 
        image_shape = (3*args.stack_frames, args.image_height, args.image_width)

    args.work_dir += f'/results/{args.target_type}_' \
                     f'seed={args.seed}_' \
                     f'tol={args.tol}/'

    utils.make_dir(args.work_dir)

    model_dir = utils.make_dir(os.path.join(args.work_dir, 'model'))
    args.model_dir = model_dir
    L = Logger(args.work_dir, use_tb=args.save_tb)

    env = ReacherWrapper(args.tol, image_shape, args.image_period, use_ground_truth=True)
    utils.set_seed_everywhere(args.seed, env)

    args.image_shape = env.image_space.shape
    args.proprioception_shape = env.proprioception_space.shape
    args.action_shape = env.action_space.shape
    args.net_params = config
    args.env_action_space = env.action_space

    episode_length_step = int(args.episode_length_time / args.dt)
    agent = OnboardWrapper(episode_length_step, mode, remote_ip=args.remote_ip, port=args.port)
    agent.send_data(args)
    agent.init_performer(PPORADPerformer, args)
    agent.init_learner(PPORADLearner, args, agent.performer)

    # sync initial weights with remote
    agent.apply_remote_policy(block=True)
    
    episode, episode_reward, episode_step, done = 0, 0, 0, True
    image, propri = env.reset()
    image = torch.as_tensor(image.astype(np.float32))[None, :, :, :]
    propri = torch.as_tensor(propri.astype(np.float32))[None, :]
    agent.send_init_ob((image, propri))
    start_time = time.time()
    for step in range(args.env_steps):
        action, lprob = agent.sample_action((image, propri))

        next_image, next_propri, reward, done, _ = env.step(action)
        next_image = torch.as_tensor(next_image.astype(np.float32))[None, :, :, :]
        next_propri = torch.as_tensor(next_propri.astype(np.float32))[None, :]

        episode_reward += reward
        episode_step += 1

        agent.push_sample((image, propri), action, reward, (next_image, next_propri), done, lprob)
        
        if done or (episode_step == episode_length_step): # set time out here
            if mode == MODE.LOCAL_ONLY:
                L.log('train/duration', time.time() - start_time, step)
                L.log('train/episode_reward', episode_reward, step)
                L.dump(step)
                L.log('train/episode', episode+1, step)
                agent.update_policy(done, next_image, next_propri)

            if mode == MODE.ONBOARD_REMOTE:
                cmd = agent.recv_cmd()
                if cmd == 'new policy':
                    agent.apply_remote_policy(True)
            
            next_image, next_propri = env.reset()
            next_image = torch.as_tensor(next_image.astype(np.float32))[None, :, :, :]
            next_propri = torch.as_tensor(next_propri.astype(np.float32))[None, :]
            agent.send_init_ob((next_image, next_propri))
            episode_reward = 0
            episode_step = 0
            episode += 1
            start_time = time.time()
        
        image = next_image
        propri = next_propri

        # if args.save_model and (step+1) % args.save_model_freq == 0:
            # agent.save_policy_to_file(step)

    # if args.save_model:
        # agent.save_policy_to_file(step)
    # Clean up
    agent.close()
    env.close()
    print('Train finished')

if __name__ == '__main__':
    main()
