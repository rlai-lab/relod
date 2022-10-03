import torch
import argparse
import utils
import time
import os

from logger import Logger
from algo.comm import MODE
from algo.local_wrapper import LocalWrapper
from algo.ppo_rad_agent import PPORADPerformer, PPORADLearner
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
    parser.add_argument('--camera_id', default=2, type=int)
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
    parser.add_argument('--env_steps', default=150000, type=int)
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
    parser.add_argument('--bootstrap_terminal', default=1, type=int, help="Bootstrap on terminal state")
    # agent
    parser.add_argument('--remote_ip', default='192.168.0.100', type=str)
    parser.add_argument('--port', default=9876, type=int)
    parser.add_argument('--mode', default='rl', type=str, help="Modes in ['r', 'l', 'rl', 'e'] ")
    # misc
    parser.add_argument('--seed', default=2, type=int)
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
        mt.reset_plot()
    elif args.mode == 'rl':
        mode = MODE.REMOTE_LOCAL
    elif args.mode == 'e':
        mode = MODE.EVALUATION
        mt = MonitorTarget()
        mt.reset_plot()
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
    agent = LocalWrapper(episode_length_step, mode, remote_ip=args.remote_ip, port=args.port)
    agent.send_data(args)
    agent.init_performer(PPORADPerformer, args)
    agent.init_learner(PPORADLearner, args, agent.performer)

    # sync initial weights with remote
    agent.apply_remote_policy(block=True)

    if args.load_model > -1:
        agent.load_policy_from_file(args.model_dir, args.load_model)
    
    # TODO: Fix this hack. This gives us enough time to toggle target in the monitor
    time.sleep(10)
    episode, episode_reward, episode_step, done = 0, 0, 0, True
    if mode == MODE.EVALUATION:
        episode_image_dir = utils.make_dir(os.path.join(args.image_dir, str(episode)))

    obs = torch.as_tensor(obs.astype(np.float32))[None, :, :, :]
    state = torch.as_tensor(state.astype(np.float32))[None, :]
    agent.send_init_ob((obs, state))
    start_time = time.time()
    for step in range(args.env_steps):
        if mode == MODE.EVALUATION:
            image = np.squeeze(obs.cpu().numpy())
            image_to_save = np.transpose(image, [1, 2, 0])
            image_to_save = image_to_save[:,:,0:3]
            cv2.imwrite(episode_image_dir+'/'+str(step)+'.png', image_to_save)

        action, lprob = agent.sample_action((obs, state))

        # step in the environment
        next_obs, next_state, reward, done, _ = env.step(action.cpu().numpy())
        next_obs = torch.as_tensor(next_obs.astype(np.float32))[None, :, :, :]
        next_state = torch.as_tensor(next_state.astype(np.float32))[None, :]

        episode_reward += reward
        episode_step += 1
        
        agent.push_sample((obs, state), action, reward, (next_obs, next_state), done, lprob)

        if done and step > 0:
            if mode == MODE.LOCAL_ONLY:
                L.log('train/duration', time.time() - start_time, step)
                L.log('train/episode_reward', episode_reward, step)
                L.dump(step)
                L.log('train/episode', episode+1, step)
                agent.update_policy(done, next_obs, next_state)
                mt.reset_plot()
            
            if mode == MODE.REMOTE_LOCAL:
                if agent.recv_cmd() == 'new policy':
                    agent.apply_remote_policy(True)

            next_obs, next_state = env.reset()
            next_obs = torch.as_tensor(next_obs.astype(np.float32))[None, :, :, :]
            next_state = torch.as_tensor(next_state.astype(np.float32))[None, :]
            agent.send_init_ob((next_obs, next_state))
            episode_reward = 0
            episode_step = 0
            episode += 1
            if mode == MODE.EVALUATION:
                episode_image_dir = utils.make_dir(os.path.join(args.image_dir, str(episode)))
                mt.reset_plot()
            start_time = time.time()
        
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
    # torch.multiprocessing.set_start_method('spawn')
    main()
