import torch
import argparse
import utils
import time
import os

from logger import Logger
from algo.comm import MODE
from algo.onboard_wrapper import OnboardWrapper
from algo.sac_rad_agent import SACRADLearner, SACRADPerformer

from senseact.utils import NormalizedEnv
from envs.create2_visual_reacher.env import Create2VisualReacherEnv

import numpy as np
import cv2
from gym.spaces import Box
from utils import append_time

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
    parser.add_argument('--min_target_size', default=0.2, type=float)
    # Reset threshold
    parser.add_argument('--reset_thresh', default=0.9, type=float, help="Action threshold between [-1, 1]")
    # replay buffer
    parser.add_argument('--replay_buffer_capacity', default=100000, type=int)
    parser.add_argument('--rad_offset', default=0.01, type=float)
    # train
    parser.add_argument('--init_steps', default=1000, type=int)
    parser.add_argument('--env_steps', default=150000, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--async_mode', default=True, action='store_true')
    parser.add_argument('--max_updates_per_step', default=1.0, type=float)
    parser.add_argument('--update_every', default=50, type=int)
    parser.add_argument('--update_epochs', default=50, type=int)
    # critic
    parser.add_argument('--critic_lr', default=1e-3, type=float)
    parser.add_argument('--critic_tau', default=0.01, type=float)
    parser.add_argument('--critic_target_update_freq', default=1, type=int)
    parser.add_argument('--bootstrap_terminal', default=0, type=int)
    # actor
    parser.add_argument('--actor_lr', default=1e-3, type=float)
    parser.add_argument('--actor_update_freq', default=1, type=int)
    # encoder
    parser.add_argument('--encoder_tau', default=0.05, type=float)
    # sac
    parser.add_argument('--discount', default=1., type=float)
    parser.add_argument('--init_temperature', default=0.1, type=float)
    parser.add_argument('--alpha_lr', default=1e-4, type=float)
    # agent
    parser.add_argument('--remote_ip', default='192.168.0.104', type=str)
    parser.add_argument('--port', default=9876, type=int)
    parser.add_argument('--mode', default='e', type=str, help="Modes in ['r', 'o', 'ro', 'e'] ")
    # misc
    parser.add_argument('--appendix', default='with_time', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--work_dir', default='.', type=str)
    parser.add_argument('--save_tb', default=False, action='store_true')
    parser.add_argument('--save_model', default=True, action='store_true')
    parser.add_argument('--save_model_freq', default=10000, type=int)
    parser.add_argument('--load_model', default=99999, type=int)
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
    elif args.mode == 'e':
        mode = MODE.EVALUATION
    else:
        raise  NotImplementedError()

    if args.device == '':
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
                     f'target_size={args.min_target_size}_'+args.appendix+'/'
    args.model_dir = args.work_dir+'model'
    print(args.model_dir)
    if mode == MODE.LOCAL_ONLY:
        utils.make_dir(args.work_dir)
        utils.make_dir(args.model_dir)
        L = Logger(args.work_dir, use_tb=args.save_tb)

    if mode == MODE.EVALUATION:
        args.image_dir = args.work_dir+'image'
        utils.make_dir(args.image_dir)
    
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
    
    env = NormalizedEnv(env) # need to check space
    
    utils.set_seed_everywhere(args.seed, env)
    env.start()

    args.image_shape = env.image_space.shape
    args.proprioception_shape = (env.proprioception_space.shape[0]+1,)
    args.x_action_dim = env.action_space.shape[0]
    args.action_shape = (env.action_space.shape[0]+1,)    
    args.net_params = config
    args.env_action_space = Box(-1, 1, shape=args.action_shape)
    
    episode_length_step = int(args.episode_length_time / args.dt)
    agent = OnboardWrapper(episode_length_step, mode, remote_ip=args.remote_ip, port=args.port)
    agent.send_data(args)
    agent.init_performer(SACRADPerformer, args)
    agent.init_learner(SACRADLearner, args, agent.performer)

    # sync initial weights with remote
    agent.apply_remote_policy(block=True)

    if args.load_model > -1:
        agent.load_policy_from_file(args.model_dir, args.load_model)
        print('loaded model:', args.load_model)
        
    if mode == MODE.EVALUATION and args.load_model > -1:
        args.init_steps = 0
    
    episodes = 0
    step = 0
    if mode == MODE.EVALUATION:
        episode_image_dir = utils.make_dir(os.path.join(args.image_dir, str(episodes)))

    while step < args.env_steps: 
        # start a new episode
        ret, episode_step, n_reset, done, flush_step = 0, 0, 0, 0, episode_length_step
        (image, prop) = env.reset()
        prop = append_time(prop, episode_step)

        # First inference took a while (~1 min), do it before the agent-env interaction loop
        if mode != MODE.REMOTE_ONLY and step == 0:
            print('first inference')
            agent.performer.sample_action((image, prop), args.init_steps+1)
        
        agent.send_init_ob((image, prop))

        # start the interaction loop
        start_time = time.time()
        while not done:
            if mode == MODE.EVALUATION:
                image_to_save = np.transpose(image, [1, 2, 0])
                image_to_save = image_to_save[:,:,0:3]
                cv2.imwrite(episode_image_dir+'/'+str(step)+'.png', image_to_save)

            if episode_step >= flush_step:
                env.stop_roomba()
                agent.flush_sample_queue()
                flush_step += episode_length_step
                
            action = agent.sample_action((image, prop), step)
            x_action = action[:args.x_action_dim]
            reset_action = action[-1]
    
            # Reset action
            if reset_action > args.reset_thresh:
                n_reset += 1
                episode_step += 80-1
                step += 80-1
                done = 0
                reward = -80

                next_image, next_prop = env.reset()
                assert agent.recv_cmd() == 'received' # sync here to avoid full queue
            
                if mode == MODE.EVALUATION:
                    episode_image_dir = utils.make_dir(os.path.join(args.image_dir, str(episodes)))
            else:
                # step in the environment
                (next_image, next_prop), reward, done, _ = env.step(x_action)

            ret += reward
            episode_step += 1
            step += 1
            next_prop = append_time(next_prop, episode_step)

            agent.push_sample((image, prop), action, reward, (next_image, next_prop), done)              
  
            stat = agent.update_policy(step)
            if stat is not None:
                for k, v in stat.items():
                    L.log(k, v, step)
        
            image = next_image
            prop = next_prop

            if args.save_model and step % args.save_model_freq == 0:
                agent.save_policy_to_file(args.model_dir, step)

        # after an episode is done, log info
        if mode == MODE.LOCAL_ONLY:
            L.log('train/duration', time.time() - start_time, step)
            L.log('train/episode_reward', ret, step)
            L.log('train/episode', episodes, step)
            L.log('train/n_reset', n_reset, step)
            L.dump(step)

        if mode == MODE.EVALUATION:
            episode_image_dir = utils.make_dir(os.path.join(args.image_dir, str(episodes)))

        episodes += 1
    
    # after training, save the model
    if args.save_model:
        agent.save_policy_to_file(args.model_dir, step)

    # Clean up
    agent.close()
    env.close()
    print('Train finished')

if __name__ == '__main__':
    main()