import torch
import argparse
import time
from algo.onboard_wrapper import OnboardWrapper
from algo.sac_rad_agent import SACRADPerformer, SACRADLearner
import utils
from envs.mujoco_visual_reacher.env import ReacherWrapper
from algo.comm import MODE
import socket
from algo.comm import send_message
from logger import Logger
import os

config = {
    '''
    'conv': [
        # in_channel, out_channel, kernel_size, stride
        [-1, 32, 3, 2],
        [32, 32, 3, 2],
        [32, 32, 3, 2],
        [32, 32, 3, 1],
    ],
    '''
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
    # replay buffer
    parser.add_argument('--replay_buffer_capacity', default=100000, type=int)
    parser.add_argument('--rad_offset', default=0.01, type=float)
    # train
    parser.add_argument('--init_steps', default=1000, type=int)
    parser.add_argument('--env_steps', default=20000, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--async_mode', default=True, action='store_true')
    parser.add_argument('--max_updates_per_step', default=1, type=float)
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

    if mode == MODE.LOCAL_ONLY:
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

    performer = SACRADPerformer(args)
    #learner = SACRADLearner(performer=performer, args=args)
    learner = None
    if mode in [MODE.REMOTE_ONLY, MODE.ONBOARD_REMOTE]:
        args_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        args_sock.connect((args.remote_ip, args.args_port))
        send_message(args, args_sock)
        args_sock.close()
        time.sleep(5)
    elif mode == MODE.LOCAL_ONLY:
        pass
    else:
        raise NotImplementedError()

    episode_length_step = int(args.episode_length_time / args.dt)
    onboard_wrapper = OnboardWrapper(episode_length_step,
                           mode,
                           remote_ip=args.remote_ip,
                           performer=performer,
                           learner=learner)

    episode, episode_reward, episode_step, done = 0, 0, 0, True
    image, propri = env.reset()
    onboard_wrapper.send_init_ob((image, propri))
    start_time = time.time()
    for step in range(args.env_steps + args.init_steps):
        action = onboard_wrapper.sample_action((image, propri), step)

        next_image, next_propri, reward, done, _ = env.step(action)

        episode_reward += reward
        episode_step += 1

        onboard_wrapper.push_sample((image, propri), action, reward, (next_image, next_propri), done)
        
        if done or (episode_step == episode_length_step): # set time out here
            if mode == MODE.LOCAL_ONLY:
                L.log('train/duration', time.time() - start_time, step)
                L.log('train/episode_reward', episode_reward, step)
                L.dump(step)
                L.log('train/episode', episode+1, step)

            next_image, next_propri = env.reset()
            onboard_wrapper.send_init_ob((next_image, next_propri))
            episode_reward = 0
            episode_step = 0
            episode += 1
            start_time = time.time()
        
        stat = onboard_wrapper.update_policy(step)
        if mode == MODE.LOCAL_ONLY and stat is not None:
            for k, v in stat.items():
                L.log(k, v, step)
        
        image = next_image
        propri = next_propri

        if mode == MODE.LOCAL_ONLY and args.save_model and (step+1) % args.save_model_freq == 0:
            performer.save_policy_to_file(step)

        time.sleep(0.04)

    if mode == MODE.LOCAL_ONLY:
        performer.save_policy_to_file(step)
    # Clean up
    onboard_wrapper.close()
    env.close()
    print('Training finished')

if __name__ == '__main__':
    main()