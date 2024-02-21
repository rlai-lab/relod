import torch
import argparse
import relod.utils as utils
import time
import numpy as np
import cv2
import os

from relod.logger import Logger
from relod.algo.comm import MODE
from relod.algo.local_wrapper import LocalWrapper
from relod.algo.sac_rad_agent import SACRADLearner, SACRADPerformer
from relod.envs.ur5_visual_reacher import VisualReacherEnv, MonitorTarget

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
    parser.add_argument('--setup', default='Visual-UR5-constrained')
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
    parser.add_argument('--episode_length_time', default=4.0, type=float)
    parser.add_argument('--dt', default=0.04, type=float)
    parser.add_argument('--size_tol', default=0.015, type=float)
    parser.add_argument('--center_tol', default=0.1, type=float)
    parser.add_argument('--reward_tol', default=2.0, type=float)
    parser.add_argument('--reset_penalty_steps', default=70, type=int)
    parser.add_argument('--reward', default=-1, type=float)
    parser.add_argument('--reset_type', default='zero', type=str, help=["zero", "random"])
    # replay buffer
    parser.add_argument('--replay_buffer_capacity', default=100000, type=int)
    parser.add_argument('--rad_offset', default=0.01, type=float)
    # train
    parser.add_argument('--init_steps', default=2000, type=int) 
    parser.add_argument('--env_steps', default=100000, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--sync_mode', default=False, action='store_true')
    parser.add_argument('--async_buffer', default=False, action='store_true')
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
    parser.add_argument('--mode', default='e', type=str, help="Modes in ['r', 'l', 'rl', 'e'] ")
    # misc
    parser.add_argument('--run_type', default='experiment', type=str)
    parser.add_argument('--description', default='', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--work_dir', default='results/', type=str)
    parser.add_argument('--save_tb', default=False, action='store_true')
    parser.add_argument('--save_model', default=False, action='store_true')
    parser.add_argument('--plot_learning_curve', default=False, action='store_true')
    parser.add_argument('--xtick', default=1200, type=int)
    parser.add_argument('--display_image', default=True, action='store_true')
    parser.add_argument('--save_image', default=False, action='store_true')
    parser.add_argument('--save_model_freq', default=10000, type=int)
    parser.add_argument('--load_model', default=-1, type=int)
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--lock', default=False, action='store_true')

    args = parser.parse_args()
    assert args.mode in ['r', 'l', 'rl', 'e']
    assert args.reward < 0 and args.reset_penalty_steps >= 0
    args.async_mode = not args.sync_mode
    return args

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
        raise  NotImplementedError()

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
        args.image_dir = args.work_dir+'/images'
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
        reset_type=args.reset_type,
    )

    utils.set_seed_everywhere(args.seed, None)
    mt = MonitorTarget()
    mt.reset_plot()
    input('Please hit Enter to proceed...')
    image, prop = env.reset()
    image_to_show = np.transpose(image, [1, 2, 0])
    image_to_show = image_to_show[:,:,-3:]
    cv2.imshow('raw', image_to_show)
    cv2.waitKey(1)
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
    
    # First inference took a while (~1 min), do it before the agent-env interaction loop
    if mode != MODE.REMOTE_ONLY:
        agent.performer.sample_action((image, prop))
        agent.performer.sample_action((image, prop))
        agent.performer.sample_action((image, prop))

    # Experiment block starts
    returns = []
    ep_lens = []
    start_time = time.time()
    print(f'Experiment starts at: {start_time}')
    for t in range(args.env_steps):
        # start a new episode
        if mode == MODE.EVALUATION:
            image, prop = env.reset()
            mt.reset_plot() 
        else:
            mt.reset_plot() 
            image, prop = env.reset() 
        agent.send_init_ob((image, prop))
        ret = 0
        ep_steps = 0        
        done = 0
        if (mode == MODE.LOCAL_ONLY or mode == MODE.EVALUATION) and args.save_image:
            episode_image_dir = args.image_dir+f'/episode={len(returns)+1}/'
            os.makedirs(episode_image_dir, exist_ok=False)

        epi_start_time = time.time()
        while not done:
            if args.display_image or ((mode == MODE.LOCAL_ONLY or mode == MODE.EVALUATION) and args.save_image):
                image_to_show = np.transpose(image, [1, 2, 0])
                image_to_show = image_to_show[:,:,-3:]
                if (mode == MODE.LOCAL_ONLY or mode == MODE.EVALUATION) and args.save_image:
                    cv2.imwrite(episode_image_dir+f'ep_step={ep_steps}.png', image_to_show)
                if args.display_image:
                    cv2.imshow('raw', image_to_show)
                    cv2.waitKey(1)

            # select an action
            action = agent.sample_action((image, prop))

            # step in the environment
            next_image, next_prop, reward, done, _ = env.step(action)

            # store
            agent.push_sample((image, prop), action, reward, (next_image, next_prop), done)

            stat = agent.update_policy(t)
            if mode == MODE.LOCAL_ONLY and stat is not None:
                for k, v in stat.items():
                    L.log(k, v, t)

            image = next_image
            prop = next_prop

            # Log
            ret += reward
            ep_steps += 1            

            if args.save_model and t % args.save_model_freq == 0:
                agent.save_policy_to_file(args.model_dir, t)


        # save the last image
        if (mode == MODE.LOCAL_ONLY or mode == MODE.EVALUATION) and args.save_image:
            image_to_show = np.transpose(image, [1, 2, 0])
            image_to_show = image_to_show[:,:,-3:]
            cv2.imwrite(episode_image_dir+f'ep_step={ep_steps}.png', image_to_show)
        
        returns.append(ret)
        ep_lens.append(ep_steps)
        if mode != MODE.EVALUATION:
            utils.save_returns(args.return_dir+'/return.txt', returns, ep_lens)

        if mode == MODE.LOCAL_ONLY:
            L.log('train/duration', time.time() - epi_start_time, t)
            L.log('train/episode_reward', ret, t)
            L.log('train/episode', len(returns), t)
            L.dump(t)
            if args.plot_learning_curve:
                utils.show_learning_curve(args.return_dir+'/learning curve.png', returns, ep_lens, xtick=args.xtick)

    duration = time.time() - start_time
    agent.save_policy_to_file(args.model_dir, t)

    # Clean up
    env.reset()
    agent.close()
    env.close()

    # always show a learning curve at the end
    if mode == MODE.LOCAL_ONLY:
        utils.show_learning_curve(args.return_dir+'/learning curve.png', returns, ep_lens, xtick=args.xtick)
    print(f"Finished in {duration}s")


if __name__ == '__main__':
    main()
