import torch
import argparse
import os
import time
import json

from algo.onboard_wrapper import OnboardWrapper
from algo.comm import MODE
from algo.sac_rad_agent import SACRADLearner, SACRADPerformer
import utils
from logger import Logger
from envs.ur5_min_time_reacher.env import ReacherEnvMinTime
from senseact.utils import NormalizedEnv # fix bug

config = {
    'latent': 50,

    'mlp': [
        [-1, 1024], # first hidden layer
        [1024, 1024], 
        [1024, -1] # output layer
    ],
}

def parse_args():
    parser = argparse.ArgumentParser(description='Local remote UR5 Reacher')
    # environment
    parser.add_argument('--env_name', default='ur5_min_time_reacher', type=str)
    parser.add_argument('--ur5_ip', default='129.128.159.210', type=str)
    parser.add_argument('--episode_length_time', default=8, type=int)
    parser.add_argument('--dt', default=0.04, type=float)
    # replay buffer
    parser.add_argument('--replay_buffer_capacity', default=1000000, type=int)
    # parser.add_argument('--rad_offset', default=0.01, type=float)
    # train
    parser.add_argument('--init_steps', default=5000, type=int) 
    parser.add_argument('--num_steps', default=120000, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--async_mode', default=True, action='store_true')
    parser.add_argument('--max_updates_per_step', default=1, type=float)
    parser.add_argument('--update_every', default=50, type=int)
    parser.add_argument('--update_epochs', default=50, type=int)
    # critic
    parser.add_argument('--critic_lr', default=3e-4, type=float)
    parser.add_argument('--critic_tau', default=0.005, type=float)
    parser.add_argument('--critic_target_update_freq', default=1, type=int)
    # actor
    parser.add_argument('--actor_lr', default=3e-4, type=float)
    parser.add_argument('--actor_update_freq', default=1, type=int)
    # encoder
    parser.add_argument('--encoder_tau', default=0.05, type=float)
    # sac
    parser.add_argument('--discount', default=1.0, type=float)
    parser.add_argument('--init_temperature', default=0.2, type=float)
    parser.add_argument('--alpha_lr', default=1e-4, type=float)
    # agent
    parser.add_argument('--remote_ip', default='129.128.159.22', type=str)
    parser.add_argument('--port', default=9876, type=int)
    parser.add_argument('--mode', default='e', type=str, help="Modes in ['r', 'o', 'ro', 'e'] ")
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
    elif args.mode == 'ro':
        mode = MODE.ONBOARD_REMOTE
    elif args.mode == 'e':
        mode = MODE.EVALUATION
    else:
        raise  NotImplementedError()

    if args.device is '':
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    args.work_dir += f'/results/{args.env_name}_' \
                     f'act_dt={args.dt}_' \
                     f'seed={args.seed}/'

    args.model_dir = args.work_dir+'model'

    if mode == MODE.LOCAL_ONLY:
        utils.make_dir(args.work_dir)
        utils.make_dir(args.model_dir)
        L = Logger(args.work_dir, use_tb=args.save_tb)

        with open(os.path.join(args.work_dir, 'args.json'), 'w') as f:
            json.dump(vars(args), f, sort_keys=True, indent=4)


    env = ReacherEnvMinTime(
            setup="UR5_2D_V2",
            host=args.ur5_ip,
            dof=2,
            control_type="velocity",
            target_type="position",
            reset_type="zero",
            derivative_type="none",
            deriv_action_max=5,
            first_deriv_max=2,
            accel_max=1,
            speed_max=1.0,
            speedj_a=2.0,
            episode_length_time=args.episode_length_time,
            episode_length_step=None,
            actuation_sync_period=1,
            dt=args.dt,
            run_mode="multiprocess",
            rllab_box=False,
            movej_t=2.0,
            delay=0.0
        )
    env = NormalizedEnv(env)
    env.start()

    utils.set_seed_everywhere(args.seed, env)

    state = env.reset()
    args.image_shape = (0,0,0)
    args.proprioception_shape = env.observation_space.shape
    args.action_shape = env.action_space.shape
    args.net_params = config

    # create local agent
    episode_length_step = int(args.episode_length_time / args.dt)
    agent = OnboardWrapper(episode_length_step, mode, remote_ip=args.remote_ip, port=args.port)
    agent.send_data(args)
    agent.init_performer(SACRADPerformer, args)
    agent.init_learner(SACRADLearner, args, agent.performer)
    
    # sync initial weights with remote
    agent.apply_remote_policy(block=True)

    if args.load_model > -1:
        agent.load_policy_from_file(args.model_dir, args.load_model)

    episode, episode_reward, episode_step, done = 0, 0, 0, True
    
    # First inference took a while (~1 min), do it before the agent-env interaction loop
    if mode != MODE.REMOTE_ONLY:
        agent.performer.sample_action((None, state), args.init_steps+1)

    if mode == MODE.EVALUATION and args.load_model > -1:
        args.init_steps = 0

    agent.send_init_ob((None, state))
    start_time = time.time()
    for step in range(args.num_steps + args.init_steps):
        # sample action for data collection
        action = agent.sample_action((None, state), step)

        # step in the environment
        next_state, reward, done, _ = env.step(action)

        episode_reward += reward
        episode_step += 1
        done = False if episode_step==episode_length_step else done
        agent.push_sample((None, state), action, reward, (None, next_state), done)

        if done or episode_step==episode_length_step:
            if mode == MODE.LOCAL_ONLY:
                L.log('train/duration', time.time() - start_time, step)
                L.log('train/episode_reward', episode_reward, step)
                L.log('train/episode', episode, step)
                L.dump(step)

            next_state = env.reset()
            agent.send_init_ob((None, next_state))
            episode_reward = 0
            episode_step = 0
            episode += 1
            
            start_time = time.time()
        
        stat = agent.update_policy(step)
        if stat is not None:
            for k, v in stat.items():
                L.log(k, v, step)
        
        state = next_state

        if args.save_model and (step+1) % args.save_model_freq == 0:
            agent.save_policy_to_file(args.model_dir, step)

    if args.save_model:
        agent.save_policy_to_file(args.model_dir, step)

    # Clean up
    agent.close()
    env.close()
    print('Training finished')

if __name__ == '__main__':
    main()