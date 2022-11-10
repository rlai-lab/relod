import torch
import argparse
import relod.utils as utils
import time
import os

from relod.logger import Logger
from relod.algo.comm import MODE
from relod.algo.local_wrapper import LocalWrapper
from relod.algo.sac_rad_agent import SACRADLearner, SACRADPerformer

from senseact.utils import NormalizedEnv
from relod.envs.create2_visual_reacher.env import Create2VisualReacherEnv
from tqdm import tqdm
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
        [-1, 1024],
        [1024, 1024],
        [1024, -1]
    ],
}

def parse_args():
    parser = argparse.ArgumentParser()
    # environment
    parser.add_argument('--env', default='create2_visual_reacher', type=str)
    parser.add_argument('--episode_length_time', default=30.0, type=float)
    parser.add_argument('--dt', default=0.045, type=float)
    parser.add_argument('--image_height', default=120, type=int)
    parser.add_argument('--image_width', default=160, type=int)
    parser.add_argument('--stack_frames', default=3, type=int)
    parser.add_argument('--camera_id', default=0, type=int)
    parser.add_argument('--min_target_size', default=0.2, type=float)
    parser.add_argument('--reset_penalty_steps', default=67, type=int)
    parser.add_argument('--reward', default=-1, type=float)
    # replay buffer
    parser.add_argument('--replay_buffer_capacity', default=100000, type=int)
    parser.add_argument('--rad_offset', default=0.01, type=float)
    # train
    parser.add_argument('--init_steps', default=1000, type=int)
    parser.add_argument('--env_steps', default=150000, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--sync_mode', default=False, action='store_true')
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
    parser.add_argument('--remote_ip', default='192.168.0.12', type=str)
    parser.add_argument('--port', default=9876, type=int)
    parser.add_argument('--mode', default='e', type=str, help="Modes in ['r', 'l', 'rl', 'e'] ")
    # misc
    parser.add_argument('--run_type', default='experiment', type=str)
    parser.add_argument('--description', default='', type=str)
    parser.add_argument('--seed', default=2, type=int)
    parser.add_argument('--work_dir', default='results/', type=str)
    parser.add_argument('--save_tb', default=False, action='store_true')
    parser.add_argument('--save_model', default=False, action='store_true')
    parser.add_argument('--plot_learning_curve', default=True, action='store_true')
    parser.add_argument('--xtick', default=1500, type=int)
    parser.add_argument('--save_image', default=True, action='store_true')
    parser.add_argument('--save_model_freq', default=10000, type=int)
    parser.add_argument('--load_model', default=-1, type=int)
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--lock', default=False, action='store_true')
    args = parser.parse_args()
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

    args.work_dir += f'/{args.env}/visual/timeout={args.episode_length_time:.0f}/seed={args.seed}'
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
    agent = LocalWrapper(episode_length_step, mode, remote_ip=args.remote_ip, port=args.port)
    agent.send_data(args)
    agent.init_performer(SACRADPerformer, args)
    agent.init_learner(SACRADLearner, args, agent.performer)

    # sync initial weights with remote
    agent.apply_remote_policy(block=True)

    if args.load_model > -1:
        agent.load_policy_from_file(args.model_dir, args.load_model)

    # branch here
    if args.run_type == 'init_policy_test':
        env.close()
        run_init_policy_test(agent, args)
        return

    # Experiment block starts
    experiment_done = False
    total_steps = 0
    sub_epi = 1
    returns = []
    epi_lens = []
    start_time = time.time()
    print(f'Experiment starts at: {start_time}')
    while not experiment_done:
        (image, propri) = env.reset()

        # First inference took a while (~1 min), do it before the agent-env interaction loop
        if mode != MODE.REMOTE_ONLY and total_steps == 0:
            agent.performer.sample_action((image, propri))
            agent.performer.sample_action((image, propri))
            agent.performer.sample_action((image, propri))

        agent.send_init_ob((image, propri))
        ret = 0
        epi_steps = 0
        sub_steps = 0
        epi_done = 0
        if (mode == MODE.LOCAL_ONLY or mode == MODE.EVALUATION) and args.save_image:
            episode_image_dir = args.image_dir+f'/episode={len(returns)+1}/'
            os.makedirs(episode_image_dir, exist_ok=False)

        epi_start_time = time.time()
        while not experiment_done and not epi_done:
            if args.display_image or ((mode == MODE.LOCAL_ONLY or mode == MODE.EVALUATION) and args.save_image):
                image_to_show = np.transpose(image, [1, 2, 0])
                image_to_show = image_to_show[:,:,-3:]
                if (mode == MODE.LOCAL_ONLY or mode == MODE.EVALUATION) and args.save_image:
                    cv2.imwrite(episode_image_dir+f'sub_epi={sub_epi}-epi_step={epi_steps}.png', image_to_show)
                if args.display_image:
                    cv2.imshow('raw', image_to_show)
                    cv2.waitKey(1)

            # select an action
            action = agent.sample_action((image, propri))

            # step in the environment
            (next_image, next_propri), reward, epi_done, _ = env.step(action)

            # store
            agent.push_sample((image, propri), action, reward, (next_image, next_propri), epi_done)

            agent.update_policy(total_steps)
            
            image = next_image
            propri = next_propri

            # Log
            total_steps += 1
            ret += reward
            epi_steps += 1
            sub_steps += 1

            if args.save_model and total_steps % args.save_model_freq == 0:
                agent.save_policy_to_file(args.model_dir, total_steps)

            if not epi_done and sub_steps >= episode_length_step: # set timeout here
                sub_steps = 0
                ret += args.reset_penalty_steps * args.reward
                total_steps += args.reset_penalty_steps
                print(f'Sub episode {sub_epi} done.')

                (image, propri) = env.reset()
                agent.send_init_ob((image, propri))
                sub_epi += 1
            
            experiment_done = total_steps >= args.env_steps

        # save the last image
        if (mode == MODE.LOCAL_ONLY or mode == MODE.EVALUATION) and args.save_image:
            image_to_show = np.transpose(image, [1, 2, 0])
            image_to_show = image_to_show[:,:,-3:]
            cv2.imwrite(episode_image_dir+f'sub_epi={sub_epi}-epi_step={epi_steps}.png', image_to_show)

        if epi_done: # episode done, save result
            returns.append(ret)
            epi_lens.append(epi_steps)
            if mode != MODE.EVALUATION:
                utils.save_returns(args.return_dir+'/return.txt', returns, epi_lens)

            if mode == MODE.LOCAL_ONLY:
                L.log('train/duration', time.time() - epi_start_time, total_steps)
                L.log('train/episode_reward', ret, total_steps)
                L.log('train/episode', len(returns), total_steps)
                L.dump(total_steps)
                if args.plot_learning_curve:
                    utils.show_learning_curve(args.return_dir+'/learning curve.png', returns, epi_lens, xtick=args.xtick)
            
            sub_epi += 1

    duration = time.time() - start_time
    agent.save_policy_to_file(args.model_dir, total_steps)

    # Clean up
    agent.close()
    env.close()

    # always show a learning curve at the end
    if mode == MODE.LOCAL_ONLY:
        utils.show_learning_curve(args.return_dir+'/learning curve.png', returns, epi_lens, xtick=args.xtick)
    print(f"Finished in {duration}s")

def run_init_policy_test(agent, args):
    timeouts = [int(args.episode_length_time/args.dt)]
    args.init_steps = 100000000
    args.env_steps = 20000
    steps_record = open(f"{args.env}_steps_record.txt", 'w')
    hits_record = open(f"{args.env}_random_stat.txt", 'w')

    if not 'conv' in config:
        image_shape = (0, 0, 0)
    else: 
        image_shape = (3*args.stack_frames, args.image_height, args.image_width)

    for timeout in timeouts:
        for seed in tqdm(range(5)):
            args.seed = seed
            env = Create2VisualReacherEnv(
                episode_length_time=args.episode_length_time, 
                dt=args.dt,
                image_shape=image_shape,
                camera_id=args.camera_id,
                min_target_size=args.min_target_size
            )
            env = NormalizedEnv(env)
            utils.set_seed_everywhere(args.seed, None)
            env.start()

            steps_record.write(f"timeout={timeout}, seed={seed}: ")
            steps_record.flush()
            # Experiment
            hits = 0
            steps = 0
            epi_steps = 0
            (image, propri) = env.reset()
            agent.performer.sample_action((image, propri))
            agent.performer.sample_action((image, propri))
            agent.performer.sample_action((image, propri))

            while steps < args.env_steps:
                action = agent.sample_action((image, propri))

                # Receive reward and next state            
                _, _, epi_done, _ = env.step(action)
                
                # print("Step: {}, Next Obs: {}, reward: {}, done: {}".format(steps, next_obs, reward, done))

                # Log
                steps += 1
                epi_steps += 1

                # Termination
                if epi_done or epi_steps == timeout:
                    env.reset()
                        
                    epi_steps = 0

                    if epi_done:
                        hits += 1
                    else:
                        steps += 65
                        
                    steps_record.write(str(steps)+', ')
                    steps_record.flush()

            steps_record.write('\n')
            steps_record.flush()
            hits_record.write(f"timeout={timeout}, seed={seed}: {hits}\n")
            hits_record.flush()
            env.close()
            time.sleep(120)

    steps_record.close()
    hits_record.close()
    agent.close()

if __name__ == '__main__':
    main()
