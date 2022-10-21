import torch
import argparse
import relod.utils as utils
import time
import numpy as np
import cv2
from tqdm import tqdm

from relod.logger import Logger
from relod.algo.comm import MODE
from relod.algo.local_wrapper import LocalWrapper
from relod.algo.sac_rad_agent import SACRADLearner, SACRADPerformer
from relod.envs.visual_franka_min_time_reacher.env import FrankaPanda_Visual_Min_Reacher

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
    parser = argparse.ArgumentParser(description='Local Franka Visual Min Task Init Policy')
    # environment
    parser.add_argument('--setup', default='Visual-Franka')
    parser.add_argument('--env', default='Visual_Franka_Min_Task_Init_Policy', type=str)
    parser.add_argument('--image_width', default=160, type=int)
    parser.add_argument('--image_height', default=90, type=int)
    parser.add_argument('--target_type', default='size', type=str)
    parser.add_argument('--random_action_repeat', default=1, type=int)
    parser.add_argument('--agent_action_repeat', default=1, type=int)
    parser.add_argument('--image_history', default=3, type=int)
    parser.add_argument('--joint_history', default=1, type=int)
    parser.add_argument('--ignore_joint', default=False, action='store_true')
    parser.add_argument('--episode_length_time', default=6.0, type=float)
    parser.add_argument('--dt', default=0.04, type=float)
    
    parser.add_argument('--size_tol', default=0.12, type=float)
    parser.add_argument('--center_tol', default=0.1, type=float)
    parser.add_argument('--reward_tol', default=2.0, type=float)
    parser.add_argument('--reset_penalty_steps', default=70, type=int)
    parser.add_argument('--reward', default=-1, type=float)
    parser.add_argument('--N', required=True, type=int, help="# timesteps for the run")
    # replay buffer
    parser.add_argument('--replay_buffer_capacity', default=100000, type=int)
    parser.add_argument('--rad_offset', default=0.01, type=float)
    # train
    parser.add_argument('--init_steps', default=5000, type=int) 
    parser.add_argument('--env_steps', default=200000, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--async_mode', default=True, action='store_true')
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
    parser.add_argument('--mode', default='l', type=str, help="Modes in ['r', 'l', 'rl', 'e'] ")
    # misc
    parser.add_argument('--description', default='test new remote script', type=str)
    parser.add_argument('--seed', default=3, type=int)
    parser.add_argument('--work_dir', default='.', type=str)
    parser.add_argument('--save_tb', default=False, action='store_true')
    parser.add_argument('--save_model', default=False, action='store_true')
    parser.add_argument('--plot_learning_curve', default=False, action='store_true')
    parser.add_argument('--xtick', default=1200, type=int)
    parser.add_argument('--display_image', default=True, action='store_true')
    parser.add_argument('--save_image', default=True, action='store_true')
    parser.add_argument('--save_model_freq', default=10000, type=int)
    parser.add_argument('--load_model', default=-1, type=int)
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--lock', default=False, action='store_true')

    args = parser.parse_args()
    assert args.mode in ['r', 'l', 'rl', 'e']
    assert args.reward < 0 and args.reset_penalty_steps >= 0

    return args


def main():
    args = parse_args()

    mode = MODE.LOCAL_ONLY

    if args.device is '':
        args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    env = FrankaPanda_Visual_Min_Reacher(
        image_width=args.image_width,
        image_height=args.image_height,
        camera_index=1, 
        size_tol=args.size_tol)

    utils.set_seed_everywhere(args.seed, env)
    
    image, prop = env.reset()

    input('go?')

    
    args.image_shape = env.image_space.shape
    args.proprioception_shape = env.observation_space.shape

    print(args.image_shape, image.shape)
    print(args.proprioception_shape, prop.shape)

    args.action_shape = env.action_space.shape
    args.env_action_space = env.action_space
    args.net_params = config

    episode_length_step = 1000000  
    agent = LocalWrapper(episode_length_step, mode, remote_ip=args.remote_ip, port=args.port)
    
    agent.send_data(args)
    agent.init_performer(SACRADPerformer, args)
    agent.init_learner(SACRADLearner, args, agent.performer)

    # sync initial weights with remote
    agent.apply_remote_policy(block=True)

    agent.send_init_ob((image, prop))

    #timeouts = [1, 2, 5, 10, 25, 50, 100, 500, 1000, 5000]
    timeouts = [75, 150, 750]

    steps_record = open(f"results/{args.env}_steps_record.txt", 'w')
    hits_record = open(f"results/{args.env}_random_stat.txt", 'w')

    for timeout in tqdm(timeouts):
        for seed in range(5):
            args.seed = seed
            utils.set_seed_everywhere(args.seed, env)

            steps_record.write(f"timeout={timeout}, seed={seed}: ")
            # Experiment
            hits = 0
            steps = 0
            epi_steps = 0
            image, prop = env.reset() 
            while steps < args.N:
                action = agent.sample_action((image, prop))

                # Receive reward and next state            
                _, _, _, done, _ = env.step(action)
                
                steps += 1
                epi_steps += 1

                # Termination
                if done or epi_steps == timeout:
                    t1 = time.time()
                    env.reset()
                    reset_step = (time.time() - t1) // args.dt
                    epi_steps = 0

                    if done:
                        hits += 1
                    else:
                        steps += reset_step
                        
                    steps_record.write(str(steps)+', ')
                    steps_record.flush()

            steps_record.write('\n')
            hits_record.write(f"timeout={timeout}, seed={seed}: {hits}\n")
            hits_record.flush()
    
    env.reset()
    steps_record.close()
    hits_record.close()

if __name__ == '__main__':
    main()
