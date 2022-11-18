import torch
import argparse
import time
import os
import cv2
import random
import threading

import numpy as np
import relod.utils as utils
import matplotlib.pyplot as plt

from datetime import datetime
from relod.logger import Logger
from relod.algo.comm import MODE
from relod.algo.local_wrapper import LocalWrapper
from relod.algo.sac_rad_agent import SACRADLearner, SACRADPerformer
from senseact.utils import NormalizedEnv
from rl_vector.vector.env_color_detector import VectorColorDetector, VectorBallDetector
from rl_suite.plot.plot import smoothed_curve
from rl_vector.egocentric_view import VectorPOV
from tqdm import tqdm
from anki_vector.util import degrees
from sys import platform
if platform == "darwin":    # For MacOS
    import matplotlib as mpl
    mpl.use("TKAgg")


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
    parser.add_argument('--episode_length_time', default=30.0, type=float)
    parser.add_argument('--dt', default=0.1, type=float)        
    parser.add_argument('--timeout', default=300, type=int, help="Timeout for the env")
    parser.add_argument('--robot_serial',default="00902998", type=str, help="Vector serial #")
    parser.add_argument('--object', default="charger", type=str, help="['charger', 'ball']")
    parser.add_argument('--stack_frames', default=4, type=int)
    parser.add_argument('--image_height', default=120, type=int)
    parser.add_argument('--image_width', default=160, type=int)
    parser.add_argument('--reset_penalty_steps', default=67, type=int)
    parser.add_argument('--reward', default=-1, type=float)
    # replay buffer
    parser.add_argument('--replay_buffer_capacity', default=100000, type=int)
    parser.add_argument('--rad_offset', default=0.01, type=float)
    # train
    parser.add_argument('--init_steps', default=1000, type=int)
    parser.add_argument('--env_steps', default=151000, type=int)
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
    parser.add_argument('--remote_ip', default='192.168.0.100', type=str)
    parser.add_argument('--port', default=9876, type=int)
    parser.add_argument('--mode', default='l', type=str, help="Modes in ['r', 'l', 'rl', 'e'] ")
    # misc
    parser.add_argument('--seed', default=3, type=int)
    parser.add_argument('--work_dir', default='.', type=str)
    parser.add_argument('--save_tb', default=False, action='store_true')
    parser.add_argument('--save_model', default=True, action='store_true')
    parser.add_argument('--save_model_freq', default=5000, type=int)
    parser.add_argument('--load_model', default=-1, type=int)
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--lock', default=False, action='store_true')
    parser.add_argument('--save_image', default=False, action='store_true')
    parser.add_argument('--save_buffer', default=True, action='store_true')
    parser.add_argument('--load_buffer', default=False, action='store_true')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    if args.mode == 'l':
        mode = MODE.LOCAL_ONLY
    elif args.mode == 'e':
        mode = MODE.EVALUATION
    else:
        raise NotImplementedError("Only local and evaluation mode supported!")

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

    if args.work_dir == '.':
        run_id = "{}-VectorDetector-{}-{}".format(datetime.now().strftime("%Y%m%d-%H%M%S"), args.object, args.robot_serial)
        args.work_dir += f'/results/{run_id}/seed={args.seed}'

    args.model_dir = args.work_dir + '/models'
    args.return_dir = args.work_dir + '/returns'
    
    args.load_buffer_path = ''
    if args.save_buffer:
        args.save_buffer_path = args.work_dir + "/{}_sac_buffer".format(args.robot_serial)
        utils.make_dir(args.save_buffer_path)

    args.load_buffer_path = ''
    if args.load_buffer:
        args.load_buffer_path =  args.work_dir + "/{}_sac_buffer".format(args.robot_serial)
  
    if args.save_image:
        args.image_dir = args.work_dir + '/images'
        if mode == MODE.LOCAL_ONLY or mode == MODE.EVALUATION:
            os.makedirs(args.image_dir, exist_ok=False)

    if mode == MODE.LOCAL_ONLY:
        os.makedirs(args.work_dir, exist_ok=True)
        os.makedirs(args.model_dir, exist_ok=True)
        os.makedirs(args.return_dir, exist_ok=True)
        L = Logger(args.work_dir, use_tb=args.save_tb)
    
    if not 'conv' in config:
        image_shape = (0, 0, 0)
    else: 
        image_shape = (3*args.stack_frames, args.image_height, args.image_width)

    cfg = {
            "robot_serial": args.robot_serial,
            "dt": args.dt,
            "prox_threshold": 0.1,
            "episode_length_step": args.timeout,
        }

    # Use hsv_threshold_gui.py script to get the hsv mask values
    if args.object == "charger":
        cfg["hsv_mask"] = {"low": [0, 0, 0], "high": [180, 255, 45],}
        cfg["head_angle"] = -1
        cfg["obj_thresh"] = 0.22
        cfg["obj_dist"] = 0.13
        env = VectorColorDetector(cfg=cfg)
    elif args.object == "ball":
        cfg["hsv_mask"] = {"low": [0, 50, 40], "high": [255, 255, 255],}
        cfg["head_angle"] = -15
        cfg["obj_thresh"] = 0.07
        cfg["obj_dist"] = 0.05
        env = VectorBallDetector(cfg=cfg)
    args.robot_cfg = cfg

    # env = NormalizedEnv(env)
    utils.set_seed_everywhere(args.seed, env)

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
        print("Loading model")
        agent.load_policy_from_file(args.model_dir, args.load_model)

    if args.load_buffer:
        # Simple hack to streamline runs
        sleep_time = 20
        for i in range(1, sleep_time+1):
            print("Sleep for {}s to give time for buffer to load".format(sleep_time - i))
            time.sleep(1)
                    
    if mode == MODE.EVALUATION and args.load_model > -1:
        args.init_steps = 0


    # Plotter process
    vp = VectorPOV(dt=env._dt, img_dim=(120, 160, 3), robot_serial=env.robot_serial)
    p = threading.Thread(target=vp.plot, args=())
    p.start()

    # Experiment block starts
    experiment_done = False
    total_steps = 0 

    sub_epi = 0
    returns = []
    epi_lens = []
    start_time = time.time()
    learning_paused = False
    print(f'Experiment starts at: {start_time}')
    while not experiment_done:
        image, propri = env.reset()
        
        # Resume learning if it was paused while charging
        if learning_paused:
            agent._learner.resume_update()
            learning_paused = False
            
        tic = time.time()

        agent.send_init_ob((image, propri))
        ret = 0
        epi_steps = 0
        if args.load_model > 0:
            data = np.loadtxt(os.path.join(args.return_dir, "return.txt"))

            if mode != MODE.EVALUATION:
                returns = list(data[1])
                epi_lens = list(data[0])
                total_steps = int(sum(epi_lens))

        sub_steps = 0
        epi_done = 0

        if (mode == MODE.LOCAL_ONLY or mode == MODE.EVALUATION) and args.save_image:
            episode_image_dir = args.image_dir + f'/episode={len(returns)+1}/'
            os.makedirs(episode_image_dir, exist_ok=False)

        while not experiment_done and not epi_done:
            # Visualizer process
            cv_img = image[9:12, :, :]
            cv_img = np.moveaxis(cv_img, 0, -1).astype(np.uint8)
            # cv_img = np.moveaxis(cv_img, 0, 1).astype(np.uint8)
            with vp._lock:
                vp.img = cv_img
            
            if (mode == MODE.LOCAL_ONLY or mode == MODE.EVALUATION) and args.save_image:
                cv2.imwrite(episode_image_dir + f'sub_epi={sub_epi+len(returns)}-epi_step={epi_steps}.png', cv_img)    

            # select an action
            action = agent.sample_action((image, propri))

            # step in the environment
            next_image, next_propri, reward, epi_done, _ = env.step(action)

            # Vector flipped over
            if env.vector_comm.is_cliff_detected() or env.vector_comm.is_picked_up():
                # Stop the wheels
                env.vector_comm.set_wheel_motors([0, 0])
                time.sleep(0.5)
                env.vector_comm.flip_back()
                # Set head angle
                env.vector_comm.robot.behavior.set_head_angle(degrees(args.robot_cfg["head_angle"]))
                time.sleep(0.5)
                # Set lift height
                env.vector_comm.set_lift_height(1.0)
                time.sleep(0.5)
                reward += -100 * args.dt
                total_steps += 100
                epi_steps += 100

            # Push to replay buffer and make learning update
            if mode != MODE.EVALUATION:
                agent.push_sample((image, propri), action, reward, (next_image, next_propri), epi_done)
                agent.update_policy(total_steps)
            
            image = next_image
            propri = next_propri

            # Log
            total_steps += 1
            ret += reward
            epi_steps += 1
            sub_steps += 1

            if total_steps % 10 == 0:
                print("Step: {}, Obs: {}, Action: {}, Reward: {:.2f}, Done: {}, dt: {:.3f}".format(
                        total_steps, propri[3:5], action, reward, epi_done, time.time()-tic))
            tic = time.time()

            if args.save_model and total_steps % args.save_model_freq == 0:
                if mode != MODE.EVALUATION:
                    agent.save_policy_to_file(args.model_dir, total_steps)
                # Plot
                if returns:
                    plot_rets, plot_x = smoothed_curve(
                            np.array(returns), np.array(epi_lens), x_tick=args.save_model_freq, window_len=args.save_model_freq)
                if len(plot_rets):
                    plt.clf()
                    plt.plot(plot_x, plot_rets)
                    plt.pause(0.001)
                    plt.savefig(args.return_dir+'/learning_curve.png')
               
            if not epi_done and sub_steps >= episode_length_step: # set timeout here
                sub_steps = 0
                sub_epi += 1
                ret += args.reset_penalty_steps * args.reward
                print(f'Sub episode {sub_epi} done.')
                
                # Save buffer when Vector is charging; Pause learning updates to prevent over-fitting
                if env.is_charging_necessary:      
                    if mode != MODE.EVALUATION:              
                        agent.save_buffer()
                    #agent._learner.pause_update()
                    learning_paused = True

                (image, propri) = env.reset()
                agent.send_init_ob((image, propri))
            
            experiment_done = total_steps >= args.env_steps
        
        # save the last image
        if (mode == MODE.LOCAL_ONLY or mode == MODE.EVALUATION) and args.save_image:
            cv_img = image[9:12, :, :]
            cv_img = np.moveaxis(cv_img, 0, -1).astype(np.uint8)
            cv2.imwrite(episode_image_dir+f'sub_epi={sub_epi+len(returns)}-epi_step={epi_steps}.png', cv_img)


        if epi_done: # episode done, save result
            returns.append(ret)
            epi_lens.append(epi_steps)
            print(f'Episode {len(epi_lens)} ended in {epi_steps} steps.')
            if mode != MODE.EVALUATION:
                utils.save_returns(args.return_dir+'/return.txt', returns, epi_lens)

            # Save buffer when Vector is charging; Pause learning updates to prevent over-fitting
            if env.is_charging_necessary:
                if mode != MODE.EVALUATION:                    
                    agent.save_buffer()
                #agent._learner.pause_update()
                learning_paused = True

    duration = time.time() - start_time
    agent.save_policy_to_file(args.model_dir, total_steps)

    # Clean up
    with vp.running.get_lock():
        vp.running.value = 0
    agent.close()
    p.join()    # Visualizer process 
    env.close()
    print(f"Finished in {duration}s")



def run_init_policy_test():
    timeouts = [12, 30]

    args = parse_args()
    cfg = {
        "robot_serial": args.robot_serial,
        "dt": 0.1,
        "prox_threshold": 0.1,
        "episode_length_time": 30,
        "hsv_mask": {
            # For yellow post-it
            # "low": [89, 70, 100],
            # "high": [170, 230, 255],
            # For charger
            "low": [0, 0, 0],
            "high": [180, 255, 45],
            # For Green Ball
            # "low": [0, 50, 40],
            # "high": [255, 255, 255],
        },
        "head_angle": -1, # -12,
        "obj_thresh": 0.24, # 0.07
        "obj_dist": 0.11,
    }


    steps_record = open(f"VectorChargerDetector_steps_record.txt", 'w')
    hits_record = open(f"VectorChargerDetector_random_stat.txt", 'w')

    for seed in range(5):
        for timeout in tqdm(timeouts):        
            np.random.seed(seed)
            random.seed(seed)

            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        
            cfg["episode_length_time"] = timeout
            env = VectorColorDetector(cfg=cfg)
            args.image_shape = env.image_space.shape
            args.proprioception_shape = env.proprioception_space.shape
            args.action_shape = env.action_space.shape
            args.net_params = config
            args.env_action_space = env.action_space            
            performer = SACRADPerformer(args)

            steps_record.write(f"timeout={timeout}, seed={seed}: ")
            # Experiment
            hits = 0
            steps = 0
            epi_steps = 0
            image, propri = env.reset()
            while steps < 20000:
                action = performer.sample_action((image, propri))

                # Receive reward and next state            
                next_image, next_propri, reward, done, _ = env.step(action)
                
                print("Step: {}, Next Obs: {}, reward: {}, done: {}".format(steps, next_propri[3:5], reward, done))

                image = next_image
                propri = next_propri

                # Log
                steps += 1
                epi_steps += 1

                # Termination
                if done or epi_steps == env._episode_length_step: 
                    env.reset()                      
                    epi_steps = 0

                    if done:
                        hits += 1
                    else:
                        steps += 20
                        
                    steps_record.write(str(steps)+', ')

            steps_record.write('\n')
            hits_record.write(f"timeout={timeout}, seed={seed}: {hits}\n")
        
    steps_record.close()
    hits_record.close()

    
if __name__ == '__main__':
    main()
    # run_init_policy_test()
