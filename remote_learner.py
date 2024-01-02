import torch
import argparse
from relod.algo.remote_wrapper import RemoteWrapper
from relod.algo.sac_rad_agent import SACRADLearner, SACRADPerformer
from relod.logger import Logger
import time
import relod.utils as utils
import os
import cv2
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()

    # agent
    parser.add_argument('--port', default=9876, type=int)
    # misc
    parser.add_argument('--device', default='cuda:0', type=str)

    args = parser.parse_args()
    return args

def main():
    server_args = parse_args()

    agent = RemoteWrapper(port=server_args.port)
    args = agent.recv_data()

    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.return_dir, exist_ok=True)
    L = Logger(args.return_dir, use_tb=args.save_tb)

    if args.save_image:
        os.makedirs(args.image_dir, exist_ok=False)

    agent.init_performer(SACRADPerformer, args)
    agent.init_learner(SACRADLearner, args, agent.performer)

    # sync initial weights with the local-agent
    agent.send_policy()

    utils.set_seed_everywhere(args.seed)

    if server_args.device is '':
        args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    else:
        args.device = server_args.device

    episode_length_step = int(args.episode_length_time / args.dt)
    
    # Experiment block starts
    experiment_done = False
    total_steps = 0
    sub_epi = 0
    returns = []
    epi_lens = []
    start_time = time.time()
    print(f'Experiment starts at: {start_time}')

    while not experiment_done:
        agent.learner.pause_update()
        (image, propri) = agent.receive_init_ob()
        agent.learner.resume_update()
        ret = 0
        epi_steps = 0
        sub_steps = 0
        epi_done = 0
        if args.save_image:
            episode_image_dir = args.image_dir+f'/episode={len(returns)+1}/'
            os.makedirs(episode_image_dir, exist_ok=False)

        epi_start_time = time.time()
        while not experiment_done and not epi_done:
            if args.save_image:
                image_to_show = np.transpose(image, [1, 2, 0])
                image_to_show = image_to_show[:,:,-3:]
                cv2.imwrite(episode_image_dir+f'{epi_steps}.png', image_to_show)

            # select an action
            action = agent.sample_action((image, propri))
            
            # receive sample
            (reward, (next_image, next_propri), epi_done, kwargs) = agent.receive_sample_from_onboard()

            # store
            agent.push_sample((image, propri), action, reward, (next_image, next_propri), epi_done, **kwargs)

            stat = agent.update_policy(total_steps)
            if stat is not None:
                for k, v in stat.items():
                    L.log(k, v, total_steps)

            image = next_image
            propri = next_propri

            # Log
            total_steps += 1
            ret += reward
            epi_steps += 1
            sub_steps += 1

            if args.save_model and total_steps % args.save_model_freq == 0:
                agent.save_policy_to_file(args.model_dir, total_steps)

            if total_steps > args.init_steps and total_steps % args.update_every == 0:
                agent.send_policy()

            if not epi_done and sub_steps >= episode_length_step: # set time out here
                sub_steps = 0
                sub_epi += 1
                ret += args.reset_penalty_steps * args.reward
                print(f'Sub episode {sub_epi} done.')
                agent.learner.pause_update()
                (image, propri) = agent.receive_init_ob()
                agent.learner.resume_update()
            
            experiment_done = total_steps >= args.env_steps
        
        # save the last image
        if args.save_image:
            image_to_show = np.transpose(image, [1, 2, 0])
            image_to_show = image_to_show[:,:,-3:]
            cv2.imwrite(episode_image_dir+f'{epi_steps}.png', image_to_show)

        if epi_done: # episode done, save result
            returns.append(ret)
            epi_lens.append(epi_steps)
            utils.save_returns(args.return_dir+'/return.txt', returns, epi_lens)

            L.log('train/duration', time.time() - epi_start_time, total_steps)
            L.log('train/episode_reward', ret, total_steps)
            L.log('train/episode', len(returns), total_steps)
            L.dump(total_steps)
            if args.plot_learning_curve:
                utils.show_learning_curve(args.return_dir+'/learning curve.png', returns, epi_lens, xtick=args.xtick)

    duration = time.time() - start_time
    agent.learner.pause_update()
    agent.save_policy_to_file(args.model_dir, total_steps)

    agent.close()
    
    # always show a learning curve at the end
    utils.show_learning_curve(args.return_dir+'/learning curve.png', returns, epi_lens, xtick=args.xtick)
    print(f"Finished in {duration}s")

if __name__ == '__main__':
    main()
