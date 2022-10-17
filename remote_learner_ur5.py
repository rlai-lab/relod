import torch
import argparse
from relod.algo.remote_wrapper import RemoteWrapper
from relod.algo.sac_rad_agent import SACRADLearner, SACRADPerformer
from relod.logger import Logger
import time
import relod.utils as utils
import os

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

class MonitorTarget:
    def __init__(self):
        self.radius=7
        self.width=160
        self.height=90
        self.margin = 20
        mpl.rcParams['toolbar'] = 'None'
        plt.ion()
        self.fig = plt.figure()
        plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
        self.fig.canvas.toolbar_visible = False
        self.ax = plt.axes(xlim=(0, self.width), ylim=(0, self.height))
        self.target = plt.Circle((0, 0), self.radius, color='red')
        self.ax.add_patch(self.target)
        plt.axis('off')

        figManager = plt.get_current_fig_manager()
        figManager.full_screen_toggle()

    def reset_plot(self):
        x, y = np.random.random(2)
        self.target.set_center(
            (self.radius + self.margin + x * (self.width - 2*self.radius - 2*self.margin),
             self.radius + self.margin + y * (self.height - 2*self.radius - 2*self.margin))
        )
        print()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        time.sleep(0.032)

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

    # Monitor
    mt = MonitorTarget()
    mt.reset_plot()
    mt.reset_plot()
    mt.reset_plot()
    mt.reset_plot()
    mt.reset_plot()

    agent = RemoteWrapper(port=server_args.port)
    args = agent.recv_data()

    utils.make_dir(args.work_dir)

    model_dir = utils.make_dir(os.path.join(args.work_dir, 'model'))
    args.model_dir = model_dir
    
    agent.init_performer(SACRADPerformer, args)
    agent.init_learner(SACRADLearner, args, agent.performer)

    # sync initial weights with oboard
    agent.send_policy()

    utils.set_seed_everywhere(args.seed)

    if server_args.device is '':
        args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    else:
        args.device = server_args.device

    L = Logger(args.work_dir, use_tb=args.save_tb)

    episode, episode_reward, episode_step, done = 0, 0, 0, True
    episode_length_step = int(args.episode_length_time / args.dt)
    (image, propri) = agent.receive_init_ob()
    start_time = time.time()
    for step in range(args.env_steps + args.init_steps):
        action = agent.sample_action((image, propri), step)
        
        (reward, (next_image, next_propri), done, kwargs) = agent.receive_sample_from_onboard()
        
        episode_reward += reward
        episode_step += 1

        agent.push_sample((image, propri), action, reward, (next_image, next_propri), done, **kwargs)

        if done or (episode_step == episode_length_step): # set time out here
            L.log('train/duration', time.time() - start_time, step)
            L.log('train/episode_reward', episode_reward, step)
            L.dump(step)

            mt.reset_plot()
            (next_image, next_propri) = agent.receive_init_ob()
            episode_reward = 0
            episode_step = 0
            episode += 1
            L.log('train/episode', episode, step)
            start_time = time.time()
            
        stat = agent.update_policy(step)
        if stat is not None:
            for k, v in stat.items():
                L.log(k, v, step)

        (image, propri) = (next_image, next_propri)

        if args.save_model and (step+1) % args.save_model_freq == 0:
            agent.save_policy_to_file(args.model_dir, step)
        
        if step > args.init_steps and (step+1) % args.update_every == 0:
            agent.send_policy()

    if args.save_model:
        agent.save_policy_to_file(args.model_dir, step)
    agent.close()
    print('Train finished')

if __name__ == '__main__':
    main()
