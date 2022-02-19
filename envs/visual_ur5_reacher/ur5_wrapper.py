import time
import copy
import numpy as np
import torch.multiprocessing as mp
import cv2
import matplotlib.pyplot as plt
import matplotlib as mpl
from multiprocessing import Process, Value, Manager
import gym
from envs.visual_ur5_reacher.reacher_env import ReacherEnv
from senseact.utils import tf_set_seeds, NormalizedEnv
import argparse
import multiprocessing as mp


def make_env(setup='Visual_UR5',
             ip='129.128.159.210',
             seed=9,
             camera_id=0,
             image_width=160,
             image_height=120,
             target_type='stationary',
             image_history=3,
             joint_history=1,
             episode_length=4.0,
             dt=0.04):
    # state
    np.random.seed(seed)
    rand_state = np.random.get_state()
    # Create Visual UR5 Reacher environment
    env = ReacherEnv(
            setup=setup,
            host=ip,
            dof=5,
            camera_id=camera_id,
            image_width=image_width,
            image_height=image_height,
            channel_first=True,
            control_type="velocity",
            target_type=target_type,
            image_history=image_history,
            joint_history=joint_history,
            reset_type="zero",
            reward_type="dense",
            derivative_type="none",
            deriv_action_max=5,
            first_deriv_max=2,
            accel_max=1.4,
            speed_max=0.3,
            speedj_a=1.4,
            episode_length_time=episode_length,
            episode_length_step=None,
            actuation_sync_period=1,
            dt=dt,
            run_mode="multiprocess",
            rllab_box=False,
            movej_t=2.0,
            delay=0.0,
            random_state=rand_state
        )
    env = NormalizedEnv(env)
    env.start()
    return env

class UR5Wrapper():
    def __init__(self,
                 setup='Visual_UR5',
                 ip='129.128.159.210',
                 seed=9,
                 camera_id=0,
                 image_width=160,
                 image_height=120,
                 target_type='stationary',
                 image_history=3,
                 joint_history=1,
                 episode_length=4.0,
                 dt=0.04,
                 ignore_joint=False,
                 ):
        self.env = make_env(
                        setup,
                        ip,
                        seed,
                        camera_id,
                        image_width,
                        image_height,
                        target_type,
                        image_history,
                        joint_history,
                        episode_length,
                        dt,
                        )

        self.observation_space = self.env.observation_space['image']
        self.ignore_joint = ignore_joint
        if ignore_joint:
            self.state_space = gym.spaces.Box(low=0, high=1., shape=(0, ), dtype=np.float32)
            pass
        else:
            self.state_space = self.env.observation_space['joint']

        self.action_space = self.env.action_space

        self.radius=7
        self.width=160
        self.height=90
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

    def step(self, action):
        obs_dict, reward, done, _ = self.env.step(action)
        if self.ignore_joint:
            return obs_dict['image'], None, reward, done, _
        else:
            return obs_dict['image'], obs_dict['joint'], reward, done, _

    def reset(self):
        x, y = np.random.random(2)
        self.target.set_center((self.radius + x * (self.width - 2 * self.radius),
                self.radius + y * (self.height - 2 * self.radius)))

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        time.sleep(0.032)

        obs_dict = self.env.reset()
        if self.ignore_joint:
            return obs_dict['image'], None
        else:
            return obs_dict['image'], obs_dict['joint']

    def terminate(self):
        self.env.terminate()

if __name__ == '__main__':
    pass

