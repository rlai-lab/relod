"""
DeepMind control suite tasks implemented in minimum-time formulation
- https://github.com/deepmind/dm_control/tree/main/dm_control
"""

import cv2
import torch
import dm_env
import os

import numpy as np
import matplotlib.pyplot as plt

from dm_control.suite.utils import randomizers
from dm_control.suite.reacher import Reacher, Physics
from dm_control.rl.control import flatten_observation
from dm_control import suite
from dm_control.rl import control
from dm_control.suite import common
from dm_control.utils import io as resources
from gym.spaces import Box
from collections import deque


class Observation:
    def __init__(self):
        self.images = None
        self.metadata = None
        self.proprioception = None


class ReacherWrapper:
    """ Minimum-time variant of reacher env with 3 modes: Easy, Hard,  """
    def __init__(self, seed, penalty=-1, mode="easy", use_image=False, img_history=3):
        """ Outputs state transition data as torch arrays """
        assert mode in ["easy", "hard", "torture"]

        if mode == "torture":
            physics = Physics.from_xml_string(*ReacherWrapper.get_modified_model_and_assets())
            task = Reacher(target_size=.001, random=seed)
            self.env = control.Environment(physics, task, time_limit=float('inf'), **{})
        else:
            self.env = suite.load(domain_name="reacher", task_name=mode, task_kwargs={'random': seed, 'time_limit': float('inf')})

        self._obs_dim = 4 if use_image else 6
        self._action_dim = 2
        
        self.reward = penalty
        self._use_image = use_image
        
        if use_image:
            self._image_buffer = deque([], maxlen=img_history)
            print(f'Visual dm reacher {mode}')
        else:
            print(f'Non visual dm reacher {mode}')

    def make_obs(self, x):
        obs = np.zeros(self._obs_dim, dtype=np.float32)
        obs[:2] = x.observation['position'].astype(np.float32)
        obs[2:4] = x.observation['velocity'].astype(np.float32)

        if not self._use_image: # this should be inferred from image
            obs[4:6] = x.observation['to_target'].astype(np.float32)
        
        return obs

    @staticmethod
    def get_modified_model_and_assets():
        """Returns a tuple containing the model XML string and a dict of assets."""
        PARENT_DIR = os.path.dirname(os.path.dirname(__file__))
        return resources.GetResource(os.path.join(PARENT_DIR, 'envs/reacher_small_finger.xml')), common.ASSETS

    def _get_new_img(self):
        img = self.env.physics.render()
        img = img[85:155, 110:210, :]
        img = np.transpose(img, [2, 0, 1])  # c, h, w
        return img

    def _initialize_episode_random_agent_only(self):
        """Sets the state of the environment at the start of each episode."""
        self.env.physics.named.model.geom_size['target', 0] = self.env.task._target_size
        randomizers.randomize_limited_and_rotational_joints(self.env.physics, self.env.task.random)

        super(Reacher, self.env.task).initialize_episode(self.env.physics)

    def _reset_agent_only(self):
        """Starts a new episode and returns the first `TimeStep`."""
        self.env._reset_next_step = False
        self.env._step_count = 0
        with self.env.physics.reset_context():
            self._initialize_episode_random_agent_only()

        observation = self.env.task.get_observation(self.env.physics)
        if self.env._flat_observation:
            observation = flatten_observation(observation)

        return dm_env.TimeStep(
            step_type=dm_env.StepType.FIRST,
            reward=None,
            discount=None,
            observation=observation)

    def reset(self, randomize_target=True):
        if self._use_image:
            obs = Observation()
            if randomize_target:
                obs.proprioception = self.make_obs(self.env.reset())
            else:
                obs.proprioception = self.make_obs(self._reset_agent_only())

            new_img = self._get_new_img()
            for _ in range(self._image_buffer.maxlen):
                self._image_buffer.append(new_img)

            obs.images = np.concatenate(self._image_buffer, axis=0)
        else:
            if randomize_target:
                obs = self.make_obs(self.env.reset())
            else:
                obs = self.make_obs(self._reset_agent_only())

        return obs

    def step(self, action):
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy().flatten()

        x = self.env.step(action)

        reward = self.reward
        done = x.reward
        info = {}

        if self._use_image:
            next_obs = Observation()
            next_obs.proprioception = self.make_obs(x)
            new_img = self._get_new_img()
            self._image_buffer.append(new_img)
            next_obs.images = np.concatenate(self._image_buffer, axis=0)
        else:
            next_obs = self.make_obs(x)
            
        return next_obs, reward, done, info

    @property
    def image_space(self):
        if not self._use_image:
            raise AttributeError(f'use_image={self._use_image}')

        image_shape = (3 * self._image_buffer.maxlen, 70, 100)
        return Box(low=0, high=255, shape=image_shape)

    @property
    def observation_space(self):
        return Box(shape=(self._obs_dim,), high=10, low=-10)

    @property
    def proprioception_space(self):
        if not self._use_image:
            raise AttributeError(f'use_image={self._use_image}')
        
        return self.observation_space

    @property
    def action_space(self):
        return Box(shape=(self._action_dim,), high=1, low=-1)

    def render(self):
        self.env.render()


if __name__ == '__main__':
    env = ReacherWrapper(seed=42, penalty=-1, mode="hard", use_image=True)
    obs = env.reset()
    img = np.transpose(obs.images, [1, 2, 0])
    cv2.imshow('', img[:,:,6:9])
    cv2.waitKey(0)

    waitKey = 1
    while True:
        a = env.action_space.sample()
        next_obs, reward, done, info = env.step(a)
        # print(obs.proprioception)
        img = np.transpose(obs.images, [1, 2, 0])
        cv2.imshow('', img[:,:,6:9])
        cv2.waitKey(waitKey)
        obs = next_obs
        if done:
            env.reset()
            waitKey = 0
            