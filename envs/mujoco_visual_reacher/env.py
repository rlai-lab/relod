import numpy as np
import cv2
import gym
from gym.spaces import Box
import multiprocessing as mp
from mujoco_py import GlfwContext
# GlfwContext(offscreen=True)
from collections import deque

class ReacherWrapper(gym.Wrapper):
    def __init__(self, tol, image_shape=(0, 0, 0), image_period=None, reward_scale=1.0, use_ground_truth=False):
        super().__init__(gym.make('Reacher-v2').unwrapped)
        self._tol = tol
        self._image_period = image_period
        self._reward_scale = reward_scale
        print('reward_scale:', reward_scale)
        self._use_ground_truth = use_ground_truth
        print('use ground truth:', self._use_ground_truth)
        
        self._use_image = False
        if image_shape != (0, 0, 0):
            self._image_buffer = deque([], maxlen=image_shape[0]//3)
            self._use_image = True
            print('time period:', image_period)

        self.image_space = Box(low=0, high=255, shape=image_shape)

        # remember to reset 
        self._latest_image = None
        self._reset = False
        self._epi_step = 0
        if not self._use_ground_truth and not self._use_image:
            print("warning: no target in state and image.")
            input("press any key to continue...")

    @property
    def proprioception_space(self):
        if self._use_ground_truth:
            return self.env.observation_space
        
        low = list(self.env.observation_space.low[0:4]) + list(self.env.observation_space.low[6:8])
        high = list(self.env.observation_space.high[0:4]) + list(self.env.observation_space.high[6:8])
        
        return Box(np.array(low), np.array(high))

    def step(self, a):
        assert self._reset

        ob, _, done, info = self.env.step(a)
        ob = self._get_ob(ob)
        self._epi_step += 1

        dist_to_target = -info["reward_dist"]

        reward = -1 * self._reward_scale
        if dist_to_target <= self._tol:
            info['reached'] = True
            done = True

        if self._use_image and (self._epi_step % self._image_period) == 0:
            new_img = self._get_new_img()
            self._image_buffer.append(new_img)
            self._latest_image = np.concatenate(self._image_buffer, axis=0)

        if done:
            self._reset = False

        return self._latest_image, ob, reward, done, info

    def reset(self):
        ob = self.env.reset()
        ob = self._get_ob(ob)

        if self._use_image:
            new_img = self._get_new_img()
            for _ in range(self._image_buffer.maxlen):
                self._image_buffer.append(new_img)

            self._latest_image = np.concatenate(self._image_buffer, axis=0)
        
        self._reset = True
        self._epi_step = 0
        return self._latest_image, ob

    def _get_new_img(self):
        img = self.env.render(mode='rgb_array')
        img = img[150:400, 50:450, :]
        img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
        img = np.transpose(img, [2, 0, 1]) # c, h, w

        return img
    
    def _get_ob(self, ob):
        if self._use_ground_truth:
            return ob

        return np.array(list(ob[0:4])+list(ob[6:8]))

    def close(self):
        super().close()
        
        del self

if __name__ == '__main__':
    import torch
    print(torch.__version__)
    env = ReacherWrapper(0.072, (9, 125, 200), image_period = 3)
    img, ob = env.reset()
    img = np.transpose(img, [1, 2, 0])
    cv2.imshow('', img[:,:,6:9])
    cv2.waitKey(0)

    waitKey = 1
    while True:
        a = env.action_space.sample()
        img, ob, reward, done, info = env.step(a)
        print(ob)
        img = np.transpose(img, [1, 2, 0])
        cv2.imshow('', img[:,:,6:9])
        cv2.waitKey(waitKey)
        if done:
            env.reset()
            waitKey = 0