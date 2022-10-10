import gym
import numpy as np
from gym.spaces import Box

class DummyEnv(gym.Env):
    def __init__(self, image_shape=(0, 0, 0), proprioception_shape=(11,)) -> None:
        super().__init__()
        self._image_shape = image_shape
        self.proprioception_space = Box(-1, 1, shape=proprioception_shape, dtype=np.float32)
        self.action_space = Box(-1, 1, shape=(2,), dtype=np.float32)
        self.image_space = Box(low=0, high=255, shape=image_shape, dtype=np.uint8)
        self._reset = False

    def step(self, a):
        assert self._reset
        image = self.image_space.sample()
        proprioception = self.proprioception_space.sample()
        reward = -1
        done = 0

        return image, proprioception, reward, done, None

    def reset(self):
        self._reset = True
        return self.step(None)[:2]

    def close(self):
        super().close()
        
        del self
