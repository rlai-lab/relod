import torch
import numpy as np
from collections import namedtuple

class VisuomotorReplayBuffer:
    Transition = namedtuple('Transition', ('img', 'prop', 'action', 'reward', 'done', 'lprob'))
    def __init__(self, image_shape, proprioception_shape, action_shape, capacity, store_lprob=False):
        self.buffer = []
        self.done_indices = []

    def push(self, images, proprioception, action, reward, done, lprob):
        """ Saves a transition."""
        self.buffer.append(self.Transition(images, proprioception, action, reward, done, lprob))
        if done:
            self.done_indices.append(len(self.buffer))

    def sample(self, batch_size):
        if batch_size >= len(self.buffer):
            batch = self.Transition(*zip(*self.buffer))
        else:
            raise NotImplemented

        if torch.is_tensor(batch.prop[0]):
            propris = torch.cat(batch.prop, dim=0)
        else:
            propris = torch.from_numpy(np.stack(batch.prop).astype(np.float32))

        images = torch.from_numpy(np.concatenate(batch.img, axis=0).astype(np.float32))
        actions = torch.from_numpy(np.stack(batch.action).astype(np.float32))
        rewards = torch.from_numpy(np.stack(batch.reward).astype(np.float32))
        dones = torch.from_numpy(np.stack(batch.done).astype(np.float32))
        lprobs = torch.from_numpy(np.stack(batch.lprob).astype(np.float32)).view(-1)

        return images, propris, actions, rewards, dones, lprobs

    @property
    def n_episodes(self):
        return len(self.done_indices)

    def reset(self):
        self.buffer = []
        self.done_indices = []

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, item):
        t = self.Transition(*zip(*self.buffer[item]))
        return t.img, t.prop, t.action, t.reward, t.done, t.lprob
        