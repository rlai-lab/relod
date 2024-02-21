from email.mime import image
import threading
import time
import pickle
import os
import numpy as np
import sys


class RadReplayBuffer(object):
    """Buffer to store environment transitions."""
    def __init__(self, image_shape, proprioception_shape, action_shape, capacity, batch_size):
        self.capacity = capacity
        self.batch_size = batch_size

        # the proprioceptive obs is stored as float32, pixels obs as uint8
        self.ignore_image = True
        self.ignore_propri = True

        if image_shape[-1] != 0:
            self.images = np.empty((capacity, *image_shape), dtype=np.uint8)
            self.next_images = np.empty((capacity, *image_shape), dtype=np.uint8)
            self.ignore_image = False

        if proprioception_shape[-1] != 0:
            self.propris = np.empty((capacity, *proprioception_shape), dtype=np.float32)
            self.next_propris = np.empty((capacity, *proprioception_shape), dtype=np.float32)
            self.ignore_propri = False

        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.dones = np.empty((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.last_save = 0
        self.full = False
        self.count = 0

    def add(self, image, propri, action, reward, next_image, next_propri, done):
        if not self.ignore_image:
            self.images[self.idx] = image
            self.next_images[self.idx] = next_image
        if not self.ignore_propri:
            self.propris[self.idx]= propri
            self.next_propris[self.idx]= next_propri
        self.actions[self.idx]= action
        self.rewards[self.idx]= reward
        self.dones[self.idx]= done

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0
        self.count = self.capacity if self.full else self.idx

    def sample(self):
        idxs = np.random.randint(
            0, self.count, size=min(self.count, self.batch_size)
        )
        if self.ignore_image:
            images = None
            next_images = None
        else:
            images = self.images[idxs]
            next_images = self.next_images[idxs]
            
        if self.ignore_propri:
            propris = None
            next_propris = None
        else:
            propris = self.propris[idxs]
            next_propris = self.next_propris[idxs]
        
        actions = self.actions[idxs]
        rewards = self.rewards[idxs]
        dones = self.dones[idxs]

        return images, propris, actions, rewards, next_images, next_propris, dones


class AsyncRadReplayBuffer(RadReplayBuffer):
    def __init__(self, image_shape, proprioception_shape, action_shape, capacity, batch_size,
                 sample_queue, minibatch_queue, init_steps, max_updates_per_step, savepath='', loadpath=''):
        super(AsyncRadReplayBuffer, self).__init__(image_shape, proprioception_shape, action_shape, capacity, batch_size)
        self.init_steps = init_steps
        self.step = 0
        self.send_count = 0
        self.max_updates_per_step = max_updates_per_step
        self.sample_queue = sample_queue
        self.minibatch_queue = minibatch_queue
        self._pause_update = False
        self.savepath = savepath
        self.loadpath = loadpath
        self._lock = threading.Lock()

        if loadpath:
            self.load()

        self.start_thread()

    def start_thread(self):
        threading.Thread(target=self.recv_from_env).start()
        threading.Thread(target=self.send_to_update).start()

    def recv_from_env(self):
        while True:
            sample = self.sample_queue.get()
            if isinstance(sample, str):
                if sample == 'pause':
                    self._pause_update = True
                    print('pause update')
                elif sample == 'resume':
                    self._pause_update = False
                    print('resume update')
                elif sample == 'save':
                    self.save()     
                elif sample == 'exit':
                    sys.exit()             
                else:
                    raise NotImplementedError()
            else:
                with self._lock:
                    self.add(*sample)
                self.step += 1

    def send_to_update(self):
        while True:
            if self._pause_update or (self.send_count > (self.step - self.init_steps) * self.max_updates_per_step):
                time.sleep(0.1)
            else:
                self.minibatch_queue.put(tuple(self.sample()))
                self.send_count += 1

    def save(self):
        if self.savepath:
            tic = time.time()
            print("Saving buffer thread spawned ...")

            with self._lock:
                data = {
                    'step': self.step,
                    'count': self.count,
                    'idx': self.idx,
                }
                
                with open(os.path.join(self.savepath, "buffer_data.pkl"), "wb") as handle:
                    pickle.dump(data, handle, protocol=4)
            
            # Sleep from time to time to release lock and get more data into the buffer
            with self._lock:
                np.save(os.path.join(self.savepath, "images.npy"), self.images)
            time.sleep(0.1)
            
            with self._lock:
                np.save(os.path.join(self.savepath, "next_images.npy"), self.next_images)
            time.sleep(0.1)

            with self._lock:
                np.save(os.path.join(self.savepath, "propris.npy"), self.propris)
                np.save(os.path.join(self.savepath, "next_propris.npy"), self.next_propris)
                np.save(os.path.join(self.savepath, "actions.npy"), self.actions)
                np.save(os.path.join(self.savepath, "rewards.npy"), self.rewards)
                np.save(os.path.join(self.savepath, "dones.npy"), self.dones)

            print("Saved the buffer locally!")
            print("Took: {:.3f}s".format(time.time()-tic))

    def load(self):
        tic = time.time()
        print("Loading buffer")

        data = pickle.load(open(os.path.join(self.loadpath, "buffer_data.pkl"), "rb"))
        self.step = data['step']
        self.count = data['count']
        self.idx = data['idx']

        self.images = np.load(os.path.join(self.loadpath, "images.npy"))
        self.next_images = np.load(os.path.join(self.loadpath, "next_images.npy"))
        self.propris = np.load(os.path.join(self.loadpath, "propris.npy"))
        self.next_propris = np.load(os.path.join(self.loadpath, "next_propris.npy"))
        self.actions = np.load(os.path.join(self.loadpath, "actions.npy"))
        self.rewards = np.load(os.path.join(self.loadpath, "rewards.npy"))
        self.dones = np.load(os.path.join(self.loadpath, "dones.npy"))
        
        print("Loaded the buffer from: {}".format(self.loadpath))
        print("Took: {:.3f}s".format(time.time()-tic))
