import torch
import os
import random

import numpy as np
import matplotlib.pyplot as plt

def make_dir(dir_path):
    try:
        os.makedirs(dir_path, exist_ok=True)
    except OSError:
        pass
    return dir_path

def save_returns(fname, rets, ep_lens):
        """ Save learning curve data as a numpy text file 
        Args:
            rets (list/array): A list or array of episodic returns
            ep_lens (list/array):  A list or array of episodic length
            savepath (str): Save path
        """
        data = np.zeros((2, len(rets)))
        data[0] = ep_lens
        data[1] = rets
        np.savetxt(fname, data)

def smoothed_curve(returns, ep_lens, x_tick=5000, window_len=5000):
    """
    Args:
        returns: 1-D numpy array with episodic returs
        ep_lens: 1-D numpy array with episodic returs
        x_tick (int): Bin size
        window_len (int): Length of averaging window
    Returns:
        A numpy array
    """
    rets = []
    x = []
    cum_episode_lengths = np.cumsum(ep_lens)

    if cum_episode_lengths[-1] >= x_tick:
        y = cum_episode_lengths[-1] + 1
        steps_show = np.arange(x_tick, y, x_tick)

        for i in range(len(steps_show)):
            rets_in_window = returns[(cum_episode_lengths > max(0, x_tick * (i + 1) - window_len)) *
                                     (cum_episode_lengths < x_tick * (i + 1))]
            if rets_in_window.any():
                rets.append(np.mean(rets_in_window))
                x.append((i+1) * x_tick)

    return np.array(rets), np.array(x)

def show_learning_curve(fname, rets, ep_lens, xtick, xlimit=None, ylimit=None, save_fig=True):
        plot_rets, plot_x = smoothed_curve(
                np.array(rets), np.array(ep_lens), x_tick=xtick, window_len=xtick)
        
        if len(plot_rets):
            plt.clf()
            if xlimit is not None:
                plt.xlim(xlimit)
        
            if ylimit is not None:
                plt.ylim(ylimit)
                
            plt.plot(plot_x, plot_rets)
            plt.pause(0.001)
            if save_fig:
                plt.savefig(fname)

def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(
            tau * param.data + (1 - tau) * target_param.data
        )

class eval_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_is_training = []
        for model in self.models:
            self.prev_is_training.append(model.is_training)
            model.train(False)

    def __exit__(self, *args):
        for model, is_training in zip(self.models, self.prev_is_training):
            model.train(is_training)
        return False

def set_seed_everywhere(seed, env=None):
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    if env is not None:
        try:
            env.seed(seed)
        except AttributeError:
            print("Env does not have seed attribute.")
        env.action_space.seed(seed)

def random_augment(images, rad_height, rad_width):
    n, c, h, w = images.shape
    _h = h - 2 * rad_height
    _w = w - 2 * rad_width
    w1 = torch.randint(0, rad_width + 1, (n,))
    h1 = torch.randint(0, rad_height + 1, (n,))
    cropped_images = torch.empty((n, c, _h, _w), device=images.device).float()
    for i, (image, w11, h11) in enumerate(zip(images, w1, h1)):
        cropped_images[i][:] = image[:, h11:h11 + _h, w11:w11 + _w]
    return cropped_images

    