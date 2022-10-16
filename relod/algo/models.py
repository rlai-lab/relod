import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Parameter
from relod.utils import random_augment

def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)


def conv_out_size(input_size, kernel_size, stride, padding=0):
    return ((input_size - kernel_size + 2 * padding) // stride) + 1

class SpatialSoftmax(torch.nn.Module):
    def __init__(self, height, width, channel, temperature=None, data_format='NCHW'):
        super(SpatialSoftmax, self).__init__()
        self.data_format = data_format
        self.height = height
        self.width = width
        self.channel = channel

        if temperature:
            self.temperature = Parameter(torch.ones(1)*temperature)
        else:
            self.temperature = 1.

        pos_x, pos_y = np.meshgrid(
                np.linspace(-1., 1., self.height),
                np.linspace(-1., 1., self.width)
                )
        pos_x = torch.from_numpy(pos_x.reshape(self.height*self.width)).float()
        pos_y = torch.from_numpy(pos_y.reshape(self.height*self.width)).float()
        self.register_buffer('pos_x', pos_x)
        self.register_buffer('pos_y', pos_y)

    def forward(self, feature):
        # Output:
        #   (N, C*2) x_0 y_0 ...
        if self.data_format == 'NHWC':
            feature = feature.transpose(1, 3).tranpose(2, 3).view(-1, self.height*self.width)
        else:
            feature = feature.contiguous().view(-1, self.height*self.width)

        softmax_attention = F.softmax(feature/self.temperature, dim=-1)
        expected_x = torch.sum(self.pos_x*softmax_attention, dim=1, keepdim=True)
        expected_y = torch.sum(self.pos_y*softmax_attention, dim=1, keepdim=True)
        expected_xy = torch.cat([expected_x, expected_y], 1)
        feature_keypoints = expected_xy.view(-1, self.channel*2)

        return feature_keypoints

class EncoderModel(nn.Module):
    """Convolutional encoder of pixels observations."""
    def __init__(self, image_shape, proprioception_shape, net_params, rad_offset, spatial_softmax=True):
        super().__init__()

        if image_shape[-1] != 0: # use image
            c, h, w = image_shape
            self.rad_h = round(rad_offset * h)
            self.rad_w = round(rad_offset * w)
            image_shape = (c, h-2*self.rad_h, w-2*self.rad_w)
            self.init_conv(image_shape, net_params)
            if spatial_softmax:
                self.latent_dim = net_params['conv'][-1][1] * 2
            else:
                self.latent_dim = net_params['latent']
            
            if proprioception_shape[-1] == 0: # no proprioception readings
                self.encoder_type = 'pixel'
                
            else: # image with proprioception
                self.encoder_type = 'multi' 
                self.latent_dim += proprioception_shape[0]

        elif proprioception_shape[-1] != 0:
            self.encoder_type = 'proprioception'
            self.latent_dim = proprioception_shape[0]

        else:
            raise NotImplementedError('Invalid observation combination')
        

    def init_conv(self, image_shape, net_params):
        conv_params = net_params['conv']
        latent_dim = net_params['latent']
        channel, height, width = image_shape
        conv_params[0][0] = channel
        layers = []
        for i, (in_channel, out_channel, kernel_size, stride) in enumerate(conv_params):
            layers.append(nn.Conv2d(in_channel, out_channel, kernel_size, stride))
            if i < len(conv_params) - 1:
                layers.append(nn.ReLU())
            width = conv_out_size(width, kernel_size, stride)
            height = conv_out_size(height, kernel_size, stride)

        self.convs = nn.Sequential(
            *layers
        )
        self.ss = SpatialSoftmax(width, height, conv_params[-1][1])
        self.fc = nn.Linear(conv_params[-1][1] * width * height, latent_dim)
        self.ln = nn.LayerNorm(latent_dim)
        self.apply(weight_init)

    def forward(self, images, proprioceptions, random_rad=True, detach=False):
        if self.encoder_type == 'proprioception':
            return proprioceptions

        if self.encoder_type == 'pixel' or self.encoder_type == 'multi':
            images = images / 255.
            if random_rad:
                images = random_augment(images, self.rad_h, self.rad_w)
            else:
                n, c, h, w = images.shape
                images = images[:, :,
                  self.rad_h : h-self.rad_h,
                  self.rad_w : w-self.rad_w,
                  ]

            h = self.ss(self.convs(images))
            if detach:
                h = h.detach()

            if self.encoder_type == 'multi':
                h = torch.cat([h, proprioceptions], dim=-1)

            return h
        else:
            raise NotImplementedError('Invalid encoder type')

def gaussian_logprob(noise, log_std):
    """Compute Gaussian log probability."""
    residual = (-0.5 * noise.pow(2) - log_std).sum(-1, keepdim=True)
    return residual - 0.5 * np.log(2 * np.pi) * noise.size(-1)

def squash(mu, pi, log_pi):
    """Apply squashing function.
    See appendix C from https://arxiv.org/pdf/1812.05905.pdf.
    """
    mu = torch.tanh(mu)
    if pi is not None:
        pi = torch.tanh(pi)
    if log_pi is not None:
        log_pi -= torch.log(F.relu(1 - pi.pow(2)) + 1e-6).sum(-1, keepdim=True)
    return mu, pi, log_pi

LOG_STD_MIN = -10
LOG_STD_MAX = 10

class ActorModel(nn.Module):
    """MLP actor network."""
    def __init__(
        self, image_shape, proprioception_shape, action_dim, net_params, rad_offset):
        super().__init__()

        self.encoder = EncoderModel(image_shape, proprioception_shape, net_params, rad_offset)

        mlp_params = net_params['mlp']
        mlp_params[0][0] = self.encoder.latent_dim
        mlp_params[-1][-1] = action_dim * 2
        layers = []
        for i, (in_dim, out_dim) in enumerate(mlp_params):
            layers.append(nn.Linear(in_dim, out_dim))
            if i < len(mlp_params) - 1:
                layers.append(nn.ReLU())
        self.trunk = nn.Sequential(
            *layers
        )

        self.outputs = dict()
        self.apply(weight_init)
        self.trunk[-1].weight.data.fill_(0.0)
        self.trunk[-1].bias.data.fill_(0.0)
        print('Using normal distribution initialization.')

    def forward(
        self, images, proprioceptions, random_rad=True, compute_pi=True, compute_log_pi=True, detach_encoder=False):
        latents = self.encoder(images, proprioceptions, random_rad, detach=detach_encoder)
        mu, log_std = self.trunk(latents).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (
            LOG_STD_MAX - LOG_STD_MIN
        ) * (log_std + 1)

        self.outputs['mu'] = mu
        self.outputs['std'] = log_std.exp()

        if compute_pi:
            std = log_std.exp()
            noise = torch.randn_like(mu)
            pi = mu + noise * std
        else:
            pi = None

        if compute_log_pi:
            log_pi = gaussian_logprob(noise, log_std)
        else:
            log_pi = None

        mu, pi, log_pi = squash(mu, pi, log_pi)
        return mu, pi, log_pi, log_std

class QFunction(nn.Module):
    """MLP for q-function."""
    def __init__(self, latent_dim, action_dim, net_params):
        super().__init__()

        mlp_params = net_params['mlp']
        mlp_params[0][0] = latent_dim + action_dim
        mlp_params[-1][-1] = 1
        layers = []
        for i, (in_dim, out_dim) in enumerate(mlp_params):
            layers.append(nn.Linear(in_dim, out_dim))
            if i < len(mlp_params) - 1:
                layers.append(nn.ReLU())
        self.trunk = nn.Sequential(
            *layers
        )

    def forward(self, latents, actions):
        latent_actions = torch.cat([latents, actions], dim=1)
        
        return self.trunk(latent_actions)

class CriticModel(nn.Module):
    """Critic network, employes two q-functions."""
    def __init__(
        self, image_shape, proprioception_shape, action_dim, net_params, rad_offset):
        super().__init__()

        self.encoder = EncoderModel(image_shape, proprioception_shape, net_params, rad_offset)

        self.Q1 = QFunction(
            self.encoder.latent_dim, action_dim, net_params
        )
        self.Q2 = QFunction(
            self.encoder.latent_dim, action_dim, net_params
        )

        self.outputs = dict()
        self.apply(weight_init)

    def forward(self, images, proprioceptions, actions, detach_encoder=False):
        # detach_encoder allows to stop gradient propogation to encoder
        latents = self.encoder(images, proprioceptions, detach=detach_encoder)
        q1s = self.Q1(latents, actions)
        q2s = self.Q2(latents, actions)

        self.outputs['q1'] = q1s
        self.outputs['q2'] = q2s

        return q1s, q2s