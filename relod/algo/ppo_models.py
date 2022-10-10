import torch

from algo.models import EncoderModel, weight_init
from torch import nn
from torch.distributions import Normal


class ActorModel(nn.Module):
    """MLP actor network."""
    def __init__(self, image_shape, proprioception_shape, action_dim, net_params, rad_offset, freeze_cnn=False):
        super().__init__()

        self.encoder = EncoderModel(image_shape, proprioception_shape, net_params, rad_offset)
        if freeze_cnn:
            print("Actor CNN weights won't be trained!")
            for param in self.encoder.parameters():
                param.requires_grad = False

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

    def forward(self, images, proprioceptions, random_rad=True, detach_encoder=False):
        latents = self.encoder(images, proprioceptions, random_rad, detach=detach_encoder)
        mu, log_std = self.trunk(latents).chunk(2, dim=-1)
        std = log_std.exp()

        self.outputs['mu'] = mu
        self.outputs['std'] = std

        dist = Normal(mu, std)

        action = dist.sample()
        lprob = dist.log_prob(action).sum(axis=-1)

        return mu, action, lprob

    def lprob(self, images, proprioceptions, actions, random_rad=True, detach_encoder=False):
        latents = self.encoder(images, proprioceptions, random_rad, detach=detach_encoder)
        mu, log_std = self.trunk(latents).chunk(2, dim=-1)
        std = log_std.exp()

        dist = Normal(mu, std)
        return dist.log_prob(actions).sum(axis=-1)


class CriticModel(nn.Module):
    def __init__(self, image_shape, proprioception_shape, net_params, rad_offset, freeze_cnn=False):
        super().__init__()

        self.encoder = EncoderModel(image_shape, proprioception_shape, net_params, rad_offset)
        if freeze_cnn:
            print("Critic CNN weights won't be trained!")
            for param in self.encoder.parameters():
                param.requires_grad = False

        mlp_params = net_params['mlp']
        mlp_params[0][0] = self.encoder.latent_dim
        mlp_params[-1][-1] = 1
        layers = []
        for i, (in_dim, out_dim) in enumerate(mlp_params):
            layers.append(nn.Linear(in_dim, out_dim))
            if i < len(mlp_params) - 1:
                layers.append(nn.ReLU())
        self.trunk = nn.Sequential(*layers)

        self.outputs = dict()
        self.apply(weight_init)

    def forward(self, images, proprioceptions, random_rad=True, detach_encoder=False):
        latents = self.encoder(images, proprioceptions, random_rad, detach=detach_encoder)
        vals = self.trunk(latents)
        return vals.view(-1)
