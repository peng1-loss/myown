import numpy as np
import scipy.signal

import torch
import torch.nn as nn


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def cnn(sizes, activation, output_dim):
    # layers = []
    # for j in range(len(sizes)-1):
    #     act = activation if j < len(sizes)-2 else output_activation
    #     layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    # return nn.Sequential(*layers)
    layers = []
    layers[0] = [nn.Conv2d(sizes,32, kernel_size=8, stride=4),activation]
    layers[1] = [nn.Conv2d(32, 64, kernel_size=4, stride=2),activation]
    layers[2] = [nn.Conv2d(64, 64, kernel_size=3, stride=1),activation]
    layers[3] = [nn.Linear(3136, 512),activation]
    layers[4] = [nn.Linear(512,output_dim),activation]
    return nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])

class CNNActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation,):#drop act_limit
        super().__init__()
        pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
        self.pi = cnn(pi_sizes, activation)
        #self.act_limit = act_limit

    def forward(self, obs):
        # Return output from network scaled to action space limits.
        return self.self.pi(obs)#drop act_limit

class CNNQFunction(nn.Module):

    def __init__(self, obs_dim, hidden_sizes, activation):#drop act_limit
        super().__init__()
        self.q = cnn([obs_dim] + list(hidden_sizes) + [1], activation)#drop act_limit

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1) # Critical to ensure q has right shape.

class CNNActorCritic(nn.Module):

    def __init__(self, observation_space, action_space, hidden_sizes=(256,256),
                 activation=nn.ReLU):
        super().__init__()
        
        obs_dim = observation_space.shape
        act_dim = action_space.shape
        #act_limit = action_space.high[0]

        # build policy and value functions
        self.pi = CNNActor(obs_dim, act_dim, hidden_sizes, activation)#drop act_limit
        self.q = CNNQFunction(obs_dim, act_dim, hidden_sizes, activation)

    def act(self, obs):
        with torch.no_grad():
            return self.pi(obs).numpy()
