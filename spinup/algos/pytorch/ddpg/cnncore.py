import numpy as np
import scipy.signal

import torch
import torch.nn as nn


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

# def cnn(input_size,output_size, activation):
#     # layers = []
#     # for j in range(len(sizes)-1):
#     #     act = activation if j < len(sizes)-2 else output_activation
#     #     layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
#     # return nn.Sequential(*layers)
#     layers = []
#     print("input_size",input_size)
#     layers.extend([nn.Conv2d(input_size, 32, kernel_size=8, stride=4),activation])
#     layers.extend([nn.Conv2d(32, 64, kernel_size=4, stride=2),activation])
#     layers.extend([nn.Conv2d(64, 64, kernel_size=3, stride=1),activation])
#     layers.extend([nn.Linear(3136, 512),activation])
#     print("output_size",output_size)
#     layers.extend([nn.Linear(512,output_size),activation])
#     return nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])

class CNNActor(nn.Module):

    def __init__(self, obs_dim, act_dim):#drop act_limit
        # super().__init__()
        # self.pi = cnn(obs_dim[-1],act_dim, activation)
        #pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
        super(CNNActor, self).__init__()
        temp=obs_dim
        self.c1 = nn.Conv2d(obs_dim[-1], 32, kernel_size=8, stride=4)
        self.c2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.c3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.l1 = nn.Linear(1024, 512)
        self.l2 = nn.Linear(512, act_dim)
        #self.act_limit = act_limit

    def forward(self, obs):
        # Return output from network scaled to action space limits.
        q = nn.functional.relu(self.c1(obs))
        q = nn.functional.relu(self.c2(q))
        q = nn.functional.relu(self.c3(q))  # (32, 64, 4, 4)
        tmp = q.reshape(-1, 1024)
        q = nn.functional.relu(self.l1(q.reshape(-1, 1024)))
        return self.l2(q)
        # return self.self.pi(obs)#drop act_limit

class CNNQFunction(nn.Module):

    def __init__(self, obs_dim,act_dim):#drop act_limit
        # super().__init__()
        # self.q = cnn(obs_dim[-1] ,act_dim , activation)#drop act_limit
        super(CNNQFunction, self).__init__()
        self.a1 = nn.Linear(act_dim, 64)
        self.a2 = nn.Linear(64, 64)
        self.a3 = nn.Linear(64, 512)

        self.c1 = nn.Conv2d(obs_dim[-1], 32, kernel_size=8, stride=4)
        self.c2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.c3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.l1 = nn.Linear(1024, 512)

        self.l2 = nn.Linear(1024, 64)
        self.l3 = nn.Linear(64, act_dim)


    def forward(self, obs, act):
        q1 = nn.functional.relu(self.c1(obs))
        q1 = nn.functional.relu(self.c2(q1))
        q1 = nn.functional.relu(self.c3(q1))
        q1 = nn.functional.relu(self.l1(q1.reshape(-1, 1024)))

        q2 = nn.functional.relu(self.a1(act))
        q2 = nn.functional.relu(self.a2(q2))
        q2 = nn.functional.relu(self.a3(q2))

        q = nn.functional.relu(self.l2(torch.cat([q1, q2], 1)))
        q = self.l3(q)
        # q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1) # Critical to ensure q has right shape.

class CNNActorCritic(nn.Module):

    def __init__(self, observation_space, action_space, hidden_sizes=(256,256),
                 ):
        super().__init__()
        
        obs_dim = observation_space.shape
        print("cnncore_obs_dim",obs_dim)
        act_dim = action_space.n
        print("act_dim",act_dim)
        #act_limit = action_space.high[0]

        # build policy and value functions
        self.pi = CNNActor(obs_dim, act_dim)#drop act_limit
        self.q = CNNQFunction(obs_dim, act_dim)

    def act(self, obs):
        with torch.no_grad():
            return self.pi(obs).numpy()
