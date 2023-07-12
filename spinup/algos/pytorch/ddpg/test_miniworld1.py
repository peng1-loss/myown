import argparse
import math
import gymnasium as gym
import pyglet
from pyglet.window import key
import miniworld

# env = gym.make("Acrobot-v1")
env = gym.make("MiniWorld-Hallway-v0", view="top", render_mode="human")
obs_dim = env.observation_space.shape
act_dim = env.action_space
print("obs:", obs_dim, "act:", act_dim)

# initialize
env.reset()
for i in range(1000):
    action = env.action_space.sample()
    o2 = env.step(action)
    env.render()
    print("state_", o2)