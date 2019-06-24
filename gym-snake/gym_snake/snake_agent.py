import sys
sys.path.insert(0, 'envs')
from snake_env import SnakeEnv
import gym
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory


ENV_NAME = 'snake-v0'
env = gym.make(ENV_NAME)
np.random.seed(123)
env.seed(123)
actions = env.action_space.n
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
print(model.summary())
