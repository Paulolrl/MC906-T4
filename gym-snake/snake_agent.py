import sys
import gym
import gym_snake
import pygame
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy, BoltzmannGumbelQPolicy, MaxBoltzmannQPolicy
from rl.memory import SequentialMemory

window_color = (0,200,20)
clock = pygame.time.Clock()


ENV_NAME = 'snake-v0'
env = gym.make(ENV_NAME)
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(200))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(200))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Dense(200))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('softmax'))

print(model.summary())

policy = MaxBoltzmannQPolicy()
memory = SequentialMemory(limit=50000, window_length=1)
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

# Training time
dqn.fit(env, nb_steps=5000000, visualize=False, verbose=3)

# Test time
dqn.test(env, nb_episodes=5, visualize=True)
