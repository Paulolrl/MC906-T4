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
model.add(Dense(120, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(120, activation='relu'))
model.add(Dropout(0.3))
# model.add(Dense(120, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(256, activation='relu'))
# model.add(Dropout(0.1))
model.add(Dense(nb_actions))
model.add(Activation('softmax'))

print(model.summary())

policy = MaxBoltzmannQPolicy()
memory = SequentialMemory(limit=100000, window_length=1)
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

# Training time
dqn.fit(env, nb_steps=50000, visualize=False, verbose=3)
=======
dqn.fit(env, nb_steps=70000, visualize=False, verbose=3)
>>>>>>> b35c02923a5c13b4fc35fcab227f16b014311edd

# Test time
dqn.test(env, nb_episodes=5, visualize=True)
