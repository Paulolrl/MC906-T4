import gym
from gym import error, spaces, utils
from gym.utils import seeding
import random
import numpy as np
import pygame
import math

class SnakeEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.prev_action = 1
        self.score = 0
        self.snake_position = [[100,100],[90,100],[80,100]]
        self.snake_head = [100,100]
        self.apple_position = [random.randrange(1,20)*10,random.randrange(1,20)*10]
        self.display = pygame.display.set_mode((200,200))
        self.clock = pygame.time.Clock()
        self.apple_image = pygame.image.load('apple.jpg')
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(0, 3, [20, 20], dtype=np.uint8)
        self.seed()
        self.button_direction = 1
        self.moves = 0

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def calcula_dist(self,apple_position, snake_head):
        return math.sqrt((apple_position[0] - snake_head[0])**2 + (apple_position[1] - snake_head[1])**2)

    def step(self, action):
        """

        Parameters
        ----------
        action :

        Returns
        -------
        ob, reward, episode_over, info : tuple
            ob (object) :
                an environment-specific object representing your observation of
                the environment.
            reward (float) :
                amount of reward achieved by the previous action. The scale
                varies between environments, but the goal is always to increase
                your total reward.
            episode_over (bool) :
                whether it's time to reset the environment again. Most (but not
                all) tasks are divided up into well-defined episodes, and done
                being True indicates the episode has terminated. (For example,
                perhaps the pole tipped too far, or you lost your last life.)
            info (dict) :
                 diagnostic information useful for debugging. It can sometimes
                 be useful for learning (for example, it might contain the raw
                 probabilities behind the environment's last state change).
                 However, official evaluations of your agent are not allowed to
                 use this for learning.
        """

        current_direction_vector = np.array(self.snake_position[0])-np.array(self.snake_position[1])
        bonus = 0
        episode_over = False


        key = action
        if key == 2 and self.prev_action != 1:
            self.button_direction = 0
        elif key == 0 and self.prev_action != 0:
            self.button_direction = 1
        elif key == 1 and self.prev_action != 2:
            self.button_direction = 3
        elif key == 3 and self.prev_action != 3:
            self.button_direction = 2
        else:
            self.button_direction = self.button_direction

        dist_antes = self.calcula_dist(self.apple_position, self.snake_head)
        score_antes = self.score
        self.take_action(self.button_direction)
        dist_depois = self.calcula_dist(self.apple_position, self.snake_head)
        if dist_depois < dist_antes:
            bonus += 0.1
        else:
            bonus -= 0.2

        reward = bonus + (self.score - score_antes)

        # if display is not None:
        #     pygame.display.set_caption("Snake Game"+"  "+"SCORE: "+str(score))
        #     pygame.display.update()
        self.prev_action = self.button_direction
        if self.is_direction_blocked(current_direction_vector) == 1:
            episode_over = True

        ob = self.get_state()

        if self.moves >= 1000:
            episode_over = True
        
        self.moves += 1

        return ob, reward, episode_over, {'score': self.score}

    def reset(self):
        print('Score:', self.score)
        self.prev_action = 1
        self.score = 0
        self.snake_position = [[100,100],[90,100],[80,100]]
        self.snake_head = [100,100]
        self.apple_position = [random.randrange(1,20)*10,random.randrange(1,20)*10]
        self.display = False
        self.button_direction = 1
        self.moves = 0
        return self.get_state()

    def get_state(self):
        ob = np.zeros((20,20), dtype=np.uint8)
        for x, y in self.snake_position:
            ob[int(x/10)-1, int(y/10)-1] = 1
        ob[int(self.snake_head[0]/10)-1, int(self.snake_head[1]/10)-1] = 2
        ob[int(self.apple_position[0]/10)-1, int(self.apple_position[1]/10)-1] = 3
        return ob

    def render(self, mode='human'):
        print(self.get_state())
        return

    def take_action(self, action):
        if action == 1:
            self.snake_head[0] += 10
        elif action == 0:
            self.snake_head[0] -= 10
        elif action == 2:
            self.snake_head[1] += 10
        elif action == 3:
            self.snake_head[1] -= 10
        else:
            pass

        if self.snake_head == self.apple_position:
            self.collision_with_apple()
            self.snake_position.insert(0,list(self.snake_head))

        else:
            self.snake_position.insert(0,list(self.snake_head))
            self.snake_position.pop()

        return

    def is_direction_blocked(self, current_direction_vector):
        # next_step = self.snake_position[0]+ current_direction_vector
        snake_head = self.snake_position[0]
        if self.collision_with_boundaries(snake_head) == 1 or self.collision_with_self(self.snake_position) == 1:
            return 1
        else:
            return 0

    def collision_with_apple(self):
        self.score += 1
        self.apple_position = [random.randrange(1,20)*10,random.randrange(1,20)*10]
        return

    def collision_with_boundaries(self,snake_head):
        if snake_head[0]>=200 or snake_head[0]<0 or snake_head[1]>=200 or snake_head[1]<0 :
            return 1
        else:
            return 0

    def collision_with_self(self, snake_position):
        snake_head = snake_position[0]
        if snake_head in snake_position[1:]:
            return 1
        else:
            return 0

    def display_snake(self, display, snake_position):
        for position in snake_position:
            pygame.draw.rect(display,red,pygame.Rect(position[0],position[1],10,10))

    def display_apple(self,display,apple_position, apple):
        display.blit(apple,(apple_position[0], apple_position[1]))
