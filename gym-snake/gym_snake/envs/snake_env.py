import gym
from gym import error, spaces, utils
from gym.utils import seeding
import random
import numpy as np
import pygame
import math
window_color = (200, 200, 200)

green = (0,255,0)
red = (255,0,0)
black = (0,0,0)
clock = pygame.time.Clock()
apple_image = pygame.image.load('apple.jpg')

def collision_with_apple(apple_position, score, snake_position):
    score += 1
    while True:
        apple_position = [random.randrange(1,20)*10,random.randrange(1,20)*10]
        for x, y in snake_position:
            if x == apple_position[0] and y == apple_position[1]:
                continue
        break
    return apple_position, score

def collision_with_boundaries(snake_head):
    if snake_head[0]>=200 or snake_head[0]<0 or snake_head[1]>=200 or snake_head[1]<0 :
        return 1
    else:
        return 0

def collision_with_self(snake_position):
    snake_head = snake_position[0]
    if snake_head in snake_position[1:]:
        return 1
    else:
        return 0

def is_direction_blocked(snake_position, current_direction_vector):
    next_step = snake_position[0]+ current_direction_vector
    snake_head = snake_position[0]
    if collision_with_boundaries(snake_head) == 1 or collision_with_self(snake_position) == 1:
        return 1
    else:
        return 0

def generate_snake(snake_head, snake_position, apple_position, button_direction, score):

    if button_direction == 0:
        snake_head[0] += 10
    elif button_direction == 2:
        snake_head[0] -= 10
    elif button_direction == 1:
        snake_head[1] += 10
    elif button_direction == 3:
        snake_head[1] -= 10
    else:
        pass

    if snake_head == apple_position:
        apple_position, score = collision_with_apple(apple_position, score, snake_position)
        snake_position.insert(0,list(snake_head))

    else:
        snake_position.insert(0,list(snake_head))
        snake_position.pop()

    return snake_position, apple_position, score

def display_snake(display, snake_position):
    for position in snake_position:
        pygame.draw.rect(display,red,pygame.Rect(position[0],position[1],10,10))

def display_apple(display,apple_position, apple):
    display.blit(apple,(apple_position[0], apple_position[1]))

def calcula_dist(apple_position, snake_head):
    return math.sqrt((apple_position[0] - snake_head[0])**2 + (apple_position[1] - snake_head[1])**2)

def display_final_score(display, display_text, final_score):
    largeText = pygame.font.Font('freesansbold.ttf',20)
    TextSurf = largeText.render(display_text, True, black)
    TextRect = TextSurf.get_rect()
    TextRect.center = ((display_width/2),(display_height/2))
    display.blit(TextSurf, TextRect)
    pygame.display.update()
    time.sleep(2)


class SnakeEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    counter=1

    def __init__(self):
        self.snake_head = [100,100]
        self.snake_position = [[100,100],[90,100],[80,100]]
        self.apple_position = [random.randrange(1,20)*10,random.randrange(1,20)*10]
        self.score = 0
        self.button_direction = 0
        self.prev_button_direction = 0
        self.current_direction_vector = np.array(self.snake_position[0])-np.array(self.snake_position[1])
        self.moves = 0
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(0, 1, [1, 11], dtype=np.uint8)
        # self.observation_space = spaces.Box(0, 3, [22, 22], dtype=np.uint8)
        pygame.init()
        self.display = pygame.display.set_mode((200,200))
        self.display.fill(window_color)
        pygame.display.update()
        pygame.event.get()

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

        bonus = 0
        episode_over = False

        key = action
        if key == 2 and self.prev_button_direction != 0:
            self.button_direction = 2
        elif key == 0 and self.prev_button_direction != 2:
            self.button_direction = 0
        elif key == 1 and self.prev_button_direction != 3:
            self.button_direction = 1
        elif key == 3 and self.prev_button_direction != 1:
            self.button_direction = 3
        else:
            self.button_direction = self.prev_button_direction

        dist_antes = calcula_dist(self.apple_position, self.snake_head)
        score_antes = self.score
        self.snake_position, self.apple_position, self.score = generate_snake(self.snake_head, self.snake_position, self.apple_position, self.button_direction, self.score)
        dist_depois = calcula_dist(self.apple_position, self.snake_head)

        # if dist_depois < dist_antes and self.score == score_antes:
        #     bonus += 0.1
        # elif self.score == score_antes:
        #     bonus -= 0.2

        self.prev_button_direction = self.button_direction

        reward = bonus + (self.score - score_antes)

        if is_direction_blocked(self.snake_position, self.current_direction_vector) == 1:
            reward = -5
            episode_over = True


        if not episode_over:
            ob = self.get_state()
        else:
            ob = np.zeros((1,11), dtype=np.uint8)

        # ob = self.get_state()
        # print(ob)

        self.moves += 1

        if self.moves >= 1000:
            episode_over = True

        return ob, reward*10, episode_over, {'score': self.score}

    def reset(self):
        print('Score:', self.score)
        self.snake_head = [100,100]
        self.snake_position = [[100,100],[90,100],[80,100]]
        self.apple_position = [random.randrange(1,20)*10,random.randrange(1,20)*10]
        self.score = 0
        self.button_direction = 0
        self.prev_button_direction = 0
        self.current_direction_vector = np.array(self.snake_position[0])-np.array(self.snake_position[1])
        self.moves = 0
        return self.get_state()

    def get_state(self):
        ob = np.zeros((22,22), dtype=np.uint8)
        # count = 0
        ob[int(self.apple_position[0]/10)+1, int(self.apple_position[1]/10)+1] = 3
        for x, y in self.snake_position:
            ob[int(x/10)+1, int(y/10)+1] = 4
            # if count == 1:
            #     ob[int(x/10)+1, int(y/10)+1] = 1
            # else:
            #     ob[int(x/10)+1, int(y/10)+1] = 4
            # count += 1
        for i in range(22):
            for j in range(22):
                if i == 0 or j == 0 or j == 21 or i == 21:
                    ob[i,j]=4

        ob[int(self.snake_head[0]/10)+1, int(self.snake_head[1]/10)+1] = 2
        newob = ob[int(self.snake_head[0]/10):int(self.snake_head[0]/10)+3, int(self.snake_head[1]/10):int(self.snake_head[1]/10)+3]
        newerob = np.zeros((1,11), dtype=np.uint8)

        prev_dir = self.prev_button_direction

        apple_x = int(self.apple_position[0]/10)+1
        apple_y = int(self.apple_position[1]/10)+1
        head_x = int(self.snake_head[0]/10)+1
        head_y = int(self.snake_head[1]/10)+1

        newerob[0][prev_dir] = 1

        if apple_x > head_x:
            newerob[0][4] = 1
        if apple_x < head_x:
            newerob[0][5] = 1
        if apple_y > head_y:
            newerob[0][6] = 1
        if apple_y < head_y:
            newerob[0][7] = 1


        if prev_dir == 0:
            if newob[1][2] == 4:
                newerob[0][8] = 1
            if newob[0][1] == 4:
                newerob[0][9] = 1
            if newob[2][1] == 4:
                newerob[0][10] = 1

        if prev_dir == 1:
            if newob[0][1] == 4:
                newerob[0][8] = 1
            if newob[1][0] == 4:
                newerob[0][9] = 1
            if newob[1][2] == 4:
                newerob[0][10] = 1

        if prev_dir == 2:
            if newob[1][0] == 4:
                newerob[0][8] = 1
            if newob[2][1] == 4:
                newerob[0][9] = 1
            if newob[0][1] == 4:
                newerob[0][10] = 1

        if prev_dir == 3:
            if newob[2][1] == 4:
                newerob[0][8] = 1
            if newob[1][2] == 4:
                newerob[0][9] = 1
            if newob[1][0] == 4:
                newerob[0][10] = 1

        # print(prev_dir)
        # print(newob)
        # print(newerob)
        return newerob

    def render(self, mode='human'):
        # print(self.get_state())
        # if self.counter >= 1:
        #     return
        self.display.fill(window_color)
        display_apple(self.display,self.apple_position,apple_image)
        display_snake(self.display,self.snake_position)
        pygame.display.update()
        pygame.event.get()
        clock.tick(20)
        return
