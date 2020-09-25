import pygame, time, random
import math, copy
from pygame.locals import *
import numpy as np

SCREEN_WIDTH = 550
SCREEN_HEIGTH = 550
BG_COLOR = pygame.Color(255,255,255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
WHITE = (255, 255, 255)
BLAKE = (0,0,0)
TEXT_COLOR = pygame.Color(255, 0, 0)

LEFT, UP, RIGHT, DOWN = 0, 1, 2, 3


class MazeEnv():
    def __init__(self):
        pygame.display.init()
        self.window = pygame.display.set_mode([SCREEN_WIDTH, SCREEN_HEIGTH])
        pygame.display.set_caption("Maze Game")
        self.robot = []

        self.GRID_SIZE = 5

        self.state_space = self.get_state_space()  # 其中key为状态编号，其值为网格的实际坐标
        self.nS = len(self.state_space)
        self.action_space = [0, 1, 2, 3]

        self.ACTIONS = [np.array([0, -1]),
                        np.array([-1, 0]),
                        np.array([0, 1]),
                        np.array([1, 0])]
        self.nA = len(self.action_space)

        self.trap_space = [[0, 3], [1, 3], [2, 0], [2, 1], [4, 2], [4, 3], [4, 4]] # 不包括宝藏的位置
        self.treasure_space = [[2, 4]]
        self.terminate_space = [[0, 3], [1, 3], [2, 0], [2, 1], [2, 4], [4, 2], [4, 3], [4, 4]]

        self.transition = {}

        self.current_state = [0, 0]  # 当前状态

        self.robot_image = pygame.image.load('../Maze-game/img/robot.jpg')
        self.treasure_image = pygame.image.load('../Maze-game/img/treasure.jpg')


    def reset(self):
        temp = copy.deepcopy(list(self.state_space.keys()))
        temp = [list(state) for state in temp]
        for state in self.terminate_space:
            temp.remove(state)

        self.current_state = temp[np.random.randint(0, len(temp))] # 随机初始化非终止状态
        return self.current_state


    def render(self):
        self.window.fill(WHITE)
        self.getEvent()

        # 绘画横向线段
        grid_length = 6
        for j in range(0, grid_length):
            pygame.draw.line(self.window, BLAKE, (0, j * 100), (500, j * 100), 2)

        # 绘画纵向线段
        for i in range(0, grid_length):
            pygame.draw.line(self.window, BLAKE, (i * 100, 0), (i * 100, 500), 2)

        # 绘制宝藏
        rect = self.treasure_image.get_rect()
        rect.left = 410
        rect.top = 205
        self.window.blit(self.treasure_image, rect)

        # 绘制陷阱
        self.createTrap()

        # 绘制robot
        pygame.draw.circle(self.window, RED, self.state_space[(self.current_state[0], self.current_state[1])], 50)

        pygame.display.flip()

    def step(self, state, action):
        next_state = (np.array(state) + self.ACTIONS[action]).tolist()
        x, y = next_state

        if next_state in self.treasure_space:
            return next_state, 1, True
        elif next_state in self.trap_space:
            return next_state, -1, True
        else:
            if x < 0 or x >= self.GRID_SIZE or y < 0 or y >= self.GRID_SIZE:
                next_state = state
            return next_state, 0, False


    def getTextSurface(self, text):
        pygame.font.init()
        # 查看所有可用字体
        # print(pygame.font.get_fonts())

        #获取字体Font对象
        font = pygame.font.SysFont('kaiti', 30)

        # 绘制文字信息
        textSurface = font.render(text, True, TEXT_COLOR)

        return textSurface


    def createTrap(self):
        pygame.draw.rect(self.window, BLAKE, Rect((300, 0), (100, 100)))
        pygame.draw.rect(self.window, BLAKE, Rect((300, 100), (100, 100)))
        pygame.draw.rect(self.window, BLAKE, Rect((0, 200), (100, 100)))
        pygame.draw.rect(self.window, BLAKE, Rect((100, 200), (100, 100)))
        pygame.draw.rect(self.window, BLAKE, Rect((200, 400), (100, 100)))
        pygame.draw.rect(self.window, BLAKE, Rect((300, 400), (100, 100)))
        pygame.draw.rect(self.window, BLAKE, Rect((400, 400), (100, 100)))


    def get_state_space(self):
        #从左到右从上到下，依次使用在网格中的坐标标号
        states = {}
        #------------以下代码使用网格坐标来表示状态，键值为行列坐标，其值为真实的坐标值-----------------
        i = 0
        for top in range(50, 550, 100):
            j = 0
            for left in range(50, 550, 100):
                states[(i, j)] = (left, top)
                j += 1
            i += 1
        return states


    def getEvent(self):
        #获取所有事件
        eventList = pygame.event.get()
        for event in eventList:
            if event.type == pygame.QUIT:
                exit()
