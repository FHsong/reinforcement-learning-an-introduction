# On-policy first-visit MC control (for "epsilon-soft policies) Page 101

import numpy as np
import gym, time
from MC_maze_env import MazeEnv

class OnPolice_MC:
    def __init__(self, env):
        self.env = env
        self.theta = 1e-4  # A small positive number determining the accuracy of estimation
        self.discount = 1
        self.pi = []
        self.Q_table = []
        self.Returns = []

        self.state_action_count_list = []

        self.episode_number = 10000

        self.state_list = None
        self.action_list = None
        self.reward_list = None

        self.epsilon = 0.1            # ε-soft policies
        self.state_to_gridState = []  # 从单个数字状态到网格坐标的映射，如 0 -> [0, 0]


    def run(self): # 实现核心算法
        pass


    def generate_one_episode(self):
        self.state_list = []
        self.action_list = []
        self.reward_list = []
        state_grid = env.reset()
        while True:
            state = self.state_to_gridState.index(state_grid)

            action_P = np.array(self.pi[state])
            action = np.random.choice(np.arange(self.env.nA), p=action_P.ravel())

            next_state, reward, done = self.env.step(state_grid, action)
            self.load_one_transmission(state, action, reward)
            state_grid = next_state
            if done:
                break


    def load_one_transmission(self, state, action, reward):
        self.state_list.append(state)
        self.action_list.append(action)
        self.reward_list.append(reward)


    def initialization(self):
        self.pi = [[1 - self.epsilon + self.epsilon/self.env.nA] for _ in range(self.env.nS)]  # 选中贪心策略的概率 1-ε+ε/|A(s)|
        for s in range(self.env.nS):  # 初始其他动作的概率 ε/|A(s)|
            for a in range(self.env.nA-1):
                self.pi[s].append(self.epsilon/self.env.nA)

        for i in range(self.env.GRID_SIZE):
            for j in range(self.env.GRID_SIZE):
                self.state_to_gridState.append([i, j])

        for s in range(self.env.nS):
            self.Q_table.append([1.] * self.env.nA)

            self.Returns.append([] * self.env.nA)

            self.state_action_count_list.append([0.] * self.env.nA)



if __name__ == '__main__':
    env = MazeEnv()
    mc_es = OnPolice_MC(env)
    pi = mc_es.run()
    print(pi)

    # 测试获得的策略
    state_grid = env.reset()
    env.render()
    time.sleep(0.5)
    while True:
        state = mc_es.state_to_gridState.index(state_grid)
        action_P = np.array(mc_es.pi[state])
        action = np.random.choice([a for a in range(mc_es.env.nA)], p=action_P.ravel())

        next_state, reward, done = mc_es.env.step(state_grid, action)
        state_grid = next_state
        env.current_state = next_state
        env.render()
        time.sleep(0.5)
        if done:
            break
