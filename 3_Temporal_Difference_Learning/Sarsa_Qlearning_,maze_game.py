import numpy as np
import gym, time
import Maze_Env
import matplotlib.pylab as plt
from tqdm import tqdm

class MazeGame:
    def __init__(self, env):
        self.env = env
        self.theta = 1e-4  # A small positive number determining the accuracy of estimation

        self.pi = []
        self.Q_table = []

        self.state_to_gridState = [] # 索引是状态index，其值为网格的虚拟坐标
        for i in range(self.env.GRID_SIZE):
            for j in range(self.env.GRID_SIZE):
                self.state_to_gridState.append([i, j])

        self.discount = 0.9  # discount rate
        self.learning_rate = 0.01
        self.epsilon = 0.1            # ε-soft policies
        self.episode_number = 10000
        self.step_number = 100        # step number in each episode



    def sarsa_run(self, runTime):
        print('----------- Sarsa -----------')
        total_run = np.zeros(self.episode_number)

        for r in tqdm(range(runTime)):

            running_reward = None
            running_reward_list = []
            self.initialization()
            for episode in range(self.episode_number):
                total_reward = 0
                state_grid = self.env.reset()
                # self.env.render()
                while True:
                    state_index = self.state_to_gridState.index(state_grid)
                    action = self.choose_action_e_greedy(state_index)            # 行动策略为ε-贪婪策略
                    next_state, reward, done= self.env.step(state_grid, action)
                    total_reward += reward

                    next_state_index = self.state_to_gridState.index(next_state)
                    self.learn('sarsa', state_index, action, reward, next_state_index)

                    state_grid = next_state
                    self.env.current_state = state_grid
                    # self.env.render()
                    if done:
                        # time.sleep(0.5)
                        break
                # 每一次迭代获得的总收获total_reward,会以0.01的份额加入到running_reward。(原代码这里rAll用了r，个人认为是total_reward更合适)
                running_reward = total_reward if running_reward is None else running_reward * 0.99 + total_reward * 0.01
                running_reward_list.append(running_reward)
            total_run += np.asarray(running_reward_list)
        total_run /= runTime
        return total_run


    def learn(self, method, current_state, action, reward, next_state):
        q_predict = self.Q_table[current_state][action]
        if next_state not in self.env.terminate_space:
            if method == 'sarsa':
                next_action = self.choose_action_e_greedy(next_state)   # 目标策略也用ε-贪婪策略
                q_target = reward + self.discount * self.Q_table[next_state][next_action]
            else: # method == 'Q-learning'
                q_target = reward + self.discount * np.max(self.Q_table[next_state])
        else:
            q_target = reward
        self.Q_table[current_state][action] += self.learning_rate * (q_target - q_predict)


    def Q_learning_run(self, runTime):
        print('----------- Q-learning -----------')
        total_run = np.zeros(self.episode_number)

        for r in tqdm(range(runTime)):
            running_reward = None
            running_reward_list = []
            self.initialization()
            for episode in range(self.episode_number):
                total_reward = 0
                state_grid = self.env.reset()
                self.env.render()
                while True:
                    state_index = self.state_to_gridState.index(state_grid)
                    action = self.choose_action_e_greedy(state_index)
                    next_state, reward, done = self.env.step(state_grid, action)
                    total_reward += reward

                    next_state_index = self.state_to_gridState.index(next_state)
                    self.learn('Q-learning', state_index, action, reward, next_state_index)

                    state_grid = next_state
                    self.env.current_state = state_grid
                    self.env.render()
                    if done:
                        time.sleep(0.5)
                        break
                # 每一次迭代获得的总收益total_reward,会以0.01的份额加入到running_reward。(原代码这里total_reward用了r，个人认为是total_reward更合适)
                running_reward = total_reward if running_reward is None else running_reward * 0.99 + total_reward * 0.01
                running_reward_list.append(running_reward)
            total_run += np.asarray(running_reward_list)
                # print('Episode [{}/{}] | Total reward: {} | Running reward: {:5f}'.
                #       format(episode, self.episode_number, total_reward, running_reward))
        total_run /= runTime
        return total_run


    def choose_action_e_greedy(self, state):
            if np.random.random() > self.epsilon:  # 选择最大Q值的动作
                action = self.getRandomAction_withSame_Q(state)
            else:  # 随机选择动作
                action = np.random.randint(0, self.env.nA)
            return action


    # 如果该状态的所有q值都是相等的，那么当选择最大的值的index时，总会选择第一个
    # 编写函数，若有相同的最大值，则随机返回最大值中对应的动作
    def getRandomAction_withSame_Q(self, current_state):
        action_Q_list = self.Q_table[current_state]

        max_Q_index = []
        max_Q = np.max(action_Q_list)
        for i in range(len(action_Q_list)):
            if action_Q_list[i] == max_Q:
                max_Q_index.append(i)
        return np.random.choice(max_Q_index)


    def initialization(self):
        for _ in range(self.env.nS):
            self.Q_table.append([0.] * self.env.nA)



if __name__ == '__main__':
    env = Maze_Env.MazeEnv()
    runTime = 1  # perform 20 times independently

    fl = MazeGame(env)
    # sarsa = fl.sarsa_run(runTime=runTime)
    Qlearning = fl.Q_learning_run(runTime=runTime)

    # plt.plot(sarsa, label='Sarsa')
    plt.plot(Qlearning, label='Q-learning')
    plt.xlabel('Episodes')
    plt.ylabel('Running reward')
    plt.legend()
    plt.savefig('Sarsa_Qlearning_maze_game.png')
    plt.close()
