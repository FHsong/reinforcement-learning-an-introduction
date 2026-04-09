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
        self.episode_number = 100
        self.step_number = 100        # step number in each episode



    def sarsa_run(self):
        print('----------- Sarsa -----------')
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
        return running_reward_list



    def learn(self, method, current_state, action, reward, next_state):
        q_predict = self.Q_table[current_state][action]
        if next_state not in self.env.terminate_space:
            if method == 'sarsa':
                next_action = self.choose_action_e_greedy(next_state)   # 目标策略也用ε-贪婪策略
                q_target = reward + self.discount * self.Q_table[next_state][next_action]
            else: # 'Q-learning'
                q_target = reward + self.discount * np.max(self.Q_table[next_state])
        else:
            q_target = reward
        self.Q_table[current_state][action] += self.learning_rate * (q_target - q_predict)


    def Q_learning_run(self):
        print('----------- Q-learning -----------')
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
                    time.sleep(1)
                    break
            # 每一次迭代获得的总收益total_reward,会以0.01的份额加入到running_reward。(原代码这里total_reward用了r，个人认为是total_reward更合适)
            running_reward = total_reward if running_reward is None else running_reward * 0.99 + total_reward * 0.01
            running_reward_list.append(running_reward)
            print('Episode [{}/{}] | Total reward: {} | Running reward: {:5f}'.
                  format(episode, self.episode_number, total_reward, running_reward))
        return running_reward_list


    def choose_action_e_greedy(self, state):
            if np.random.random() > self.epsilon:  # 选择最大Q值的动作
                # 若有多个相同的最大Q值，则随机选择
                action = np.random.choice(
                    [action_ for action_, value_ in enumerate(self.Q_table[state]) if value_ == np.max(self.Q_table[state])])
            else:  # 随机选择动作
                action = np.random.randint(0, self.env.nA)
            return action


    def initialization(self):
        self.Q_table= [[0.] * self.env.nA for _ in range(self.env.nS)]



if __name__ == '__main__':
    env = Maze_Env.MazeEnv()

    fl = MazeGame(env)
    # sarsa = fl.sarsa_run()
    Qlearning = fl.Q_learning_run()

    # plt.plot(sarsa, label='Sarsa')
    plt.plot(Qlearning, label='Q-learning')
    plt.xlabel('Episodes')
    plt.ylabel('Running reward')
    plt.legend()
    plt.savefig('Sarsa_Qlearning_maze_game.png')
    plt.close()
