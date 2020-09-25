import numpy as np
import gym, time
import matplotlib.pylab as plt
from tqdm import tqdm

class FronzenLake_v0:
    def __init__(self, env):
        self.env = env
        self.theta = 1e-4  # A small positive number determining the accuracy of estimation

        self.pi = []
        self.Q_table = []
        self.terminate_state_list = [5, 7, 11, 12, 15]

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
                state = self.env.reset()
                for step in range(self.step_number):
                    action = self.choose_action_e_greedy(state)            # 行动策略为ε-贪婪策略
                    next_state, reward, done, _ = self.env.step(action)
                    total_reward += reward

                    next_action = self.choose_action_e_greedy(next_state)  # 目标策略也是ε-贪婪策略
                    self.Q_table[state][action] += self.learning_rate * \
                                           (reward + self.discount * self.Q_table[next_state][next_action] - self.Q_table[state][action])
                    state = next_state
                    if done:
                        break
                # 每一次迭代获得的总收获total_reward,会以0.01的份额加入到running_reward。(原代码这里rAll用了r，个人认为是total_reward更合适)
                running_reward = total_reward if running_reward is None else running_reward * 0.99 + total_reward * 0.01
                running_reward_list.append(running_reward)
            total_run += np.asarray(running_reward_list)


                # print('Episode [{}/{}] | Total reward: {} | Running reward: {:5f}'.
                #       format(episode, self.episode_number, total_reward, running_reward))
        total_run /= runTime
        return total_run
        # pi = [np.argmax(self.Q_table[s]) for s in range(self.env.nS)]


    def Q_learning_run(self, runTime):
        print('----------- Q-learning -----------')
        total_run = np.zeros(self.episode_number)

        for r in tqdm(range(runTime)):
            running_reward = None
            running_reward_list = []
            self.initialization()
            for episode in range(self.episode_number):
                total_reward = 0
                state = self.env.reset()
                for step in range(self.step_number):
                    action = self.choose_action_e_greedy(state)
                    next_state, reward, done, _ = self.env.step(action)
                    total_reward += reward

                    self.Q_table[state][action] += self.learning_rate * \
                                                   (reward + self.discount * np.max(self.Q_table[next_state]) - self.Q_table[state][action])
                    state = next_state
                    if done:
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
                action = np.argmax(self.Q_table[state])
            else:  # 随机选择动作
                action = np.random.randint(0, self.env.action_space.n)
            return action


    def initialization(self):
        for s in range(self.env.nS):
            if s not in self.terminate_state_list:
                s_table = [1. for _ in range(self.env.action_space.n)]
            else:
                s_table = [0. for _ in range(self.env.action_space.n)]
            self.Q_table.append(s_table)



if __name__ == '__main__':
    env = gym.make('FrozenLake-v0')
    runTime = 50  # perform 20 times independently

    fl = FronzenLake_v0(env)
    sarsa = fl.sarsa_run(runTime=runTime)
    time.sleep(1)
    Qlearning = fl.Q_learning_run(runTime=runTime)


    plt.plot(sarsa, label='Sarsa')
    plt.plot(Qlearning, label='Q-learning')
    plt.xlabel('Episodes')
    plt.ylabel('Running reward')
    plt.legend()
    plt.savefig('Sarsa_Qlearning_FrozenLake_v0.png')
    plt.close()
