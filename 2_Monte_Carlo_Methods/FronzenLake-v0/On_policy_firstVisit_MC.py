import numpy as np
import gym

class MC_ES:
    def __init__(self, env):
        self.env = env
        self.theta = 1e-4  # A small positive number determining the accuracy of estimation
        self.discount = 1
        self.pi = []
        self.Q_table = []
        self.Returns = []

        self.state_action_count_list = []

        self.episode_number = 1000000

        self.state_list = None
        self.action_list = None
        self.reward_list = None

        self.epsilon = 0.1            # ε-soft policies



    def run(self):
        self.initialization()
        for episode in range(1, self.episode_number):
            self.generate_one_episode()
            # print(self.state_list)
            # print(self.reward_list, '\n')

            state_action_list = []
            for t in range(len(self.state_list)):
                (s, a) = (self.state_list[t], self.action_list[t])
                state_action_list.append((s, a))

            G = 0.
            for t in range(len(self.state_list)-1, -1, -1):
                G = self.discount * G + self.reward_list[t]

                state_action_list.pop()
                S_t = self.state_list[t]
                A_t = self.action_list[t]
                if (S_t, A_t) not in state_action_list:
                    # self.Returns[S_t][A_t].append(G)
                    # self.Q_table[S_t][A_t] = np.average(self.Returns[S_t][A_t])
                    self.state_action_count_list[S_t][A_t] += 1
                    self.Q_table[S_t][A_t] = self.Q_table[S_t][A_t] + \
                                             (1. / self.state_action_count_list[S_t][A_t]) * (G - self.Q_table[S_t][A_t])

                A_star = np.argmax(self.Q_table[S_t])
                for a in range(self.env.action_space.n):
                    if a == A_star:
                        self.pi[S_t][a] = 1 - self.epsilon + self.epsilon / self.env.action_space.n
                    else:
                        self.pi[S_t][a] = self.epsilon / self.env.nA

            print('Episode: {} | Total reward: {}'.format(episode, G))
        print(self.pi)
        print([np.argmax(l) for l in self.pi])




    def generate_one_episode(self):
        self.state_list = []
        self.action_list = []
        self.reward_list = []
        state = self.env.reset()
        while True:
            # env.render()
            action_P = np.array(self.pi[state])
            action = np.random.choice(np.arange(self.env.action_space.n), p=action_P.ravel())
            next_state, reward, done, _ = self.env.step(action)
            self.load_one_transmission(state, action, reward)
            state = next_state
            if done:
                break



    def load_one_transmission(self, state, action, reward):
        self.state_list.append(state)
        self.action_list.append(action)
        self.reward_list.append(reward)



    def initialization(self):
        self.pi = [[1 - self.epsilon + self.epsilon / self.env.env.action_space.n] for _ in range(self.env.nS)]  # 选中贪心策略的概率
        for s in range(self.env.nS):
            for a in range(self.env.env.action_space.n - 1):
                self.pi[s].append(self.epsilon / self.env.env.action_space.n)

        for s in range(self.env.nS):
            s_table = [1. for _ in range(self.env.action_space.n)]
            self.Q_table.append(s_table)

            a_list = [[] for _ in range(self.env.action_space.n)]
            self.Returns.append(a_list)

            s_a_count = [0 for _ in range(self.env.action_space.n)]
            self.state_action_count_list.append(s_a_count)


if __name__ == '__main__':
    env = gym.make('FrozenLake-v0')
    # env.reset()
    # for _ in range(100):
    #     next_state, reward, done, _ = env.step(0)
    #     print(next_state, reward)
    mc_es = MC_ES(env)
    mc_es.run()
