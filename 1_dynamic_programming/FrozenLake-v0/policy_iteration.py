''' 策略迭代求解冰冻湖 '''
"""
FrozenLake-v0是一个4*4的网络格子，每个格子可以是起始块，目标块、冻结块或者危险块：
S F F F
F H F H
F F F H
H F F G
其中，S是起始位置 F是可通过的冰冻湖 H是必须小心的洞 G是目标
我们的目标是让智能体学习如何从开始块如何行动到目标块上，而不是移动到危险块上。
智能体可以选择向上、向下、向左或者向右移动，同时游戏中还有可能吹来一阵风，将智能体吹到任意的方块上。
    
目标是找到从S到G的最佳路径且不会陷入H
The episode ends when you reach the goal or fall in a hole.
You receive a reward of 1 if you reach the goal, and zero otherwise.
"""
import numpy as np
import math, gym, openpyxl


class PolicyIteration:
    def __init__(self, env):
        self.env = env
        self.theta = 1e-4  # A small positive number determining the accuracy of estimation
        self.discount = 1
        self.state_values = None
        self.pi = None

        '''
        # Write state transmitting matrix to Excel file
        f = openpyxl.Workbook()
        sheet = f.create_sheet()
        for state in range(self.env.nS):
            sheet.cell(state+1, 1, state)
            for action in range(self.env.action_space.n):
                a_str = str(action) + ': ('
                for next_trans in self.env.P[state][action]:
                    trans_prob, next_state, reward, _ = next_trans
                    a_str += str(next_state) + ','
                a_str += ')'
                sheet.cell(state+1, action+2, a_str)
        f.save('state transmitting.xlsx')
        f.close()
        '''


    def run(self, render=True):
        state_values = self.initialization()

        iteration = 1
        policy_stable = False
        while policy_stable != True:
            print('Iteration {} \n'.format(iteration))
            new_state_values = self.policy_evaluation(state_values, self.pi)
            policy_stable = self.policy_improvement(new_state_values, self.pi)
            state = self.env.reset()
            self.env.render()
            while True:
                next_state, reward, done, _ = self.env.step(self.pi[state])
                state = next_state
                if render == True:
                    self.env.render()
                if done:
                    break
            iteration += 1
        return self.pi


    def initialization(self):
        state_values = np.zeros(self.env.nS)
        self.pi = np.zeros(self.env.nS, dtype=int)  # Set initial policy to moving left
        return state_values


    def policy_evaluation(self, state_values, pi):
        while True:
            new_state_values = state_values
            old_state_values = state_values.copy()

            for state in range(self.env.nS):
                action = pi[state]
                # build the value table with the selected action
                value = 0
                for next_trans in self.env.P[state][action]:
                    trans_prob, next_state, reward, _ = next_trans
                    value += trans_prob * (reward + self.discount * state_values[next_state])
                new_state_values[state] = value

            max_delta_value = abs(old_state_values - new_state_values).max()
            if max_delta_value < self.theta:
                break
        return new_state_values


    def policy_improvement(self, state_values, pi):
        policy_stable = True
        for state in range(self.env.nS):
            old_action = pi[state]
            value_list = []
            for action in range(self.env.action_space.n):
                value = 0
                for next_trans in self.env.P[state][action]:
                    trans_prob, next_state, reward, _ = next_trans
                    value += trans_prob * (reward + self.discount * state_values[next_state])
                value_list.append( value)

            pi[state] = np.argmax(value_list)
            if old_action != pi[state]:
                policy_stable = False
        return policy_stable



if __name__ == '__main__':
    env = gym.make('FrozenLake-v0')
    render = True

    policy_iterative = PolicyIteration(env)
    pi = policy_iterative.run(render=True)
    print(pi)

