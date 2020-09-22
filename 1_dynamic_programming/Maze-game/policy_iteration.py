import numpy as np
from maze_env import MazeEnv
import time


class PolicyIteration:
    def __init__(self, env):
        self.env = env
        self.theta = 1e-4  # A small positive number determining the accuracy of estimation
        self.discount = 1
        self.state_values = None
        self.pi = None


    def run(self):
        state_values = self.initialization()

        iteration = 1
        policy_stable = False
        while policy_stable != True:
            print('Iteration {} \n'.format(iteration))
            new_state_values = self.policy_evaluation(state_values, self.pi)
            policy_stable = self.policy_improvement(new_state_values, self.pi)

            state_values = new_state_values

            iteration += 1

        print(self.pi)
        # 渲染游戏
        self.env.reset()
        self.env.render()
        while True:
            time.sleep(1)
            next_state, reward, done= self.env.step(self.env.current_state, self.pi[self.env.current_state[0], self.env.current_state[1]])
            self.env.current_state = next_state
            self.env.render()
            if done:
                break


    def policy_evaluation(self, state_values, pi):
        while True:
            new_state_values = state_values
            old_state_values = state_values.copy()

            for i in range(self.env.GRID_SIZE):
                for j in range(self.env.GRID_SIZE):
                    if [i, j] not in self.env.terminate_space:
                        [next_i, next_j], reward, done = self.env.step([i, j], pi[i, j])
                        new_state_values[i, j] = 0.1 * (reward + self.discount * state_values[next_i, next_j])

            max_delta_value = abs(old_state_values - new_state_values).max()
            if max_delta_value < 1e-4:
                break
        return new_state_values


    def policy_improvement(self, state_values, pi):
        policy_stable = True
        for i in range(self.env.GRID_SIZE):
            for j in range(self.env.GRID_SIZE):
                if [i, j] not in self.env.terminate_space:
                    old_action = pi[i, j]
                    value_list = []
                    for action in range(self.env.nA):
                        [next_i, next_j], reward, done = self.env.step([i, j], action)
                        value = (reward + self.discount * state_values[next_i, next_j])
                        value_list.append(value)

                    pi[i, j] = np.argmax(value_list)
                    if pi[i, j] != old_action:
                        policy_stable = False
        return policy_stable




    def initialization(self):
        state_values = np.zeros((self.env.GRID_SIZE, self.env.GRID_SIZE))
        self.pi = np.zeros((self.env.GRID_SIZE, self.env.GRID_SIZE), dtype=int)  # Set initial policy to moving left
        return state_values


if __name__ == '__main__':
    env = MazeEnv()
    policy_iteration = PolicyIteration(env)
    policy_iteration.run()