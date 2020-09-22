import numpy as np
from maze_env import MazeEnv
import time


class ValueIteration:
    def __init__(self, env):
        self.env = env
        self.theta = 1e-4  # A small positive number determining the accuracy of estimation
        self.discount = 1
        self.state_values = None
        self.pi = None


    def run(self):
        state_values = self.initialization()
        while True:
            old_state_values = state_values.copy()
            for i in range(self.env.GRID_SIZE):
                for j in range(self.env.GRID_SIZE):
                    if [i, j] not in self.env.terminate_space:
                        value_list = []
                        for action in range(self.env.nA):
                            [next_i, next_j], reward, done = self.env.step([i, j], action)
                            value = 0.1 * (reward + self.discount * state_values[next_i, next_j])
                            value_list.append(value)
                        state_values[i, j] = np.max(value_list)

            max_delta_value = abs(old_state_values - state_values).max()
            if max_delta_value < 1e-4:
                break

        # Output a deterministic policy
        for i in range(self.env.GRID_SIZE):
            for j in range(self.env.GRID_SIZE):
                if [i, j] not in self.env.terminate_space:
                    value_list = []
                    for action in range(self.env.nA):
                        [next_i, next_j], reward, done = self.env.step([i, j], action)
                        value = 0.1 * (reward + self.discount * state_values[next_i, next_j])
                        value_list.append(value)
                    self.pi[i, j] = np.argmax(value_list)
        print(self.pi)


    def initialization(self):
        state_values = np.zeros((self.env.GRID_SIZE, self.env.GRID_SIZE))
        self.pi = np.zeros((self.env.GRID_SIZE, self.env.GRID_SIZE), dtype=int)  # Set initial policy to moving left
        return state_values


if __name__ == '__main__':
    env = MazeEnv()
    value_iteration = ValueIteration(env)
    value_iteration.run()