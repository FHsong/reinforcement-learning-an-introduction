import numpy as np
import matplotlib.pylab as plt
from tqdm import tqdm

# 0 is the left terminal state
# 6 is the right terminal state
# 1 ... 5 represents A ... E
INITIAL_VALUES = np.zeros(7)
INITIAL_VALUES[1:6] = 0.5
# For convenience, we assume all rewards are 0
# and the left terminal state has value 0, the right terminal state has value 1
# This trick has been used in Gambler's Problem
INITIAL_VALUES[6] = 1

# set up true state values
TRUE_VALUE = np.zeros(7)
TRUE_VALUE[1:6] = np.arange(1, 6) / 6.0
TRUE_VALUE[6] = 1

ACTION_LEFT = 0
ACTION_RIGHT = 1


class RandomWalk:
    def __init__(self):
        self.discount = 1



    def temporal_difference(self, values, alpha=0.1, batch=False):
        state = 3
        trajectory = [3]
        rewards = [0]

        while True:
            old_state = state
            if np.random.binomial(1, 0.5) == ACTION_LEFT:
                state -= 1
            else:
                state += 1
            trajectory.append(state)
            reward = 0

            if not batch:
                values[old_state] += alpha * (reward + values[state] - values[old_state])

            if state == 0 or state == 6:
                break
            rewards.append(reward)

        return trajectory, rewards


    def monte_carlo(self, values, alpha=0.1, batch=False):
        state = 3
        trajectory = [3]

        while True:
            if np.random.binomial(1, 0.5) == ACTION_LEFT:
                state -= 1
            else:
                state += 1
            trajectory.append(state)
            if state == 6:
                returns = 1.
                break
            elif state == 0:
                returns = 0.
                break

        if not batch:
            for state_ in trajectory[:-1]:
                values[state_] += alpha * (returns - values[state_])
        return trajectory, [returns] * (len(trajectory) - 1)




    # Left figure in Example 6.2 left
    def compute_state_value(self):
        marker_list = ['+', 'o', '<', 'v', 'x', 'd']
        color_list = ['m', 'y', 'g', 'c', 'r', 'k']

        episodes = [0, 1, 10 , 100]
        current_values = np.copy(INITIAL_VALUES)
        k = 0
        for i in range(episodes[-1] + 1):
            if i in episodes:
                plt.plot(current_values,
                               marker=marker_list[k], markersize=6, color=color_list[k], linestyle='-', label=str(i) + ' episodes')
                # plt.xticks(current_values, ['0', 'A', 'B', 'C', 'D', 'E', '6'])
                k += 1
            self.temporal_difference(current_values)
        plt.plot(TRUE_VALUE,
                 marker=marker_list[k], markersize=6, color=color_list[k], linestyle='-', label='True values')
        # plt.xticks(current_values, ['0', 'A', 'B', 'C', 'D', 'E', '6'])
        plt.xlabel('State')
        plt.ylabel('Estimated value')
        plt.legend()
        plt.savefig('Example 6.2 left.png')


    def compute_root_mean_squared(self):
        td_alphas = [0.15, 0.1, 0.05]
        mc_alphas = [0.01, 0.02, 0.03, 0.04]
        episodes = 100 + 1
        runs = 100

        for i, alpha in enumerate(td_alphas + mc_alphas):
            total_errors = np.zeros(episodes)
            if i < len(td_alphas):
                method = 'TD'
                linestyle = 'solid'
            else:
                method = 'MC'
                linestyle = 'dashdot'

            for r in tqdm(range(runs)):  # 运行一百次这样的episodes=100，然后取平均
                errors = []
                current_values = np.copy(INITIAL_VALUES)
                for i in range(0, episodes):
                    errors.append(
                        np.sqrt(
                            np.sum(
                                np.power(TRUE_VALUE - current_values, 2)) / 5.0))
                    if method == 'TD':
                        self.temporal_difference(current_values, alpha=alpha)
                    else:
                        self.monte_carlo(current_values, alpha=alpha)
                total_errors += np.asarray(errors)
            total_errors /= runs
            plt.plot(total_errors, linestyle=linestyle, label=method + ', alpha = %.02f' % (alpha))
        plt.xlabel('episodes')
        plt.ylabel('RMS')
        plt.legend()
        plt.savefig('Example 6.2 right.png')



    def batch_updating(self, method, episodes, alpha=0.001):
        # perform 100 runs independently
        runs = 100
        total_errors = np.zeros(episodes)
        for r in tqdm(range(0, runs)):
            current_values = np.copy(INITIAL_VALUES)
            errors = []
            # track shown trajectories and reward/return sequences
            trajectories = []
            rewards = []
            for episode in range(episodes):
                if method == 'TD':
                    trajectory_, rewards_ = self.temporal_difference(current_values, batch=True)
                else:
                    trajectory_, rewards_ = self.monte_carlo(current_values, batch=True)
                trajectories.append(trajectory_)
                rewards.append(rewards_)

                while True:
                    updates = np.zeros(7)
                    for trajectory_, rewards_ in zip(trajectories, rewards):
                        for i in range(0, len(trajectory_) - 1):
                            if method == 'TD':
                                updates[trajectory_[i]] += rewards_[i] + current_values[trajectory_[i+1]] - current_values[trajectory_[i]]
                            else:
                                updates[trajectory_[i]] += rewards_[i] - current_values[trajectory_[i]]
                    updates *= alpha
                    if np.sum(np.abs(updates)) < 1e-3:
                        break
                    # perform batch updating
                    current_values += updates
                    # calculate rms error
                errors.append(np.sqrt(np.sum(np.power(current_values - TRUE_VALUE, 2)) / 5.0))
            total_errors += np.asarray(errors)
        total_errors /= runs
        return total_errors


    def figure_6_2(self):
        episodes = 100 + 1
        td_erros = self.batch_updating('TD', episodes)
        mc_erros = self.batch_updating('MC', episodes)

        plt.plot(td_erros, label='TD')
        plt.plot(mc_erros, label='MC')
        plt.xlabel('episodes')
        plt.ylabel('RMS error')
        plt.legend()

        plt.savefig('figure_6_2.png')
        plt.close()



if __name__ == '__main__':
    rw = RandomWalk()
    # rw.compute_state_value()
    # rw.compute_root_mean_squared()
    rw.figure_6_2()