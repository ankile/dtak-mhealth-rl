from datetime import datetime
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from utils.transition_matrix import make_absorbing, transition_matrix_is_valid


class MDP_2D:
    def __init__(self, S, A, T, R, gamma):
        self.S = S
        self.A = A
        self.T = T
        self.R = R
        self.gamma = gamma

        self.height = self.S.shape[0]
        self.width = self.S.shape[1]

        self.V = np.zeros(self.S.shape)
        self.policy = np.zeros(self.S.shape)
        self.theta = 0.0001
        self.state = self.S[0][0]

        # sanity checks:
        assert T.shape == (
            len(self.A),
            self.height * self.width,
            self.height * self.width,
        )  # action x state x state

        # Check if transition probabilities are valid
        assert transition_matrix_is_valid(
            T
        ), "Your matrix of transition probabilities is not valid."

        assert R.shape == (
            self.height * self.width,
            len(self.A),
            self.height * self.width,
        )  # state x action x next_state

    def bellman_eq(self, state):
        row, col = state // self.width, state % self.width
        vals = np.zeros(len(self.A))

        for action in self.A:
            transition_probs = np.array(self.T[action][state])
            rewards = np.array(self.R[state][action])
            vals[action] = np.sum(
                transition_probs * (rewards + self.gamma * self.V.flatten())
            )

            # Check if action is possible
            if col == 0 and action == 0:
                vals[action] = np.NINF
            if col == self.width - 1 and action == 1:
                vals[action] = np.NINF
            if row == 0 and action == 2:
                vals[action] = np.NINF
            if row == self.height - 1 and action == 3:
                vals[action] = np.NINF

        return vals

    def value_iteration(self):
        difference = np.inf
        while difference >= self.theta:
            difference = 0
            for state in self.S.flatten():
                row, col = state // self.width, state % self.width
                old_V = self.V[row, col]
                v = self.bellman_eq(state)

                self.policy[row, col] = np.argmax(v)
                self.V[row, col] = np.max(v)

                difference = np.maximum(difference, np.abs(old_V - self.V[row, col]))

    def save_heatmap(self, setup_name, policy_name, labels, base_dir="images"):
        # draw heatmap and save in figure
        hmap = sns.heatmap(
            self.V,
            annot=labels,
            fmt="",
            xticklabels="",
            yticklabels="",
            cbar=False,
            cbar_kws={"label": "Value"},
            annot_kws={"size": 25 / np.sqrt(len(self.V))},
        )
        hmap.set(xlabel="States", title=f"{policy_name} Value Iteration")
        hmap = hmap.figure
        file_name = policy_name.replace(" ", "_").lower()
        setup_name = setup_name.replace(" ", "_").lower()
        print(file_name)
        plt.savefig(f"{base_dir}/{setup_name}/{file_name}_{datetime.now()}.png")
        plt.clf()

    def solve(
        self,
        setup_name="Placeholder Setup Name",
        policy_name="Placeholder Policy Name",
        save_heatmap=True,
        base_dir="images",
    ):
        self.value_iteration()

        if len(self.policy) > 0:
            grid = []
            precision = 3
            for row in self.S:
                grid_row = []
                for state in row:
                    if self.policy[state // self.width][state % self.width] == 0:
                        grid_row.append(
                            "\u2190 \n "
                            + str(
                                round(
                                    self.V[state // self.width][state % self.width],
                                    precision,
                                )
                            )
                        )
                    elif self.policy[state // self.width][state % self.width] == 1:
                        grid_row.append(
                            "\u2192 \n "
                            + str(
                                round(
                                    self.V[state // self.width][state % self.width],
                                    precision,
                                )
                            )
                        )
                    elif self.policy[state // self.width][state % self.width] == 2:
                        grid_row.append(
                            "\u2191 \n "
                            + str(
                                round(
                                    self.V[state // self.width][state % self.width],
                                    precision,
                                )
                            )
                        )
                    else:
                        grid_row.append(
                            "\u2193 \n "
                            + str(
                                round(
                                    self.V[state // self.width][state % self.width],
                                    precision,
                                )
                            )
                        )
                grid.append(grid_row)

            labels = np.array(grid)

            if save_heatmap:
                self.save_heatmap(setup_name, policy_name, labels, base_dir=base_dir)

        return self.V, self.policy

    def reset(self):
        self.state = self.S[0][0]


class Experiment_2D:
    def __init__(
        self,
        height,
        width,
        action_success_prob=0.8,
        rewards_dict={-1: 100, -2: -100, -6: -100, -10: -100},
        gamma=0.9,
        transition_mode="simple",
    ):
        fixed_rewards_dict = {}
        for idx in rewards_dict:
            if not (idx >= 0 and idx < width * height):
                fixed_rewards_dict[idx % (width * height)] = rewards_dict[idx]
            else:
                fixed_rewards_dict[idx] = rewards_dict[idx]
        rewards_dict = fixed_rewards_dict

        self.S, self.A, self.T, self.R, self.gamma = self.make_MDP_params(
            height, width, action_success_prob, rewards_dict, gamma, transition_mode
        )
        self.rewards_dict = rewards_dict
        self.height = height
        self.width = width
        self.gamma = gamma
        self.action_success_prob = action_success_prob
        self.mdp = MDP_2D(self.S, self.A, self.T, self.R, self.gamma)

    @staticmethod
    def _get_target(i, action, width, height):
        row, col = i // width, i % width
        left, right, up, down = i - 1, i + 1, i - width, i + width

        if action == 0:  # left
            target = left if col > 0 else i
        elif action == 1:  # right
            target = right if col < width - 1 else i
        elif action == 2:  # up
            target = up if row > 0 else i
        else:  # down
            target = down if row < height - 1 else i

        return target

    @staticmethod
    def _fill_transition_matrix(
        T, A, width, height, action_success_prob, mode="simple"
    ):
        def _set_probs_for_state_simple(i, action, target):
            if target == i:
                T[action, i, i] = 1
            else:
                T[action, i, target] = action_success_prob
                T[action, i, i] = 1 - action_success_prob

        def _set_probs_for_state(i, action, target):
            def in_bounds(row, col):
                return 0 <= row < height and 0 <= col < width

            row, col = i // width, i % width

            # Update transition probability for intended action
            # Target could end up in same state if action would take agent out of bounds
            T[action, i, target] = action_success_prob

            # Calculate remaining probability
            remaining_prob = (1 - action_success_prob) / 4

            # Update transition probabilities for neighbors
            for d in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                dr, dc = d
                r, c = row + dr, col + dc
                if in_bounds(r, c) and (neighbor := r * width + c) != target:
                    T[action, i, neighbor] = remaining_prob
                else:
                    T[action, i, i] += remaining_prob

        set_functions = {
            "simple": _set_probs_for_state_simple,
            "full": _set_probs_for_state,
        }

        assert mode in set_functions, f"Mode {mode} not supported"
        set_fun = set_functions[mode]

        for action in A:
            for i in range(width * height):
                # Determine the intended target
                target = Experiment_2D._get_target(i, action, width, height)
                set_fun(i, action, target)

    @staticmethod
    def make_MDP_params(
        height,
        width,
        action_success_prob,
        rewards_dict,
        gamma,
        transition_mode="simple",
    ):
        S = np.arange(height * width).reshape(height, -1)
        A = np.array((0, 1, 2, 3))  # 0 is left, 1 is right, 2 is up, 3 is down

        T = np.zeros((A.shape[0], height * width, height * width))

        Experiment_2D._fill_transition_matrix(
            T, A, width, height, action_success_prob, mode=transition_mode
        )

        # make reward states absorbing
        for idx in rewards_dict:
            if rewards_dict[idx] > 0:
                make_absorbing(T, idx)

        # previous state, action, new state
        R = np.zeros((width * height, 4, width * height))

        def assign_reward(idx, magnitude):
            # check right border
            if idx + 1 % width != width and idx + 1 < width * height:
                R[idx + 1, 0, idx] = magnitude
            # check left border
            if idx - 1 % width != width - 1 and idx - 1 >= 0:
                R[idx - 1, 1, idx] = magnitude
            # check bottom border
            if idx <= width * (height - 1) and idx + width < width * height:
                R[idx + width, 2, idx] = magnitude
            # check top border
            if idx >= width and idx - width >= 0:
                R[idx - width, 3, idx] = magnitude

        for idx in rewards_dict:
            assign_reward(idx, rewards_dict[idx])

        return S, A, T, R, gamma

    def myopic(self, gamma):
        self.mdp.gamma = gamma

    def confident(self, action_success_prob):
        # probability is LOWER than the "true": UNDERCONFIDENT
        S, A, T, R, gamma = self.make_MDP_params(
            self.height,
            self.width,
            action_success_prob,
            self.rewards_dict,
            self.gamma,
        )
        self.mdp = MDP_2D(S, A, T, R, gamma)

    def pessimistic(self, scaling, new_gamma=None, transition_mode="simple"):
        mdp = self.mdp
        S, A, T, R, gamma = self.make_MDP_params(
            mdp.height,
            mdp.width,
            self.action_success_prob,
            self.rewards_dict,
            mdp.gamma,
            transition_mode=transition_mode,
        )

        # Change the transition probabilities to be more pessimistic
        neg_rew_idx = [idx for idx in self.rewards_dict if self.rewards_dict[idx] < 0]

        T[:, :, neg_rew_idx] *= scaling
        T /= T.sum(axis=2, keepdims=True)

        if new_gamma is not None:
            gamma = new_gamma

        self.mdp = MDP_2D(S, A, T, R, gamma)
