import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


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
        self.theta = np.nextafter(0, 1)
        self.state = self.S[0][0]

        # sanity checks:
        assert T.shape == (
            len(self.A),
            self.height * self.width,
            self.height * self.width,
        )  # action x state x state
        assert R.shape == (
            self.height * self.width,
            len(self.A),
            self.height * self.width,
        )  # state x action x next_state

    def bellman_eq(self, state):
        vals = np.zeros(len(self.A))

        # TODO: Think about: IF ACTION IMPOSSIBLE, ASSIGN np.NINF value -- do this by if the sum of the self.T[action][state] = 0 then do this
        for action in self.A:
            to_sum = []
            for p in range(len(self.T[action][state])):
                to_sum.append(
                    self.T[action][state][p]
                    * (
                        self.R[state][action][p]
                        + (self.gamma * self.V[p // self.width][p % self.width])
                    )
                )

            vals[action] = sum(to_sum)

        def check_action(state, width, height):
            if state % width == 0:  # left-border
                vals[0] = np.NINF
            if state % width == width - 1:  # right-border
                vals[1] = np.NINF
            if state < width:  # top
                vals[2] = np.NINF
            if state >= width * (height - 1):  # bottom
                vals[3] = np.NINF

        check_action(state, self.width, self.height)

        return vals

    def value_iteration(self):
        while True:
            difference = 0
            for row in self.S:
                for state in row:
                    old_V = self.V[state // self.width][state % self.width]
                    v = self.bellman_eq(state)

                    self.policy[state // self.width][state % self.width] = np.argmax(v)
                    self.V[state // self.width][state % self.width] = np.max(v)

                    difference = max(
                        difference,
                        np.abs(old_V - self.V[state // self.width][state % self.width]),
                    )

            if difference < self.theta:
                break

    def save_heatmap(self, setup_name, policy_name, labels):
        # draw heatmap and save in figure
        hmap = sns.heatmap(
            self.V,
            annot=labels,
            fmt="",
            xticklabels=False,
            yticklabels=False,
            cbar=False,
            cbar_kws={"label": "Value"},
            annot_kws={"size": 25 / np.sqrt(len(self.V))},
        )
        hmap.set(xlabel="States", title=f"{policy_name} Value Iteration")
        hmap = hmap.figure
        file_name = policy_name.replace(" ", "_").lower()
        setup_name = setup_name.replace(" ", "_").lower()
        print(file_name)
        plt.savefig(f"images/{setup_name}/{file_name}.png")
        plt.clf()

    def solve(
        self,
        setup_name="Placeholder Setup Name",
        policy_name="Placeholder Policy Name",
        save_heatmap=True,
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
                self.save_heatmap(setup_name, policy_name, labels)

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
    ):
        fixed_rewards_dict = {}
        for idx in rewards_dict:
            if not (idx >= 0 and idx < width * height):
                fixed_rewards_dict[idx % (width * height)] = rewards_dict[idx]
            else:
                fixed_rewards_dict[idx] = rewards_dict[idx]
        rewards_dict = fixed_rewards_dict

        self.S, self.A, self.T, self.R, self.gamma = self.make_MDP_params(
            height, width, action_success_prob, rewards_dict, gamma
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

            # Update transition probabilities
            T[action, i, target] = action_success_prob

            # Calculate remaining probability
            remaining_prob = (1 - action_success_prob) / 4

            for d in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                dr, dc = d
                r, c = row + dr, col + dc
                if in_bounds(r, c):
                    neighbor = r * width + c
                    T[action, i, neighbor] += remaining_prob
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

    def make_MDP_params(
        self,
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

        self._fill_transition_matrix(
            T, A, width, height, action_success_prob, mode=transition_mode
        )

        def make_absorbing(idx):
            for i in range(4):
                for j in range(width * height):
                    T[i, idx, j] = int(idx == j)

        # make reward states absorbing
        for idx in rewards_dict:
            if rewards_dict[idx] > 0:
                make_absorbing(idx)

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
        self.mdp = MDP_2D(self.S, self.A, self.T, self.R, gamma)

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

    def pessimistic(self, scaling, new_gamma=None):
        S, A, T, R, gamma = self.make_MDP_params(
            self.height,
            self.width,
            self.action_success_prob,
            self.rewards_dict,
            self.gamma,
            transition_mode="simple",
        )

        # Change the transition probabilities to be more pessimistic
        neg_rew_idx = [idx for idx in self.rewards_dict if self.rewards_dict[idx] < 0]

        T[:, :, neg_rew_idx] *= scaling
        T /= T.sum(axis=2, keepdims=True)

        if new_gamma is not None:
            gamma = new_gamma

        self.mdp = MDP_2D(S, A, T, R, gamma)

    def pessimistic_new(self, scaling, new_gamma=None):
        S, A, T, R, gamma = self.make_MDP_params(
            self.height,
            self.width,
            self.action_success_prob,
            self.rewards_dict,
            self.gamma,
            transition_mode="full",
        )

        # Change the transition probabilities to be more pessimistic
        neg_rew_idx = [idx for idx in self.rewards_dict if self.rewards_dict[idx] < 0]

        T[:, :, neg_rew_idx] *= scaling
        T /= T.sum(axis=2, keepdims=True)

        if new_gamma is not None:
            gamma = new_gamma

        self.mdp = MDP_2D(S, A, T, R, gamma)

    def reward(self, agent_R_idx, agent_R_magnitude, ignore_default_R):
        S, A, T, R, gamma = self.make_MDP_params(
            self.height,
            self.width,
            self.action_success_prob,
            self.rewards_dict,
            self.gamma,
        )
        R[agent_R_idx - 1, 1, agent_R_idx] = agent_R_magnitude
        R[agent_R_idx + 1, 0, agent_R_idx] = agent_R_magnitude

        if ignore_default_R:
            R[length - 2, 1, length - 1] = 0

        self.mdp = MDP_2D(S, A, T, R, gamma)
