import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


class MDP:
    def __init__(self, S, A, T, R, gamma):
        self.S = S
        self.A = A
        self.T = T
        self.R = R
        self.gamma = gamma

        self.V = np.zeros(len(self.S))
        self.policy = np.zeros(len(self.S))
        self.theta = np.nextafter(0, 1)
        self.state = self.S[0]

        # sanity checks:
        assert T.shape == (
            len(self.A),
            len(self.S),
            len(self.S),
        )  # action x state x state
        assert R.shape == (
            len(self.S),
            len(self.A),
            len(self.S),
        )  # state x action x next_state

    def bellman_eq(self, state):
        vals = np.zeros(len(self.A))

        for action in self.A:
            to_sum = []
            for p in range(len(self.T[action][state])):
                to_sum.append(
                    self.T[action][state][p]
                    * (self.R[state][action][p] + (self.gamma * self.V[p]))
                )

            vals[action] = sum(to_sum)

        def check_action(state, length):
            if state == 0:  # left-border
                vals[0] = np.NINF
            if state == length - 1:  # right-border
                vals[1] = np.NINF

        check_action(state, len(self.S))

        return vals

    def value_iteration(self):
        while True:
            difference = 0
            for state in self.S:
                old_V = self.V[state]
                v = self.bellman_eq(state)

                self.policy[state] = np.argmax(v)
                self.V[state] = np.max(v)

                difference = max(difference, np.abs(old_V - self.V[state]))

            if difference < self.theta:
                break

        # print(self.V)

    def solve(
        self,
        setup_name="Placeholder Setup Name",
        policy_name="Placeholder Policy Name",
        heatmap=True,
    ):
        self.value_iteration()

        if len(self.policy) > 0:
            grid = []
            precision = 2
            for state in self.S:
                if self.policy[state] == 0:
                    grid.append("\u2190" + "\n " + str(round(self.V[state], precision)))
                else:
                    grid.append("\u2192" + "\n " + str(round(self.V[state], precision)))

            # draw_grid = ''.join(grid)
            # arrows_list = draw_grid.split()

            labels = np.array(grid)

            # reformat the list of arrows into a correctly-shaped array to add to heatmap
            # labels = np.array(arrows_list)
            labels = np.reshape(labels, (1, len(labels)))

            # draw heatmap and save in figure
            if heatmap:
                hmap = sns.heatmap(
                    np.reshape(self.V, (1, len(self.V))),
                    annot=labels,
                    fmt="",
                    yticklabels=False,
                    cbar=False,
                    cbar_kws={"label": "Value"},
                    annot_kws={"size": 25 / np.sqrt(len(self.V))},
                )
                hmap.set(xlabel="State", title=f"{policy_name} Value Iteration")
                hmap = hmap.figure
                file_name = policy_name.replace(" ", "_").lower()
                setup_name = setup_name.replace(" ", "_").lower()
                print(file_name)
                plt.savefig(f"images/{setup_name}/{file_name}.png")
                plt.clf()

        return self.V, self.policy

    def reset(self):
        self.state = self.S[0]


class Experiment_1D:
    def __init__(
        self, length, make_right_prob=0.8, rewards_dict={-1: 10, -2: -1}, gamma=0.9
    ):
        # storing these variables as self for reset() function.
        fixed_rewards_dict = {}
        for idx in rewards_dict:
            if not (idx >= 0 and idx < length):
                fixed_rewards_dict[idx % length] = rewards_dict[idx]
            else:
                fixed_rewards_dict[idx] = rewards_dict[idx]
        rewards_dict = fixed_rewards_dict

        self.make_right_prob = make_right_prob
        self.length = length
        self.gamma = gamma
        self.rewards_dict = rewards_dict

        self.S, self.A, self.T, self.R, self.gamma = self.make_MDP_params(
            length, make_right_prob, rewards_dict, gamma
        )
        self.mdp = MDP(self.S, self.A, self.T, self.R, self.gamma)

    def reset(self):
        self.S, self.A, self.T, self.R, self.gamma = self.make_MDP_params(
            self.length, self.make_right_prob, self.rewards_dict, self.gamma
        )
        self.mdp = MDP(self.S, self.A, self.T, self.R, self.gamma)

    def make_MDP_params(self, length, make_right_prob, rewards_dict, gamma):
        S = np.arange(length)
        A = np.array((0, 1))  # 0 is left and 1 is right

        T = np.zeros((2, length, length))

        T[0] = np.diag(np.array([1] + [1 - make_right_prob] * (length - 2) + [1]))

        for i in range(1, length - 1):
            T[0, i, i - 1] = make_right_prob

        T[1] = np.diag(np.array([1 - make_right_prob] * (length - 1) + [1]))

        for i in range(0, length - 1):
            T[1, i, i + 1] = make_right_prob

        def make_absorbing(idx):
            for i in range(2):
                for j in range(length):
                    T[i, idx, j] = 0

        # make reward states absorbing
        for idx in rewards_dict:
            if rewards_dict[idx] > 0:
                make_absorbing(idx)

        R = np.zeros((length, 2, length))
        # variable to keep track of default reward magnitude
        # default_R_magnitude = 100

        def assign_reward(idx, magnitude):
            if idx - 1 in list(range(length)):
                R[idx - 1, 1, idx] = magnitude
            if idx + 1 in list(range(length)):
                R[idx + 1, 0, idx] = magnitude

        for idx in rewards_dict:
            assign_reward(idx, rewards_dict[idx])

        # R[length - 2, 1, length - 1] = default_R_magnitude

        # R[neg_idx - 1, 1, neg_idx] = neg_magnitude
        # R[neg_idx + 1, 0, neg_idx] = neg_magnitude

        return S, A, T, R, gamma

    def myopic(self, gamma):
        self.mdp = MDP(self.S, self.A, self.T, self.R, gamma)

    def confident(self, make_right_prob):
        # probability is LOWER than the "true": UNDERCONFIDENT
        S, A, T, R, gamma = self.make_MDP_params(
            self.length, make_right_prob, self.rewards_dict, self.gamma
        )
        self.mdp = MDP(S, A, T, R, gamma)

    def reward(self, agent_R_idx, agent_R_magnitude, ignore_default_R):
        pass


if __name__ == "__main__":
    length = 10
    default_prob = 0.8
    sns.set()

    # our baseline:
    test = Experiment_1D(length, default_prob)
    test.mdp.solve("Baseline World")
    neg_idx = 8
    neg_magnitude = -10
    test.mdp.reset()

    # REWARD AGENT RUNS:
    reward_agent_R_val = 50
    reward = test.reward(2, reward_agent_R_val, False)
    test.mdp.solve("Reward Agent: reward value={}".format(reward_agent_R_val))
    test.mdp.reset()

    # MYOPIC EXPERIMENT RUNS:
    for gamma in np.arange(0.01, 1, 0.1):
        test.mdp.reset()
        myopic = test.myopic(gamma=gamma)
        test.mdp.solve("Myopic Agent: \u03B3={:.3f}".format(gamma))

    # UNDERCONFIDENT + OVERCONFIDENT EXPERIMENT RUNS:
    for prob in np.arange(0.01, 1, 0.1):
        test.mdp.reset()
        confident = test.confident(make_right_prob=prob)
        if prob < default_prob:
            test.mdp.solve("Underconfident Agent: p={:.3f}".format(prob))
        elif prob > default_prob:
            test.mdp.solve("Overconfident Agent: p={:.3f}".format(prob))


"""

TODO: 

- Make conjectures for agent and see how the behavior goes against those conjectures for the writeup

- Look at how to differentiate behavior between the agents 
    - Can we tell what a myopic agent looks like vs, say, under/over-confident agents.

- Different incrememts, size of world, diff knobs to choose based on result looking for and experiments we want to execute.

- Basically: go back to main question of understanding the differences between the irrational agents based on behavior.
    - Design experiments around this question!

- Possibly layout the different PNGs together to see the layout in ust one picture.


- **TODO BY NEXT WEEK**:
    - Visualize the pattern based on experiments chosen
    - Writeup results on whether we can differentiate between the different agents on their world.

- To consider:
    - Workshop for next semester. Need to submit work to be considered...
    - ICML conference (July) -> workshops attatched to conference.
        - March details released -> due date in May.

"""
