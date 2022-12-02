import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class MDP():
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
        assert T.shape == (len(self.A), len(self.S), len(self.S)) # action x state x state
        assert R.shape == (len(self.S), len(self.A), len(self.S)) # state x action x next_state

    def bellman_eq(self, state):
        vals = np.zeros(len(self.A))

        for action in self.A:
            to_sum = []
            for p in range(len(self.T[action][state])):
                to_sum.append(self.T[action][state][p] * (self.R[state][action][p] + (self.gamma * self.V[p])))

            vals[action] = sum(to_sum)

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
        
        print(self.V)

    def solve(self, policy_name='Placeholder Policy Name'):
        self.value_iteration()

        if len(self.policy) > 0:
            grid = []
            for state in self.S:
                if self.policy[state] == 0:
                    grid.append(u'\u2190 ')
                else:
                    grid.append(u'\u2192 ')

            toDraw = ''.join(grid)
            arrows_list = toDraw.split()

            # reformat the list of arrows into a correctly-shaped array to add to heatmap
            labels = np.array(arrows_list)
            labels = np.reshape(labels, (1, len(labels)))

            # draw heatmap and save in figure
            hmap = sns.heatmap(np.reshape(self.V, (1, len(self.V))), annot=labels, fmt='', yticklabels=False, cbar_kws={'label': 'Value'})
            hmap.set(xlabel='State', title=f'{policy_name} Value Iteration')
            hmap = hmap.figure
            file_name = policy_name.replace(' ', '_').lower()
            plt.savefig(f'{file_name}.png')
            plt.clf()
            
            print(toDraw)

        return

    def reset(self):
        self.state = self.S[0]


class Experiment_1D():

    def __init__(self, length, make_right_prob = 0.8, neg_idx = 8, neg_magnitude = -1):
        self.S, self.A, self.T, self.R, self.gamma = self.make_MDP_params(length, make_right_prob, neg_idx, neg_magnitude)
        self.neg_idx = neg_idx
        self.neg_magnitude = neg_magnitude
        self.mdp_1d = MDP(self.S, self.A, self.T, self.R, self.gamma)

    def make_MDP_params(self, length, make_right_prob, neg_idx, neg_magnitude):
        S = np.arange(length)
        A = np.array((0, 1)) # 0 is left and 1 is right
        gamma = 0.5

        T = np.zeros((2, length, length))

        T[0] = np.diag(np.array([1] + [1 - make_right_prob] * (length-2) + [1]))

        for i in range(1, length-1):
            T[0, i, i-1] = make_right_prob

        T[1] = np.diag(np.array([1 - make_right_prob] * (length-1) + [1]))

        for i in range(0, length-1):
            T[1, i, i+1] = make_right_prob

        R = np.zeros((length, 2, length))
        R[length - 2, 1, length - 1] = 100
        R[neg_idx - 1, 1, neg_idx] = neg_magnitude
        R[neg_idx + 1, 0, neg_idx] = neg_magnitude

        return S, A, T, R, gamma

    def myopic(self, gamma):
        self.mdp_1d = MDP(self.S, self.A, self.T, self.R, gamma)

    def confident(self, make_right_prob):
        # probability is LOWER than the "true": UNDERCONFIDENT
        S, A, T, R, gamma = self.make_MDP_params(length, make_right_prob, self.neg_idx, self.neg_magnitude)
        self.mdp_1d = MDP(S, A, T, R, gamma)

    def reward(self, R):
        # TODO:
        pass


if __name__ == '__main__':
    length = 10
    default_prob = 0.8
    sns.set()

    # our baseline:
    test = Experiment_1D(length, default_prob)
    test.mdp_1d.solve('Baseline World')
    neg_idx = 8
    neg_magnitude = -10

    # MYOPIC EXPERIMENT RUNS:
    for gamma in np.arange(0.01, 1, 0.1):
        myopic = test.myopic(gamma = gamma)
        test.mdp_1d.solve('Myopic Agent: \u03B3={:.3f}'.format(gamma))
    
    # UNDERCONFIDENT + OVERCONFIDENT EXPERIMENT RUNS:
    for prob in np.arange(0.01, 1, 0.1):
        confident = test.confident(make_right_prob = prob)
        if prob < default_prob:
            test.mdp_1d.solve('Underconfident Agent: p={:.3f}'.format(prob))
        elif prob > default_prob:
            test.mdp_1d.solve('Overconfident Agent: p={:.3f}'.format(prob))

