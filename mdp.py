import numpy as np

class MDP():
    def __init__(self, S, A, T, R, gamma):
        self.S = S
        self.A = A
        self.T = T
        self.R = R
        self.gamma = gamma

        # sanity checks:
        assert T.shape == (len(self.A), len(self.S), len(self.S)) # action x state x state
        # assert T.sum(axis=0) == 1 # rows of T have to = 1
        assert R.shape == (len(self.S), len(self.A), len(self.S)) # state x action x next_state

        # TODO: include value iteration, value per state, Q matrix, optimal policy


## experiment test -- knob = length of 1d world:
class Experiment_1D():

    def __init__(self, length):
        # length is the
        S, A, T, R, gamma = self.make_MDP_params(length)
        self.mdp_1d = MDP(S, A, T, R, gamma)

    def make_MDP_params(self, length):
        S = np.arange(length)
        A = np.array((0, 1)) # 0 is left and 1 is right
        gamma = 0.8

        # always move right with 80% and stay with 20%, move left with 80% and stay with 20%
        T = np.zeros((2, length, length))

        # define T_action == 0 left
        T[0] = np.diag(np.array([1] + [0.2] * (length-2) + [1])) # 20% chance of staying in the same state unless in state 0 or state l-1

        # Do this more slickly with a np.roll()
        for i in range(1, length-1):
            T[0, i, i-1] = 0.8

        T[1] = np.diag(np.array([0.2] * (length-1) + [1])) # 20% chance of staying in the same state unless in state 0 or state l-1

        # Do this more slickly with a np.roll()
        for i in range(0, length-1):
            T[1, i, i+1] = 0.8

        R = np.zeros((length, 2, length)) # R is sparse
        R[length-2, 1, length-1] = 10

        return S, A, T, R, gamma

if __name__ == '__main__':
    # example of actually running an exp:
    # for length in range(100):
    #     test_env = Experiment_1d(length)

    # solve mdp, print optimal policy etc.

    # checking if implementation is correct
    test = Experiment_1D(5)
    print(test.mdp_1d.S)
    print(test.mdp_1d.A)
    print(test.mdp_1d.T)
    print(test.mdp_1d.R[0, 1, 1])
    print(test.mdp_1d.R[3, 1, 4])


    """
    don't hardcode the types of the users in the experiment class
    looking for 5 instantiations of the different classes/experiments
        - world: negative reward floating around -- kevin
        - other 4 correspond to the different users -- eman + me
            - don't worry too much about value iter:
                - still valuable to get this level of generalization down

    """
