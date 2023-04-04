import numpy as np

from utils.transition_matrix import make_absorbing


def cliff_reward(x, y, s, d, latent_cost=0) -> dict:
    """
    Creates a cliff on the bottom of the gridworld.
    The start state is the bottom left corner.
    The goal state is the bottom right corner.
    Every cell in the bottom row except the goal and start state is a cliff.

    The agent needs to walk around the cliff to reach the goal.

    :param x: width of the gridworld
    :param y: height of the gridworld
    :param d: reward for reaching the goal
    :param c: latent cost
    :param T: the transition matrix
    :param s: cost of falling off the cliff

    returns a dictionary of rewards for each state in the gridworld.
    """
    # Create the reward dictionary
    reward_dict = {}
    for i in range(x * y):
        reward_dict[i] = latent_cost  # add latent cost

    # Set the goal state
    reward_dict[x * (y - 1) + x - 1] = d

    # Set the cliff states
    cliff_begin_x = 1
    cliff_end_x = x - 1
    cliff_y = y - 1
    for i in range(cliff_begin_x, cliff_end_x):
        reward_dict[x * cliff_y + i] = s

    return reward_dict


def cliff_transition(T, x, y) -> np.ndarray:
    """
    Makes the cliff absorbing.
    The cliff states are the only absorbing states.
    """

    cliff_begin_x = 1
    cliff_end_x = x - 1
    cliff_y = y - 1

    # Make the cliff absorbing
    T_new = T.copy()

    for i in range(cliff_begin_x, cliff_end_x):
        idx = x * cliff_y + i
        make_absorbing(T_new, idx)

    return T_new
