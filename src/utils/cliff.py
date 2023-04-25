import numpy as np

from utils.transition_matrix import make_absorbing


def cliff_reward(
    x,
    y,
    s,
    d,
    latent_reward=0,
    allow_disengage=False,
    disengage_reward=0,
) -> dict:
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
    :param allow_disengage: whether to allow the agent to disengage in the world

    returns a dictionary of rewards for each state in the gridworld.
    """
    # Create the reward dictionary
    reward_dict = {}
    for i in range(x * y):
        reward_dict[i] = latent_reward  # add latent cost

    # Define the world boundaries
    cliff_begin_x = 1
    cliff_end_x = x - 1
    cliff_y = y - (1 + int(allow_disengage))

    # Set the goal state
    reward_dict[x * cliff_y + x - 1] = d

    # Set the cliff states
    for i in range(cliff_begin_x, cliff_end_x):
        reward_dict[x * cliff_y + i] = s

    # Set the disengage states
    if allow_disengage:
        for i in range(0, x):
            reward_dict[x * (y - 1) + i] = disengage_reward

    return reward_dict


def cliff_transition(T, height, width, allow_disengage=False) -> np.ndarray:
    """
    Makes the cliff absorbing.
    """

    cliff_begin_x = 1
    cliff_end_x = width - 1
    # The cliff is one cell above the bottom row when we allow for disengagement
    cliff_y = height - (1 + int(allow_disengage))

    # Make the cliff absorbing
    T_new = T.copy()

    for i in range(cliff_begin_x, cliff_end_x):
        idx = width * cliff_y + i
        make_absorbing(T_new, idx)

    if allow_disengage:
        for i in range(0, width):
            idx = width * (height - 1) + i
            make_absorbing(T_new, idx)

    return T_new
