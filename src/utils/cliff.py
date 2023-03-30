def cliff(x, y, w, h, s, d, T, latent_cost=0):
    """
    Creates a cliff on the bottom of the gridworld.
    The start state is the bottom left corner.
    The goal state is the bottom right corner.

    The agent needs to walk around the cliff to reach the goal.

    :param x: width of the gridworld
    :param y: height of the gridworld
    :param w: width of the cliff
    :param h: height of the cliff
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
    cliff_begin_x = x - 1 - w
    cliff_end_x = x - 1
    cliff_begin_y = y - 1 - h
    cliff_end_y = y - 1
    for i in range(cliff_begin_x, cliff_end_x):
        for j in range(cliff_begin_y, cliff_end_y):
            reward_dict[x * j + i] = s

    # Make the cliff absorbing
    T_new = T.copy()
    for i in range(cliff_begin_x, cliff_end_x):
        for j in range(cliff_begin_y, cliff_end_y):
            T_new[:, x * j + i, x * j + i] = 1

    return reward_dict
