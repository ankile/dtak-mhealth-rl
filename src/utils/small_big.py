import os

import numpy as np
import seaborn as sns
from src.utils.enums import TransitionMode

from src.worlds.mdp2d import Experiment_2D


def smallbig_reward(
    height: int,
    width: int,
    big_reward: float,
    small_reward_frac: float,
) -> dict:
    """
    Places the small and big rewards in the bottom-left and bottom-right corners, respectively.

    returns a dictionary of rewards for each state in the gridworld.
    """

    reward_dict = {}

    small_reward = big_reward * small_reward_frac

    # Put small reward in lower left corner
    small_reward_state = (height - 1) * width
    reward_dict[small_reward_state] = small_reward

    # Put big reward in lower right corner
    big_reward_state = height * width - 1
    reward_dict[big_reward_state] = big_reward

    return reward_dict


def make_smallbig_experiment(
    height: int,
    width: int,
    big_reward: float,
    small_reward_frac: float,
) -> Experiment_2D:
    wall_dict = smallbig_reward(
        height,
        width,
        big_reward,
        small_reward_frac,
    )

    experiment = Experiment_2D(
        height,
        width,
        rewards_dict=wall_dict,
        transition_mode=TransitionMode.SIMPLE,
    )

    return experiment
