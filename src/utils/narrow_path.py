import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from src.utils.enums import TransitionMode

from src.visualization.worldviz import plot_world_reward
from src.worlds.mdp2d import Experiment_2D

HEIGHT, WIDTH = 3, 7


def c2s(i, j) -> int:
    """
    Coordinate to State.
    Converts a 2D coordinate to a 1D state.
    """
    return i * WIDTH + j


BIG = c2s(1, 6)
SMALL = c2s(1, 0)
POISON = [c2s(0, i) for i in range(4, 7)] + [c2s(2, i) for i in range(4, 7)]
START = c2s(1, 3)

"""
Schema for the N by M Narrow Path World:

    0   1   2   3   4   5   6 
0   .   .   .   .   P   P   P 
1   S   .   .   O   .   .   B 
2   .   .   .   .   P   P   P 


O   = Start
B   = Big Reward
S   = Small Reward
P   = Poison Ivy
.   = Path
"""


def get_goal_states(*args) -> set:
    """
    Returns a set of goal states for the narrow_path world.
    """

    return set([BIG, SMALL])


def narrow_path_reward(
    big_reward,
    small_reward,
    poison_reward,
    **kwargs,
) -> dict:
    """
    Returns a dictionary of rewards for the narrow_path world.

    Parameters
    ----------
    vegetarian_reward: float
        Reward for the vegetarian option.
    donut_reward: float
        Reward for the donut option.
    noodle_reward: float
        Reward for the noodle option.

    Returns
    -------
    dict
        Dictionary of rewards.

    """

    return {
        BIG: big_reward,
        SMALL: small_reward,
        **{state: poison_reward for state in POISON},
    }


def make_narrow_path_experiment(
    prob,
    gamma,
    big_reward,
    small_reward,
    poison_reward,
    **kwargs,
) -> Experiment_2D:
    narrow_path_dict = narrow_path_reward(
        big_reward=big_reward,
        small_reward=small_reward,
        poison_reward=poison_reward,
    )

    experiment = Experiment_2D(
        height=HEIGHT,
        width=WIDTH,
        action_success_prob=prob,
        gamma=gamma,
        rewards_dict=narrow_path_dict,
        transition_mode=TransitionMode.FULL,
    )

    return experiment


if __name__ == "__main__":
    params = {
        "prob": 0.4,
        "gamma": 0.99,
        "big_reward": 100,
        "small_reward": 90,
        "poison_reward": -100,
    }

    experiment = make_narrow_path_experiment(**params)

    # Make plot with 5 columns where the first column is the parameters
    # and the two plots span two columns each

    # create figure with 5 columns
    fig = plt.figure(figsize=(12, 4))
    gs = GridSpec(1, 5, figure=fig)

    # add text to first column
    ax1 = fig.add_subplot(gs[0, 0])  # type: ignore
    ax1.axis("off")

    # add subplots to remaining 4 columns
    ax2 = fig.add_subplot(gs[0, 1:3])  # type: ignore
    ax3 = fig.add_subplot(gs[0, 3:5])  # type: ignore

    # Adjust layout and spacing (make room for titles)
    fig.tight_layout()
    plt.subplots_adjust(top=0.9)

    # Add the parameters to the first subplot
    ax1.text(
        0.05,
        0.95,
        "\n".join([f"{k}: {v}" for k, v in params.items()]),
        horizontalalignment="left",
        verticalalignment="top",
        transform=ax1.transAxes,
    )

    plot_world_reward(experiment, setup_name="narrow_path", ax=ax2, show=False)

    experiment.mdp.solve(
        save_heatmap=False,
        show_heatmap=False,
        heatmap_ax=ax3,
        base_dir="local_images",
        label_precision=1,
    )

    # set titles for subplots
    ax1.set_title("Parameters", fontsize=16)
    ax2.set_title("World Rewards", fontsize=16)
    ax3.set_title("Optimal Policy for Parameters", fontsize=16)

    # Show the plot
    plt.show()
