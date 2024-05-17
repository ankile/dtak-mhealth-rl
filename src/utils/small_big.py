import os

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from src.utils.enums import TransitionMode

from src.utils.transition_matrix import make_absorbing
from src.visualization.worldviz import plot_world_reward
from src.worlds.mdp2d import Experiment_2D



def smallbig_reward(
    height,
    width,
    big_reward,
    small_reward,
    latent_cost,
) -> dict:
    """
    Places the small and big rewards in the bottom-left and bottom-right corners, respectively.

    returns a dictionary of rewards for each state in the gridworld.
    """

    reward_dict = {}
    for i in range(height * width):
        reward_dict[i] = latent_cost  # add latent cost

    # Put small reward in lower left corner
    small_reward_state = (height - 1) * width
    # small_reward_state = 18 * width + 11
    reward_dict[small_reward_state] = small_reward

    # Put big reward in lower right corner
    big_reward_state = height * width - 1
    # big_reward_state = 18 * width + 18
    reward_dict[big_reward_state] = big_reward

    return reward_dict


def make_smallbig_experiment(
    prob,
    gamma,
    height,
    width,
    big_reward,
    small_reward,
    latent_reward=0,
) -> Experiment_2D:
    wall_dict = smallbig_reward(
        height,
        width,
        big_reward,
        small_reward,
        latent_reward,
    )

    experiment = Experiment_2D(
        height,
        width,
        action_success_prob=prob,
        rewards_dict=wall_dict,
        transition_mode=TransitionMode.FULL,
        gamma=gamma,
    )

    return experiment



if __name__ == "__main__":
    params = {
        "prob": 0.4,
        "gamma": 0.99,
        "height": 7,
        "width": 7,
        "big_reward": 300,
        "small_reward": 100,
        "latent_reward": 0,
    }

    experiment = make_smallbig_experiment(**params)

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

    # Create a mask for the bottom row if the user is allowed to disengage
    mask = None

    plot_world_reward(experiment, setup_name="Smallbig", ax=ax2, show=False, mask=mask)

    experiment.mdp.solve(
        save_heatmap=False,
        show_heatmap=False,
        heatmap_ax=ax3,
        heatmap_mask=mask,
        base_dir="local_images",
        label_precision=2,
    )

    # set titles for subplots
    ax1.set_title("Parameters", fontsize=16)
    ax2.set_title("World Rewards", fontsize=16)
    ax3.set_title("Optimal Policy for Parameters", fontsize=16)

    # Show the plot
    plt.show()