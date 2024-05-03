import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from src.utils.enums import TransitionMode

from src.utils.transition_matrix import make_absorbing
from src.visualization.worldviz import plot_world_reward
from src.worlds.mdp2d import Experiment_2D


def wall_reward(
    height: int,
    width: int,
    wall_width: int,
    wall_height: int,
    neg_mag: float,
    reward_mag: float,
    small_r_mag: float,
    latent_cost: float = 0,
) -> dict:
    """
    Creates a wall in the middle of the gridworld.

    returns a dictionary of rewards for each state in the gridworld.
    """

    reward_dict = {}
    for i in range(height * width):
        reward_dict[i] = latent_cost  # add latent cost
    wall_end_x = width - 2
    wall_begin_x = wall_end_x - wall_width +2
    wall_end_y = wall_height-1
    wall_begin_y = 0
    for i in range(wall_begin_x, wall_end_x):
        for j in range(wall_begin_y, wall_end_y):
            reward_dict[width * j + i] = neg_mag
    reward_dict[width - 1] = reward_mag

    # COMPOSITION: Adding a small reward along the path
    reward_dict[height * width - 3 - 2*width] = small_r_mag
    # reward_dict[width*(height-1)] = small_r_mag

    return reward_dict


def make_wall_experiment(
    prob,
    gamma,
    height,
    width,
    neg_mag,
    reward_mag,
    small_r_mag,
    latent_reward=0,
) -> Experiment_2D:
    wall_dict = wall_reward(
        height,
        width,
        wall_width=width - 2,
        wall_height=height - 1,
        neg_mag=neg_mag,
        reward_mag=reward_mag,
        small_r_mag=small_r_mag,
        latent_cost=latent_reward,
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
        "gamma": 0.993,
        "height": 5,
        "width": 7,
        "reward_mag": 500,
        "small_r_mag": 100,  # small_mag of 0 = normal cliff world
        "neg_mag": -50,
        "latent_reward": 0,
    }

    experiment = make_wall_experiment(**params)

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
    plot_world_reward(experiment, setup_name="Wall-Smallbig", ax=ax2, show=False, mask=mask)

    experiment.mdp.solve(
        save_heatmap=False,
        show_heatmap=False,
        heatmap_ax=ax3,
        heatmap_mask=mask,
        base_dir="local_images",
        label_precision=1,
    )

    # set titles for subplots
    ax1.set_title("Parameters", fontsize=16)
    ax2.set_title("World Rewards", fontsize=16)
    ax3.set_title("Optimal Policy for Parameters", fontsize=16)

    # Show the plot
    plt.show()