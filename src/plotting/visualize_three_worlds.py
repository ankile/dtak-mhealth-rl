# Import necessary libraries for data science
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from src.plotting.config import FIG_FULL_SIZE, FIG_TITLE_FONT_SIZE

# Import functions to create the gridworlds
from src.utils.wall import make_wall_experiment
from src.utils.small_big import make_smallbig_experiment
from src.utils.cliff import make_cliff_experiment
from src.worlds.mdp2d import Experiment_2D


def get_reward(experiment: Experiment_2D) -> np.ndarray:
    """
    Get the reward matrix from the experiment.
    """
    height, width = experiment.mdp.height, experiment.mdp.width
    wall_dict = experiment.rewards_dict

    # Convert rewards dict to height by width matrix
    rewards = np.zeros((height, width), dtype=int)
    for key, value in wall_dict.items():
        rewards[key // width, key % width] = value

    return rewards


def make_three_world_rewards(
    height: int,
    width: int,
) -> tuple:
    """
    Create the three gridworlds for the paper.
    """
    wall, cliff, smallbig = make_three_worlds(height, width)

    return tuple(map(get_reward, (wall, cliff, smallbig)))


def make_three_worlds(height, width):
    wall = make_wall_experiment(
        height,
        width,
        reward_mag=100,
        neg_mag=-10,
        latent_cost=0,
    )

    cliff = make_cliff_experiment(
        height,
        width,
        reward_mag=100,
        small_r_mag=0,
    )

    smallbig = make_smallbig_experiment(
        height,
        width,
        big_reward=100,
        small_reward_frac=0.5,
    )

    return wall, cliff, smallbig


def relative_luminance(rgb: tuple) -> float:
    """
    Calculate the relative luminance of an RGB color.

    :param rgb: A tuple of (R, G, B) values (0-1 range)
    :return: The relative luminance value
    """

    r, g, b, _ = rgb
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def custom_formatter(value: float, threshold: float = 1e3) -> str:
    """
    Custom formatter for heatmap annotations.
    """
    if abs(value) < threshold:
        return f"{value:d}"
    else:
        return f"{value:.0e}"


def make_reward_heatmap(
    rewards: np.ndarray, ax: plt.Axes, start: tuple, goals: list
) -> None:
    """
    Make a heatmap of the rewards.
    """
    cmap = sns.color_palette("rocket", as_cmap=True)
    annot = np.vectorize(custom_formatter)(rewards)
    sns.heatmap(
        rewards, annot=annot, fmt="", ax=ax, cbar=False, cmap=cmap, vmin=-10, vmax=100
    )
    ax.set_xlabel("x")

    # Add starting point
    col_thresh = 0.2
    start_bg_color = cmap(rewards[start[0], start[1]])  # type: ignore
    start_text_color = (
        "black" if relative_luminance(start_bg_color) > col_thresh else "white"
    )
    offset = 0.1
    ax.text(
        start[1] + offset,
        start[0] + offset,
        "S",
        ha="left",
        va="top",
        color=start_text_color,
        fontsize=16,
        fontweight="bold",
    )

    # Add goal states
    for goal in goals:
        goal_bg_color = cmap(rewards[goal[0], goal[1]])  # type: ignore
        goal_text_color = (
            "black" if relative_luminance(goal_bg_color) > col_thresh else "white"
        )
        ax.text(
            goal[1] + offset,
            goal[0] + offset,
            "G",
            ha="left",
            va="top",
            color=goal_text_color,
            fontsize=16,
            fontweight="bold",
        )


if __name__ == "__main__":
    height, width = 5, 5
    world_names = ["(a) Wall", "(b) Cliff", "(c) SmallBig"]
    world_starts = [(0, 0), (height - 1, 0), (0, 0)]
    world_goals = [
        [(0, width - 1)],
        [(height - 1, width - 1)],
        [(height - 1, 0), (height - 1, width - 1)],
    ]

    rewards = make_three_world_rewards(
        height=height,
        width=width,
    )

    fig, axes = plt.subplots(1, 3, figsize=FIG_FULL_SIZE, sharey=True, sharex=True)

    for ax, reward, world_name, start, goals in zip(
        axes, rewards, world_names, world_starts, world_goals
    ):
        make_reward_heatmap(reward, ax, start, goals)
        ax.set_title(world_name, fontsize=FIG_TITLE_FONT_SIZE)

    fig.tight_layout()
    axes[0].set_ylabel("y")

    # Save figure
    fig.savefig("images/plots/three_worlds.png", bbox_inches="tight")

    plt.show()
