import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.worlds.mdp2d import Experiment_2D


def plot_world_reward(
    experiment: Experiment_2D, setup_name, ax, save=False, show=False, mask=None
):
    height, width = experiment.mdp.height, experiment.mdp.width
    wall_dict = experiment.rewards_dict

    # Convert rewards dict to height by width matrix
    rewards = np.zeros((height, width), dtype=int)
    for key, value in wall_dict.items():
        rewards[key // width, key % width] = value

    # Set masked states to be the letter 'D' for disengaged
    # and rest of the states the integer value of the reward as string
    annot = np.zeros((height, width), dtype=np.object0)
    for i in range(height):
        for j in range(width):
            annot[i, j] = str(rewards[i, j])

    if mask is not None:
        annot = np.where(mask, "D.E.", annot)

    # Create a heatmap of the rewards in the world
    ax = sns.heatmap(rewards, annot=annot, ax=ax, cbar=False, fmt="s")
    ax.set_title(f"World visualization")

    if save:
        plt.savefig(f"images/{setup_name}/world.png", bbox_inches="tight")

    if show:
        plt.show()
