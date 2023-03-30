import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_world_reward(experiment, setup_name, ax, save=False):
    height, width = experiment.mdp.height, experiment.mdp.width
    wall_dict = experiment.rewards_dict

    # Convert rewards dict to height by width matrix
    rewards = np.zeros((height, width), dtype=int)
    for key, value in wall_dict.items():
        rewards[key // width, key % width] = value

    # Create a heatmap of the rewards in the world
    ax = sns.heatmap(rewards, annot=True, fmt="d", ax=ax, cbar=False)
    ax.set_title(f"World visualization")

    if save:
        plt.savefig(f"images/{setup_name}/world.png", bbox_inches="tight")
