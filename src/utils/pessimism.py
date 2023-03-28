import os
import numpy as np
from tqdm import tqdm
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import seaborn as sns
import matplotlib.pyplot as plt

from utils.wall import wall
from worlds.mdp2d import Experiment_2D


def run_experiment(experiment: Experiment_2D, scalers, gammas, name, pbar=True):
    results = np.zeros((len(scalers), len(gammas)), dtype=int)
    probs = np.zeros(len(scalers), dtype=float)

    # Create the progress bar
    pbar = tqdm(total=len(scalers) * len(gammas), disable=not pbar)

    # Run the experiment
    for i, scaling_pow in enumerate(scalers):
        scaling = 2**scaling_pow
        for j, gamma in enumerate(gammas):
            pbar.set_postfix(
                scaling=scaling,
                gamma=gamma,
            )
            experiment.mdp.reset()
            # experiment.pessimistic(scaling=scaling, new_gamma=gamma)
            experiment.pessimistic_new(scaling=scaling, new_gamma=gamma)
            experiment.mdp.solve(
                setup_name=name,
                policy_name=f"Pessimistic scale={scaling:.1f} gamma={gamma:.1f}",
                save_heatmap=False,
            )

            results[i, j] = experiment.mdp.policy[0, 0]
            width = experiment.mdp.width
            probs[i] = experiment.mdp.T[1, width - 2, width - 1]
            pbar.update(1)

    return results, probs


def plot_strategy_heatmap(
    results,
    probs,
    gammas,
    ax,
    title=None,
    legend=True,
    annot=True,
    ax_labels=True,
):
    ax = sns.heatmap(results, annot=annot, cmap="Blues", fmt="d", ax=ax, cbar=False)

    # Create a FixedLocator with a tick for every other value of gamma
    ax.xaxis.set_major_locator(ticker.FixedLocator(np.arange(0, len(gammas), 2)))
    ax.set_xticklabels(
        gammas[::2].round(5),
        rotation=90,
        size=8,
    )
    ax.set_yticklabels(probs.round(2), size=8, rotation=0)
    ax.invert_yaxis()
    if ax_labels:
        ax.set_xlabel("Gamma")
        ax.set_ylabel("Confidence")

    ax.set_title(title or "Optimal strategy (1: Right, 3: Down)")

    # Set legend to the right to explain the numbers 1 and 3 with same colors as the heatmap
    if legend:
        ax.legend(
            handles=[
                mpatches.Patch(color="white", label="1: Right"),
                mpatches.Patch(color="darkblue", label="3: Down"),
            ],
        )


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


def setup_wall_world_experiment(
    setup_name,
    height,
    width,
    prob,
    gamma,
    neg_mag,
    reward_mag,
    latent_cost,
) -> Experiment_2D:
    # Set up the experiment
    sns.set()
    if not os.path.exists(f"images/{setup_name}"):
        os.makedirs(f"images/{setup_name}")

    wall_dict = wall(
        height,
        width,
        wall_width=width - 2,
        wall_height=height - 1,
        neg_mag=neg_mag,
        reward_mag=reward_mag,
        latent_cost=latent_cost,
    )

    return Experiment_2D(
        height, width, rewards_dict=wall_dict, gamma=gamma, action_success_prob=prob
    )
