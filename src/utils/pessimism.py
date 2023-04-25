import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm

from src.utils.wall import wall
from src.worlds.mdp2d import Experiment_2D


def run_pessimism(
    experiment: Experiment_2D,
    scalers,
    gammas,
    name,
    transition_mode,
    pbar: bool | tqdm = True,
    postfix: bool = True,
):
    results = np.zeros((len(scalers), len(gammas)), dtype=int)
    probs = np.zeros(len(scalers), dtype=float)

    # Create the progress bar
    if isinstance(pbar, bool):
        pbar = tqdm(total=len(scalers) * len(gammas), disable=not pbar)

    # Run the experiment
    for i, scaling in enumerate(scalers):
        for j, gamma in enumerate(gammas):
            if postfix:
                pbar.set_postfix(
                    scaling=f"{scaling:<4.2f}",
                    gamma=f"{gamma:<4.2f}",
                )
            experiment.pessimistic(
                scaling=scaling, new_gamma=gamma, transition_mode=transition_mode
            )
            experiment.mdp.solve(
                setup_name=name,
                policy_name=f"Pessimistic scale={scaling:.2f} gamma={gamma:.2f}",
                save_heatmap=False,
            )

            results[i, j] = experiment.mdp.policy[0, 0]
            width = experiment.mdp.width
            probs[i] = experiment.mdp.T[1, width - 2, width - 1]
            pbar.update(1)

    return results, probs


def run_underconfident(
    experiment: Experiment_2D,
    probs: np.ndarray,
    gammas: np.ndarray,
    name: str,
    pbar: bool | tqdm = True,
    postfix: bool = True,
):
    results = np.zeros((len(probs), len(gammas)), dtype=int)

    # Create the progress bar
    if isinstance(pbar, bool):
        pbar = tqdm(total=len(probs) * len(gammas), disable=not pbar)

    # Run the experiment
    for i, prob in enumerate(probs):
        for j, gamma in enumerate(gammas):
            if postfix:
                pbar.set_postfix(
                    prob=f"{prob:<4.2f}",
                    gamma=f"{gamma:<4.2f}",
                )
            experiment.confident(action_success_prob=prob)
            experiment.mdp.solve(
                setup_name=name,
                policy_name=f"Underconfident prob={prob:.2f} gamma={gamma:.2f}",
                save_heatmap=False,
            )

            results[i, j] = experiment.mdp.policy[0, 0]
            pbar.update(1)

    return results, probs


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
