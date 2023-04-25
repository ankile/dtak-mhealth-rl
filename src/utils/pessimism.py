import numpy as np
from tqdm import tqdm

from src.worlds.mdp2d import Experiment_2D


def apply_pessimism_to_transition(T, rewards_dict, scaling) -> np.ndarray:
    # Change the transition probabilities to be more pessimistic
    neg_rew_idx = [idx for idx in rewards_dict if rewards_dict[idx] < 0]

    T_new = T.copy()

    T[:, :, neg_rew_idx] *= scaling
    T /= T.sum(axis=2, keepdims=True)

    return T_new


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
