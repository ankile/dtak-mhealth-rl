import os
from matplotlib import pyplot as plt
import numpy as np
from utils.transition_matrix import transition_matrix_is_valid
from worlds.mdp2d import Experiment_2D

# Import the cliff world
from utils.cliff import cliff_reward, cliff_transition
from visualization.worldviz import plot_world_reward


def cliff_experiment(
    setup_name: str,
    height,
    width,
    prob,
    gamma,
    reward_mag,
    neg_mag=-np.inf,
    latent_cost=0,
) -> Experiment_2D:
    if not os.path.exists(savepath := f"local_images/{setup_name}"):
        os.makedirs(savepath)

    cliff_dict = cliff_reward(
        x=width,
        y=height,
        s=neg_mag,
        d=reward_mag,
        latent_cost=latent_cost,
    )

    experiment = Experiment_2D(
        height,
        width,
        rewards_dict=cliff_dict,
        gamma=gamma,
        action_success_prob=prob,
        transition_mode="full",
    )

    assert transition_matrix_is_valid(
        experiment.mdp.T
    ), "whoop whoop Transition matrix is invalid"

    T_new = cliff_transition(
        T=experiment.mdp.T,
        x=width,
        y=height,
    )

    experiment.mdp.T = T_new

    return experiment


if __name__ == "__main__":
    default_prob = 0.7
    gamma = 0.99
    height = 4
    width = 12
    reward_mag = 1e2
    neg_mag = -1e4
    latent_cost = 0

    experiment = cliff_experiment(
        setup_name="Cliff",
        height=height,
        width=width,
        prob=default_prob,
        gamma=gamma,
        reward_mag=reward_mag,
        neg_mag=neg_mag,
        latent_cost=latent_cost,
    )

    fig, ax = plt.subplots()

    # plot_world_reward(experiment, setup_name="Cliff", ax=ax, show=True)

    experiment.mdp.solve(
        setup_name="Cliff",
        policy_name="Baseline World",
        save_heatmap=True,
        base_dir="local_images",
    )
