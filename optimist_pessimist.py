import os
from datetime import datetime

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from utils.pessimism import (
    plot_strategy_heatmap,
    plot_world_reward,
    run_experiment,
    setup_wall_world_experiment,
)
from utils.wall import wall
from worlds.mdp import Experiment_2D

# Naming the setup
setup_name = "Optimist Pessimist"
setup_name = setup_name.replace(" ", "_").lower()

# Setting the parameters
default_prob = 0.8
default_gamma = 0.9
height = 5
width = 6
neg_mag = -10
reward_mag = 300
latent_cost = 0

# Set up parameters to search over
scalers = np.arange(0.1, 5.5, 1)
gammas = np.arange(0.01, 1, 0.05)

# Set up the experiment
test = setup_wall_world_experiment(
    setup_name,
    height,
    width,
    default_prob,
    default_gamma,
    neg_mag,
    reward_mag,
    latent_cost,
)

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))

plot_world_reward(test, setup_name, ax=ax1, save=False)

# Run the experiment
results, probs = run_experiment(test, scalers, gammas, name=setup_name)

# Create a heatmap of the resulting strategies
plot_strategy_heatmap(results, probs, gammas, ax=ax2)

setup_config_string = (
    f"(p={default_prob}, h={height}, w={width}, neg={neg_mag}, "
    f"reward={reward_mag}, latent={latent_cost}), g={[round(gammas[0], 2), round(gammas[-1], 2)]}, "
    f"s={[round(probs[0], 2), round(probs[-1], 2)]}"
)

ax2.set_title(f"Strategies for different gammas and pessimism")
fig.suptitle(f"Optimist Pessimist {setup_config_string}")
plt.savefig(
    f"images/{setup_name}/{datetime.now()}_strategy_reward{setup_config_string}.png"
)
plt.show()
