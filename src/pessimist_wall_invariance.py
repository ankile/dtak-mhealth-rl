import os
from datetime import datetime

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from utils.pessimism import (
    plot_strategy_heatmap,
    run_experiment,
    setup_wall_world_experiment,
)


# Naming the setup
setup_name = "Pessimist Wall World Invariance"
setup_name = setup_name.replace(" ", "_").lower()
output_dir = f"images/{setup_name}"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Setting the parameters
default_config = dict(
    prob=0.8,
    gamma=0.9,
    height=5,
    width=6,
    neg_mag=-10,
    latent_cost=0,
    reward_mag=300,
)

# Choose the transition dynamics
transition_mode = "simple"

# Set the number of subplots per row
cols = 7  # 7 or 5

# Set the number of scales and gammas to use
granularity = 10  # 10 or 5

# Set up parameters to search over
scalers = np.linspace(0.01, 8, granularity)
gammas = np.linspace(0.4, 1, granularity)

parameters = {
    "reward_mag": np.linspace(100, 500, cols),
    "neg_mag": np.linspace(-20, 0, cols),
    "latent_cost": [-3, -2, -1, 0, 1, 2, 3],
    "prob": np.linspace(0.5, 0.95, cols),
    "width": [3, 4, 5, 6, 7, 8, 9],
    "height": [2, 3, 4, 5, 6, 7, 8],
} if cols == 7 else {
    "reward_mag": np.linspace(100, 500, cols),
    "neg_mag": np.linspace(-20, 0, cols),
    "latent_cost": [-3, -2, -1, 0, 1, 2, 3][1:-1],
    "prob": np.linspace(0.5, 0.95, cols),
    "width": [3, 4, 5, 6, 7, 8, 9][1:-1],
    "height": [2, 3, 4, 5, 6, 7, 8][1:-1],
}

rows = len(parameters)

# Create the figure and axes to plot on
fig, axs = plt.subplots(
    nrows=rows, ncols=cols, figsize=(16, 10), sharex=True, sharey=True
)

fig.subplots_adjust(top=0.9)

pbar = tqdm(total=rows * cols)

for i, (param_name, param_values) in enumerate(parameters.items()):
    pbar.set_description(f"Running {param_name}")
    ax_row = axs[i]
    for j, (value, ax) in enumerate(zip(param_values, ax_row)):
        pbar.set_postfix_str(f"{param_name}={value}")
        # Set up the experiment
        config = {**default_config, param_name: value}
        test = setup_wall_world_experiment(**config, setup_name=setup_name)

        # Run the experiment and get the results
        results, probs = run_experiment(
            test,
            scalers,
            gammas,
            name=setup_name,
            transition_mode=transition_mode,
            pbar=False,
        )

        # Create a heatmap of the resulting strategies on the second axis
        plot_strategy_heatmap(
            results,
            probs,
            gammas,
            ax=ax,
            title=f"{param_name}={round(value, 2)}",
            annot=False,
            legend=j == 0,
            ax_labels=False,
        )

        # Set axis labels for leftmost and bottom subplots
        if i == rows - 1:
            ax.set_xlabel("Gamma")
        if j == 0:
            ax.set_ylabel("Confidence")
        
        pbar.update(1)

# Save the figure
setup_config_string = ",".join(f"{k}={v}" for k, v in default_config.items())

# Add transition mode to the config string
setup_config_string += f",t={transition_mode}"

fig.suptitle(f"Pessimist Invariance ({setup_config_string})")
plt.tight_layout()
plt.savefig(
    f"{output_dir}/{datetime.now()}.png"
)
plt.show()
