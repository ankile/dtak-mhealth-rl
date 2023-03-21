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

# Set up parameters to search over
scalers = np.arange(0.1, 5.5, 1)
gammas = np.arange(0.4, 1, 0.05 / 2)

parameters = {
    'reward_mag': [100, 200, 300, 400, 500],
    'neg_mag': [0, -5, -10, -15, -20],
    'latent_cost': [1, 0, -1, -2, -3],
    'prob': [0.5, 0.6, 0.7, 0.8, 0.9],
}
# parameters = {
#     'reward_mag': [100, 300, 500],
#     'neg_mag': [0, -10, -20],
#     'latent_cost': [1, -1, -3],
# }

rows = len(parameters)
cols = len(list(parameters.values())[0])

# Create the figure and axes to plot on
fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=(16, 10), sharex=True, sharey=True)

fig.subplots_adjust(top=0.9)

pbar = tqdm(total=rows * cols)

for i, (param_name, param_values) in enumerate(parameters.items()):
    pbar.set_description(f"Running {param_name}")
    ax_row = axs[i]
    for j, (value, ax) in enumerate(zip(param_values, ax_row)):
        pbar.update(1)
        pbar.set_postfix(param_name=param_name, value=value)
        # Set up the experiment
        config = {**default_config, param_name: value}
        test = setup_wall_world_experiment(**config, setup_name=setup_name)

        # Run the experiment
        results, probs = run_experiment(test, scalers, gammas, name=setup_name, pbar=False)

        # Create a heatmap of the resulting strategies on the second axis
        plot_strategy_heatmap(
            results,
            probs,
            gammas,
            ax=ax,
            title=f"{param_name}={value}",
            annot=False,
            legend=j == 0,
            ax_labels=False,
        )

        # Set axis labels for leftmost and bottom subplots
        if i == rows - 1:
            ax.set_xlabel("Gamma")
        if j == 0:
            ax.set_ylabel("Confidence")

# Save the figure
setup_config_string = ",".join(f"{k}={v}" for k, v in default_config.items())

fig.suptitle(f"Pessimist Invariance ({setup_config_string})")
plt.tight_layout()
plt.savefig(
    # f"images/{setup_name}/{datetime.now()}_strategy_reward{setup_config_string}.png"
    f"images/{setup_name}/{datetime.now()}.png"
)
plt.show()
