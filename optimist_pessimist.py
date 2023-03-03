from mdp import *
import os
from datetime import datetime
from wall import wall
from tqdm import tqdm

# import mpatches for the legend
import matplotlib.patches as mpatches

# import ticker for the x-axis
import matplotlib.ticker as ticker

# Naming the setup
setup_name = "Optimist Pessimist"
setup_name = setup_name.replace(" ", "_").lower()

# Setting the parameters
default_prob = 0.8
default_gamma = 0.9
height = 6
width = 7
neg_mag = -10
reward_mag = 100
latent_cost = -1

# Set up parameters to search over
scalers = np.arange(0.5, 5.5, 1)
gammas = np.arange(0.01, 1, 0.1)

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
test = Experiment_2D(height, width, rewards_dict=wall_dict, gamma=default_gamma)

# Convert rewards dict to 8 by 5 matrix
rewards = np.zeros((height, width), dtype=int)
for key, value in wall_dict.items():
    rewards[key // width, key % width] = value

# Create a heatmap of the rewards in the world
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))
ax1 = sns.heatmap(rewards, annot=True, fmt="d", ax=ax1, cbar=False)
ax1.set_title(f"World visualization")

results = np.zeros((len(scalers), len(gammas)), dtype=int)

# Create the progress bar
pbar = tqdm(total=len(scalers) * len(gammas))

# Run the experiment
for i, scaling_pow in enumerate(scalers):
    scaling = 2 ** scaling_pow
    for j, gamma in enumerate(gammas):
        pbar.set_postfix(
            scaling=scaling, gamma=gamma,
        )
        test.mdp.reset()
        test.pessimistic(scaling=scaling, new_gamma=gamma)
        test.mdp.solve(
            setup_name=setup_name,
            policy_name=f"Pessimistic scale={scaling:.1f} gamma={gamma:.1f}",
            save_heatmap=False,
        )

        results[i, j] = test.mdp.policy[0, 0]
        pbar.update(1)


ax2 = sns.heatmap(results, annot=True, cmap="Blues", fmt="d", ax=ax2, cbar=False)

# Create a FixedLocator with a tick per gamma
ax2.xaxis.set_major_locator(ticker.FixedLocator(np.arange(0, len(gammas), 1)))
ax2.set_xticklabels(gammas.round(5), rotation=90, size=8, )
ax2.set_yticklabels(scalers, size=8)
ax2.set_xlabel("Gamma")
ax2.set_ylabel("Scaling")

# Set legend to the right to explain the numbers 1 and 3 with same colors as the heatmap
ax2.legend(
    handles=[
        mpatches.Patch(color="white", label="1: Right"),
        mpatches.Patch(color="darkblue", label="3: Down"),
    ],
)

ax2.invert_yaxis()

setup_config_string = f"(p={default_prob}, h={height}, w={width}, neg={neg_mag}, " \
    f"reward={reward_mag}, latent={latent_cost}), g={[gammas[0], gammas[-1]]}, s={[scalers[0], scalers[-1]]}"

ax2.set_title(f"Strategies for different gammas and pessimism")
fig.suptitle(f"Optimist Pessimist {setup_config_string}")
plt.savefig(f"images/{setup_name}/{datetime.now()}_strategy_reward{setup_config_string}.png")
plt.show()
