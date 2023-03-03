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
height = 5
width = 5
neg_mag = -20
reward_mag = 100
latent_cost = 0

# Set up parameters to search over
scalers = np.arange(0.1, 10.5, 0.1)

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

# Create storage
data = np.zeros((len(scalers), 3), dtype=float)

# Run the experiment
for i, scaling in enumerate(scalers):
    test.mdp.reset()
    test.pessimistic(scaling=scaling)

    data[i, 0] = scaling

    trans = test.mdp.T
    data[i, 1:] = trans[1][3].reshape(height, width)[0, -2:]

# Create figure with transparent facecolor
fig, ax = plt.subplots(facecolor="white", figsize=(10, 6))
ax.grid(False)

ax.plot(data[:, 0], data[:, 1], color="black")
ax.fill_between(data[:, 0], data[:, 1], 1, color="blue", alpha=0.5)
ax.fill_between(data[:, 0], 0, data[:, 1], color="red", alpha=0.5)
ax.axhline(0.5, color="black", linestyle="--", label="50% probability")
ax.axhline(0.2, color="black", linestyle=":", label="Default probability")
ax.axhline(0.8, color="black", linestyle="-.", label="80% probability")


# Set limits
ax.set_xlim(scalers[0], scalers[-1])
ax.set_ylim(0, 1)

# Create a legend outside the figure but still visible
ax.legend()
plt.subplots_adjust(bottom=0.2)
ax.set_xlabel("Pessimism scaling factor")
ax.set_ylabel("Perceived probability of action success")
ax.set_title("Pessimistic agent's perception of action success")

# Make the x-axis and y-axis ticks more readable
ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))

# Add 'Success' in the top left corner
ax.text(
    0.05,
    0.95,
    "Perceived success rate\n(intended action happens)",
    horizontalalignment="left",
    verticalalignment="top",
    transform=ax.transAxes,
    color="black",
    fontsize=14,
)

# Add 'Failure' in the bottom right corner
ax.text(
    0.95,
    0.05,
    "Perceived failure rate\n(nothing happens)",
    horizontalalignment="right",
    verticalalignment="bottom",
    transform=ax.transAxes,
    color="black",
    fontsize=14,
)


# Show and save the figure
plt.show()
fig.savefig(f"images/{setup_name}/pessimist_transition.png")

