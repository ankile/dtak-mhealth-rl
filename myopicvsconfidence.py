from ..dtak.worlds.mdp import Experiment_1D
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

default_prob = 1
default_gamma = 0.9
sns.set()
setup_name = "Myopic vs Confidence"
setup_name = setup_name.replace(" ", "_").lower()

if not os.path.exists(f"images/{setup_name}"):
    os.makedirs(f"images/{setup_name}")

length = 10
starting_state = 7
reward_mag = 100
small_reward_mag = -1
big_cost = -20
latent_cost = -4
rewards_dict = {}

for i in range(length):
    rewards_dict[i] = latent_cost

for i in range(starting_state + 1, length):
    rewards_dict[i] = big_cost

if small_reward_mag != -1:
    rewards_dict[0] = small_reward_mag
else:
    rewards_dict[0] = reward_mag
rewards_dict[length - 1] = reward_mag


choices = []
gammas = np.arange(0.5, 0.995, 0.05)
probs = np.arange(0.05, 1.001, 0.05)

for gamma in gammas:
    choices_row = []
    for prob in probs:
        test = Experiment_1D(length, prob, rewards_dict=rewards_dict, gamma=gamma)
        value, policy = test.mdp.solve(
            setup_name=setup_name,
            policy_name="Custom Agent: \u03B3={:.3f}, p={:.3f}".format(gamma, prob),
            heatmap=False,
        )
        test.mdp.reset()
        # if value[starting_state] <= 0:
        #     choices_row.append(-1)
        # else:
        choices_row.append(int(policy[starting_state]))
    choices.append(choices_row)

probs = list(probs.round(2))
gammas = list(gammas.round(3))
choices_df = pd.DataFrame(choices, columns=probs)
choices_df.index = pd.Index(gammas)

hmap = sns.heatmap(
    choices_df, annot=True, fmt="", cbar=False, cbar_kws={"label": "Choice"}
)
title = f"Long Path: {starting_state}, Short Path: {length-starting_state-1}, Latent Cost: {latent_cost}, Big Cost: {big_cost}, Reward: {reward_mag}"
hmap.set(xlabel="Confidence", ylabel="Gamma", title=title)
hmap = hmap.figure
plt.savefig(f"images/{setup_name}/summary.png")
plt.clf()

# print(choices_df)


# adjust myopic with 1.0 confidence - use test.myopic -- go from 0.5 to 0.95 by 0.05

# adjust confidence with gamma = 0.8 (?) see what values would be best -- go from 0.1 to 1.0 by 0.05

# grid of values which would show the squared error between the model on the row and the model on the column
