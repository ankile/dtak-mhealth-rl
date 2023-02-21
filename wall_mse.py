from mdp import *
import os
import pandas as pd

default_prob = 0.8
default_gamma = 0.9
sns.set()
setup_name = 'Wall_MSE'
setup_name = setup_name.replace(' ', '_').lower()

if not os.path.exists(f'images/{setup_name}'):
   os.makedirs(f'images/{setup_name}')

length = 10
starting_state = 7
reward_mag = 100
small_reward_mag = -1
big_cost = -20
latent_cost = -4
rewards_dict = {}

for i in range(length):
    rewards_dict[i] = latent_cost

for i in range(starting_state+1, length):
    rewards_dict[i] = big_cost

if small_reward_mag != -1:
    rewards_dict[0] = small_reward_mag
else:
    rewards_dict[0] = reward_mag
rewards_dict[length - 1] = reward_mag

mses = []
gammas = np.arange(0.8, 0.999, 0.015)
probs = np.arange(0.6, 0.999, 0.05)

base = Experiment_1D(length, make_right_prob=default_prob, rewards_dict=rewards_dict, gamma=default_gamma) # has 100% confidence
value_base, policy_base = base.mdp.solve(setup_name=setup_name, policy_name='Custom Agent: \u03B3={:.3f}, p={:.3f}'.format(default_gamma, default_prob), heatmap=False)

for gamma in gammas:
    mse_row = []
    for prob in probs:
        test = Experiment_1D(length, make_right_prob=prob, rewards_dict=rewards_dict, gamma=gamma)
        value_test, policy_test = test.mdp.solve(setup_name=setup_name, policy_name='Custom Agent: \u03B3={:.3f}, p={:.3f}'.format(gamma, prob), heatmap=False)
        test.mdp.reset()
        mse = ((value_test - value_base)**2).mean().round(1)
        mse_row.append(mse)
    mses.append(mse_row)

base.mdp.reset()

probs = list(probs.round(2))
gammas = list(gammas.round(3))
mses_df = pd.DataFrame(mses, columns=probs)
mses_df.index = gammas

hmap = sns.heatmap(mses_df, annot=True, fmt='', cbar=True, cbar_kws={'label': 'MSE'}, annot_kws={"size": 35 / np.sqrt(len(mses_df))})
title = f'Long Path: {starting_state}, Short Path: {length-starting_state-1}, Latent Cost: {latent_cost}, Big Cost: {big_cost}, Reward: {reward_mag}'
hmap.set(xlabel='Confidence', ylabel='Gamma', title=title)
hmap = hmap.figure
plt.savefig(f'images/{setup_name}/summary.png')
plt.clf()

# use newton's method to find where the MSE is close as possible to 0
# derivate some type of relationship to fit these points