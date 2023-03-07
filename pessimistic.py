from worlds.mdp import *
import os
from utils.wall import wall

default_prob = 0.8
sns.set()

setup_name = "Pessimistic"
setup_name = setup_name.replace(" ", "_").lower()

if not os.path.exists(f"images/{setup_name}"):
    os.makedirs(f"images/{setup_name}")

height = 10
width = 5

wall_dict = wall(
    height,
    width,
    wall_width=3,
    wall_height=9,
    neg_mag=-30,
    reward_mag=100,
    latent_cost=0,
)
test = Experiment_2D(10, 5, rewards_dict=wall_dict)
test.mdp.solve(setup_name=setup_name, policy_name="Baseline World")
test.mdp.reset()

# Pessimistic
for scaling in np.arange(2, 10, 1):
    test.mdp.reset()
    test.pessimistic(scaling=scaling)
    test.mdp.solve(
        setup_name=setup_name, policy_name=f"Pessimistic Agent: scaling={scaling:.3f}"
    )
