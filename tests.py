from mdp import *
import os

default_prob = 0.8
sns.set()


def reward_radius(height, width, reward_distance, cost_radius, reward_mag, cost_mag):
    reward_dict = {}
    reward_idx = width * reward_distance + reward_distance
    for x in range(reward_distance - cost_radius, reward_distance + cost_radius + 1):
        for y in range(
            reward_distance - cost_radius, reward_distance + cost_radius + 1
        ):
            reward_dict[y * width + x] = cost_mag
    reward_dict[reward_idx] = reward_mag
    return reward_dict


def latent_cost(length, reward_mag, cost_mag):
    reward_dict = {}
    for i in range(length):
        reward_dict[i] = cost_mag
    reward_dict[length - 1] = reward_mag
    return reward_dict


def cost_wall(height, width, reward_x, reward_y, cost_thickness, reward_mag, cost_mag):
    reward_dict = {}
    return


def minefield(height, width, reward_x, reward_y, cost_density, reward_mag, cost_mag):
    reward_dict = {}
    return


def immediate_vs_jackpot(
    height, width, reward_idx1, reward_idx2, reward_mag1, reward_mag2
):
    reward_dict = {}
    return


# TODO: make reward states an absorbing state
# TODO: make sure value iteration gives negative values, and that you can't make an illegal action --> negative value means disengage
# TODO: put a latent cost onto making any action, so that underconfident agents are penalized for staying in the same spot (?), this is already somewhat incorporated by the myopicness

##### LATENT COSTS #####
setup_name = "Latent Cost"
setup_name = setup_name.replace(" ", "_").lower()

if not os.path.exists(f"images/{setup_name}"):
    os.makedirs(f"images/{setup_name}")

# test = Experiment_2D(10, 10, rewards_dict=reward_radius(10, 10, 5, 5, 100, -20))
# test = Experiment_2D(2, 2, rewards_dict={3: 10, 2:-2})
length = 10
test = Experiment_1D(
    length, default_prob, rewards_dict=latent_cost(length, reward_mag=20, cost_mag=-1)
)
test.mdp.solve(setup_name=setup_name, policy_name="Baseline World")
neg_idx = 8
neg_magnitude = -10
test.mdp.reset()

# REWARD AGENT RUNS:
reward_agent_R_val = 50
reward = test.reward(2, reward_agent_R_val, False)
test.mdp.solve(
    setup_name=setup_name,
    policy_name="Reward Agent: reward value={}".format(reward_agent_R_val),
)
test.mdp.reset()

# MYOPIC EXPERIMENT RUNS:
for gamma in np.arange(0.5, 0.99, 0.1):
    test.mdp.reset()
    myopic = test.myopic(gamma=gamma)
    test.mdp.solve(
        setup_name=setup_name, policy_name="Myopic Agent: \u03B3={:.3f}".format(gamma)
    )

# UNDERCONFIDENT + OVERCONFIDENT EXPERIMENT RUNS:
for prob in np.arange(0.05, 1, 0.1):
    test.mdp.reset()
    confident = test.confident(make_right_prob=prob)
    if prob < default_prob:
        test.mdp.solve(
            setup_name=setup_name,
            policy_name="Underconfident Agent: p={:.3f}".format(prob),
        )
    elif prob > default_prob:
        test.mdp.solve(
            setup_name=setup_name,
            policy_name="Overconfident Agent: p={:.3f}".format(prob),
        )


##### TWO REWARDS #####

setup_name = "Two Rewards"
setup_name = setup_name.replace(" ", "_").lower()

if not os.path.exists(f"images/{setup_name}"):
    os.makedirs(f"images/{setup_name}")

length = 21
rewards_dict = {}
# for i in range(10, 21):
#     rewards_dict[i] = -3
rewards_dict[0] = 10
rewards_dict[20] = 20
test = Experiment_1D(length, default_prob, rewards_dict=rewards_dict)
test.mdp.solve(setup_name=setup_name, policy_name="Baseline World")
test.mdp.reset()

# REWARD AGENT RUNS:
reward_agent_R_val = 50
reward = test.reward(2, reward_agent_R_val, False)
test.mdp.solve(
    setup_name=setup_name,
    policy_name="Reward Agent: reward value={}".format(reward_agent_R_val),
)
test.mdp.reset()

# MYOPIC EXPERIMENT RUNS:
for gamma in np.arange(0.5, 0.99, 0.1):
    test.mdp.reset()
    myopic = test.myopic(gamma=gamma)
    test.mdp.solve(
        setup_name=setup_name, policy_name="Myopic Agent: \u03B3={:.3f}".format(gamma)
    )

# UNDERCONFIDENT + OVERCONFIDENT EXPERIMENT RUNS:
for prob in np.arange(0.05, 1, 0.1):
    test.mdp.reset()
    confident = test.confident(make_right_prob=prob)
    if prob < default_prob:
        test.mdp.solve(
            setup_name=setup_name,
            policy_name="Underconfident Agent: p={:.3f}".format(prob),
        )
    elif prob > default_prob:
        test.mdp.solve(
            setup_name=setup_name,
            policy_name="Overconfident Agent: p={:.3f}".format(prob),
        )


##### TEST #####

setup_name = "Test"
setup_name = setup_name.replace(" ", "_").lower()

if not os.path.exists(f"images/{setup_name}"):
    os.makedirs(f"images/{setup_name}")

length = 5
rewards_dict = {}
# for i in range(10, 21):
#     rewards_dict[i] = -3
rewards_dict[4] = 10
test = Experiment_1D(length, rewards_dict=rewards_dict, make_right_prob=1)
test.mdp.solve(setup_name=setup_name, policy_name="Baseline World")
