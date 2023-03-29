import timeit

import numpy as np

from utils.pessimism import (
    run_experiment,
    setup_wall_world_experiment,
)

from worlds.mdp2d import MDP_2D

# Naming the setup
setup_name = "Wall Performance Test"
setup_name = setup_name.replace(" ", "_").lower()

# Setting the parameters
default_prob = 0.8
default_gamma = 0.95
height = 20
width = 20
neg_mag = -10
reward_mag = 300
latent_cost = 0

# Choose the transition dynamics
transition_mode = "simple"

# Set up parameters to search over
scalers = np.arange(0.1, 5.5, 1)
gammas = np.arange(0.01, 1, 0.05)

class MDP_2D_Theta(MDP_2D):
    def __init__(self, S, A, T, R, gamma):
        super().__init__(S, A, T, R, gamma)
        self.theta = np.nextafter(0, 1)

class MDP_2D_Orig(MDP_2D_Theta):

    def bellman_eq(self, state):
        vals = np.zeros(len(self.A))

        # TODO: Think about: IF ACTION IMPOSSIBLE, ASSIGN np.NINF value -- do this by if the sum of the self.T[action][state] = 0 then do this
        for action in self.A:
            to_sum = []
            for p in range(len(self.T[action][state])):
                to_sum.append(
                    self.T[action][state][p]
                    * (
                        self.R[state][action][p]
                        + (self.gamma * self.V[p // self.width][p % self.width])
                    )
                )

            vals[action] = sum(to_sum)

        def check_action(state, width, height):
            if state % width == 0:  # left-border
                vals[0] = np.NINF
            if state % width == width - 1:  # right-border
                vals[1] = np.NINF
            if state < width:  # top
                vals[2] = np.NINF
            if state >= width * (height - 1):  # bottom
                vals[3] = np.NINF

        check_action(state, self.width, self.height)

        return vals

    def value_iteration(self):
        while True:
            difference = 0
            for row in self.S:
                for state in row:
                    old_V = self.V[state // self.width][state % self.width]
                    v = self.bellman_eq(state)

                    self.policy[state // self.width][state % self.width] = np.argmax(v)
                    self.V[state // self.width][state % self.width] = np.max(v)

                    difference = max(
                        difference,
                        np.abs(old_V - self.V[state // self.width][state % self.width]),
                    )

            if difference < self.theta:
                break


# Create a function that wraps the run_experiment function to enable timing
def run_experiment_with_timing(test, *args, **kwargs):
    test.mdp.reset()
    test.mdp.solve(
        setup_name="Runtime analysis=",
        policy_name="Baseline",
        save_heatmap=False,
    )

n_exp = 100

# Run the experiement with timing enabled by a decorator from the timeit module 3 times
print("Timing new code...", end="")

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

result_numpy_theta = timeit.timeit(lambda: run_experiment_with_timing(test), number=n_exp)
print(f"Done! (t={result_numpy_theta:.2f}s)")

# Set the value iteration theta to the prev value
print("Running new code with original theta...", end="")

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
test.mdp = MDP_2D_Theta(test.mdp.S, test.mdp.A, test.mdp.T, test.mdp.R, test.mdp.gamma)
result_numpy = timeit.timeit(lambda: run_experiment_with_timing(test), number=n_exp)
print(f"Done! (t={result_numpy:.2f}s)")

# Set the value iteration theta to the prev value and functions to the prev functions
print("Running original code...", end="")

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
test.mdp = MDP_2D_Orig(test.mdp.S, test.mdp.A, test.mdp.T, test.mdp.R, test.mdp.gamma)
result_orig = timeit.timeit(lambda: run_experiment_with_timing(test), number=n_exp)
print(f"Done! (t={result_orig:.2f}s)")

# Print the relative speedups
print(f"Relative speedup Numpy and Theta: {result_orig/result_numpy_theta:.2f}, Just Numpy: {result_orig/result_numpy:.2f}")