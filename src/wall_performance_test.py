import timeit

import numpy as np

from utils.pessimism import (
    run_experiment,
    setup_wall_world_experiment,
)


# Naming the setup
setup_name = "Wall Performance Test"
setup_name = setup_name.replace(" ", "_").lower()

# Setting the parameters
default_prob = 0.8
default_gamma = 0.9
height = 5
width = 6
neg_mag = -10
reward_mag = 300
latent_cost = 0

# Choose the transition dynamics
transition_mode = "full"

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

# The old value iteration theta and functions
old_theta = np.nextafter(0, 1)

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
def run_experiment_with_timing(*args, **kwargs):
    run_experiment(
        test, scalers, gammas, transition_mode=transition_mode, name=setup_name
    )



# Run the experiement with timing enabled by a decorator from the timeit module 3 times
result_numpy_theta = timeit.timeit(run_experiment_with_timing, number=3)

test.mdp.theta = old_theta
result_numpy = timeit.timeit(run_experiment_with_timing, number=3)

test.mdp.bellman_eq = bellman_eq
test.mdp.value_iteration = value_iteration
result_orig = timeit.timeit(run_experiment_with_timing, number=3)

# Print the results
print(f"Runtimes: Original = {result_orig}, Numpy optimization = {result_numpy}, Increased Theta = {result_numpy_theta}")