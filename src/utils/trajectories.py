import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from src.utils.enums import TransitionMode

from src.utils.transition_matrix import make_absorbing
from src.visualization.worldviz import plot_world_reward
from src.worlds.mdp2d import Experiment_2D

from src.utils.wall import wall_reward, make_wall_experiment
from src.utils.small_big import smallbig_reward, make_smallbig_experiment
from src.utils.cliff import cliff_reward, make_cliff_experiment

import sys

def is_valid_action(action, cur_state, width, height):
    '''
    Check if the action is valid for the current state
    '''
    if action == 0:
        if cur_state % width == 0:
            return False
    elif action == 1:
        if cur_state % width == width - 1:
            return False
    elif action == 2:
        if cur_state // width == 0:
            return False
    elif action == 3:
        if cur_state // width == height - 1:
            return False
    return True


def state_update(action, cur_state, width):
    '''
    Update the state based on the action
    '''
    if action == 0:
        return cur_state - 1
    elif action == 1:
        return cur_state + 1
    elif action == 2:
        return cur_state - width
    elif action == 3:
        return cur_state + width


def optimal_policy_callback(policy, start, end, height, width):
    '''
    Given 2D policy for every square, return an adjacency list containing only the optimal path.
    For actions: 0 is left, 1 is right, 2 is up, 3 is down
    End states is a list
    '''
    optimal_path = {}
    cur_state = start

    while True:
        action = policy[cur_state // width][cur_state % width]
        if not is_valid_action(action, cur_state, width, height):
            print("Invalid path")
            break
        new_state = state_update(action, cur_state, width)
        optimal_path[cur_state] = new_state
        if new_state in end:
            break
        cur_state = new_state
    print("Optimal path:", optimal_path)
    return optimal_path


def combined_policies(optimal_list):
    '''
    Takes in a list of optimal paths as adjacency lists and returns a combined adjacency list
    '''
    combined = {}
    for path in optimal_list:
        for key in path:
            if key not in combined:
                combined[key] = [path[key]]
            else:
                # append path[key] to combined[key] if it is not already present
                if path[key] not in combined[key]:
                    combined[key].append(path[key])
    return combined

# def combined_viz(combined, height, width):
#     '''
#     Visualizes the combined adjacency list
#     '''

#     fig, ax = plt.subplots()
#     ax.set_xlim(0, width)
#     ax.set_ylim(0, height)
#     for key in combined:
#         for value in combined[key]:
#             x1, y1 = key % width, height - 1 - key // width
#             x2, y2 = value % width, height - 1 - value // width
#             ax.arrow(x1, y1, x2 - x1, y2 - y1, head_width=0.1, head_length=0.1, fc='r', ec='r')
#     plt.show()

# visualize combined as a connected graph
def combined_viz(combined, height, width):
    '''
    Visualizes the combined adjacency list
    '''

    fig, ax = plt.subplots()
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    for key in combined:
        for value in combined[key]:
            x1, y1 = key % width, height - 1 - key // width
            x2, y2 = value % width, height - 1 - value // width
            ax.arrow(x1, y1, x2 - x1, y2 - y1, head_width=0.1, head_length=0.1, fc='r', ec='r')
    plt.show()
def unique_dicts(list_of_dicts):
    # Create a set for storing unique tuples
    unique_tuples = set()

    for d in list_of_dicts:
        # Create a sorted tuple of (key, value) pairs
        sorted_tuple = tuple(sorted(d.items()))
        unique_tuples.add(sorted_tuple)

    # If you need a list of dictionaries instead of tuples, convert them back
    unique_dicts_list = [dict(t) for t in unique_tuples]
    return unique_dicts_list

def unroll_policy_dict(policy):
    seq_states = []
    cur_state = 0
    while cur_state in policy:
        seq_states.append(cur_state)
        cur_state = policy[cur_state]
    return seq_states

if __name__ == "__main__":
    # Input validation
    if len(sys.argv) != 2:
        raise ValueError("Must specify type of world")
    
    world_type = sys.argv[1]
    if world_type not in ["wall", "smallbig", "cliff"]:
        raise ValueError("Invalid world type")
    
    # Create the worlds
    gammas = np.linspace(0.4, 0.99, 20)
    probs = np.linspace(0.4, 0.99, 20)

    policy_list = []

    if world_type == "wall":
        height = 5
        width = 7
        for gamma in gammas:
            for prob in probs:
                print(gamma, prob)
                experiment = make_wall_experiment(
                    prob=prob,
                    gamma=gamma,
                    height=height,
                    width=width,
                    reward_mag=500,
                    small_r_mag=100,
                    neg_mag=-50,
                    latent_reward=0,
                )

                policy = experiment.mdp.policy_callback()
                print(policy)
                optimal_path = optimal_policy_callback(policy, 0, [width-1, height * width - 3 - 2*width], 5, 7)
                policy_list.append(optimal_path)
        
    elif world_type == "smallbig":
        height = 7
        width = 7
        for gamma in gammas:
            for prob in probs:
                experiment = make_smallbig_experiment(
                    prob=prob,
                    gamma=gamma,
                    height=height,
                    width=width,
                    big_reward=300,
                    small_reward=100,
                    latent_reward=0,
                )

                policy = experiment.mdp.policy_callback()
                optimal_path = optimal_policy_callback(policy, 0, [width*(height-1), height*width-1], 7, 7)
                policy_list.append(optimal_path)

    elif world_type == "cliff":
        height = 5
        width = 9
        for gamma in gammas:
            for prob in probs:
                experiment = make_cliff_experiment(
                    prob=prob,
                    gamma=gamma,
                    height=height,
                    width=width,
                    reward_mag=1e2,
                    neg_mag=-1e8,
                    latent_reward=0,
                )

                policy = experiment.mdp.policy_callback()
                optimal_path = optimal_policy_callback(policy, 0, [height*width-1], 5, 9)
                policy_list.append(optimal_path)
    # print set of unique optimal paths
    unique_policies = unique_dicts(policy_list)
    print("OPTIMAL PATHS:")
    for path in unique_policies:
        print(unroll_policy_dict(path))

    print("FINAL TREE:", combined_policies(policy_list))
    combined_viz(combined_policies(policy_list), height, width)
