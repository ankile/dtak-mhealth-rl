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

    path_count = 0
    while True:
        path_count += 1
        action = policy[cur_state // width][cur_state % width]
        if not is_valid_action(action, cur_state, width, height):
            print("Invalid path")
            return {}
        new_state = state_update(action, cur_state, width)
        optimal_path[cur_state] = new_state
        if new_state in end:
            break
        cur_state = new_state
        if path_count > 30:
            print("Path too long")
            return {}
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

    # Start with the four corners
    policies = [solve(corner 1), solve(corner 2), solve(corner 3), solve(corner 4)]

    while num_of_policies_tested < 30:
        # choose next policy to test given the results of the previous policies
        # add new policy to this list

    combined = combined_policies(policies)
    visualize policies    
