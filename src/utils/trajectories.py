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
    # Input validation
    if len(sys.argv) != 2:
        raise ValueError("Must specify type of world")
    
    world_type = sys.argv[1]
    if world_type not in ["wall", "smallbig", "cliff"]:
        raise ValueError("Invalid world type")
    
    # Create the worlds
    # gammas = np.linspace(0.01, 0.999, 30)
    # probs = np.linspace(0.01, 0.999, 30)
    # gammas = [0.4,0.99]
    # probs = [0.4, 0.99]
    
    #bottom = [[0.4,0.4], [0.5,0.4], [0.6,0.4],[0.7,0.4], [0.8,0.4],[0.9,0.4],[0.99,0.4]]
    #top = [[0.4,0.99], [0.5,0.99], [0.6,0.99],[0.7,0.99], [0.8,0.99],[0.9,0.99],[0.99,0.99]]
    #left = [[0.4,0.4], [0.4,0.5], [0.4,0.6],[0.4,0.7], [0.4,0.8],[0.4,0.9],[0.4,0.99]]
    #right = [[0.99,0.4], [0.99,0.5], [0.99,0.6],[0.99,0.7], [0.99,0.8],[0.99,0.9],[0.99,0.99]]

    bottom = [[0.4,0.4],[0.6,0.4],[0.8,0.4],[0.99,0.4]]
    top = [[0.4,0.99],[0.6,0.99],[0.8,0.99],[0.99,0.99]]
    left = [[0.4,0.4],[0.4,0.6],[0.4,0.8],[0.4,0.99]]
    right = [[0.99,0.4],[0.99,0.6],[0.99,0.8], [0.99,0.99]]

    sides = [bottom, top, left, right]

    policy_list = []
    new_path_dict = {}

    if world_type == "wall":
        height = 5
        width = 7
        for side in sides:
            for elem in side:
                gamma = elem[0]
                prob = elem[1]
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
                optimal_path = optimal_policy_callback(policy, 0, [width-1, height * width - 3 - 2*width], 5, 7)
                policy_list.append(optimal_path)
                new_path_dict[(gamma,prob)] = unroll_policy_dict(optimal_path)

        
    elif world_type == "smallbig":
        height = 7
        width = 7
        for side in sides:
            for elem in side:
                gamma = elem[0]
                prob = elem[1]
                print(gamma, prob)
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
                optimal_path = optimal_policy_callback(policy, 0, [width*(height-1), height*width-1], height, width)
                # optimal_path = optimal_policy_callback(policy, 11*width+11, [18*width+11, 18*width+18], height, width)
                policy_list.append(optimal_path)
                new_path_dict[(gamma,prob)] = unroll_policy_dict(optimal_path)

    elif world_type == "cliff":
        height = 5
        width = 9
        for side in sides:
            for elem in side:
                gamma = elem[0]
                prob = elem[1]
                print(gamma, prob)
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
                new_path_dict[(gamma,prob)] = unroll_policy_dict(optimal_path)
    # print set of unique optimal paths
    unique_policies = unique_dicts(policy_list)

    combined_viz(combined_policies(policy_list), height, width)

    for side in sides:
        i = 0
        while i < (len(side) - 1):
            print(side[i], side[i+1])
            print(i, len(side))
            if new_path_dict[(side[i][0],side[i][1])] != new_path_dict[(side[i+1][0],side[i+1][1])]:
                print("Different")
                gamma = (side[i][0] + side[i+1][0])/2
                prob = (side[i][1] + side[i+1][1])/2
                height = 5
                width = 7
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
                optimal_path = optimal_policy_callback(policy, 0, [width-1, height * width - 3 - 2*width], 5, 7)
                if optimal_path in policy_list or optimal_path == {}:
                    print("Already in List")
                else:
                    side.append([gamma, prob])
                    side.sort()
                    i -= 1
                policy_list.append(optimal_path)
                new_path_dict[(gamma,prob)] = unroll_policy_dict(optimal_path)
                unique_policies = unique_dicts(policy_list)

                combined_viz(combined_policies(policy_list), height, width)
            i += 1