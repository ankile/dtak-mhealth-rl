def follow_policy(policy, height, width, initial_state, terminal_states):
    action_dict = {0: "L", 1: "R", 2: "U", 3: "D"}
    state = initial_state
    actions_taken = []
    seen_states = set()

    while state not in terminal_states and state not in seen_states:
        seen_states.add(state)
        row, col = state // width, state % width
        action = policy[row, col]
        actions_taken.append(action_dict[action])

        if action == 0:  # left
            col = max(0, col - 1)
        elif action == 1:  # right
            col = min(width - 1, col + 1)
        elif action == 2:  # up
            row = max(0, row - 1)
        elif action == 3:  # down
            row = min(height - 1, row + 1)

        state = row * width + col

    return "".join(actions_taken)


def get_all_absorbing_states(T, height, width):
    absorbing_states = set()

    for state in range(height * width):
        for action in range(4):
            if T[action, state, state] == 1:
                absorbing_states.add(state)
                break

    return absorbing_states


def param_generator(parameters):
    for param_name, param_values in parameters.items():
        for value in param_values:
            yield (param_name, value)
