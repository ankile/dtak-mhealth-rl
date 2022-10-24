# implemented from https://towardsdatascience.com/reinforcement-learning-implement-grid-world-from-scratch-c5963765ebff

import numpy as np
# setting up world
ROWS = 4
COLS = 4
WIN_STATE = (3, 3)
LOSE_STATE = (1,2)
START = (0, 0)
DETERMINISTIC = True

world_actions = {
    'up': 0,
    'down': 1,
    'left': 2,
    'right': 3
}

class State:
    def __init__(self, state = START):
        self.rows = ROWS
        self.cols = COLS
        self.lose_state = LOSE_STATE
        self.win_state = WIN_STATE
        self.board = np.zeros((self.rows, self.cols))
        self.lose_state_val = -1
        self.win_state_val = 10
        self.board[LOSE_STATE[0], LOSE_STATE[1]] = self.lose_state_val
        self.board[WIN_STATE[0], WIN_STATE[1]] = self.win_state_val
        self.state = state
        self.atEnd = False
        self.determine = DETERMINISTIC
        self.T = 0.7


    def give_reward(self, state):
        return self.board[state[0], state[1]]

    def at_end(self):
        if (self.state == WIN_STATE):
            self.atEnd = True

    def next_move(self, action):
        """
        possible actions include: up, down, left, right
        """
        if self.determine:
            if action == 'up':
                next_state = (self.state[0] - 1, self.state[1])
            if action == 'down':
                next_state = (self.state[0] + 1, self.state[1])
            if action == 'left':
                next_state = (self.state[0], self.state[1] - 1)
            if action == 'right':
                next_state = (self.state[0], self.state[1] + 1)

            # checking if the next state is legal:
            if (next_state[0] >= 0) and (next_state[0] <= (ROWS - 1)):
                if (next_state[1] >= 0) and (next_state[1] <= (COLS - 1)):
                    return next_state

            return self.state

    # visualize
    def visualize(self, world=True, policy=[]):
        if len(policy) > 0:
            grid = []
            for i in range(ROWS):
                for j in range(COLS):
                    state = (i, j)
                    if policy[state] == 0:
                        grid.append(u'\u2191 ')
                    elif policy[state] == 1:
                        grid.append(u'\u2193 ')
                    elif policy[state] == 2:
                        grid.append(u'\u2190 ')
                    else:
                        grid.append(u'\u2192 ')
                grid.append('\n')

            toDraw = ''.join(grid)
            print(toDraw)

        if world:
            grid = []
            for i in range(ROWS):
                for j in range(COLS):
                    state = (i, j)
                    if state == START:
                        grid.append('o ')
                    elif state == WIN_STATE:
                        grid.append('X ')
                    elif state == LOSE_STATE:
                        grid.append('_ ')
                    else:
                        grid.append('. ')
                grid.append('\n')

            toDraw = ''.join(grid)
            print(toDraw)

class Agent:
    def __init__(self):
        self.states = []
        self.actions = ['up', 'down', 'left', 'right']
        self.State = State()
        self.V = np.zeros((ROWS, COLS))
        self.policy = self.create_initial_pol()
        self.gamma = 0.9
        # tolerance to know when to stop value iteration
        self.theta = 0.0001

    def create_initial_pol(self):
        policy = np.zeros((ROWS, COLS))
        return policy

    # Generating Q
    def updated_action_values(self):
        vals = np.zeros(len(self.actions))

        for action in self.actions:
            to_sum = []
            action_set = set(self.actions)
            action_set.remove(action)

            to_sum.append(self.State.T * self.V[self.State.next_move(action)[0], self.State.next_move(action)[1]] +
                         ((1-self.State.T)/3) * sum([self.V[self.State.next_move(a)[0], self.State.next_move(a)[1]] for a in action_set]))

            vals[world_actions[action]] = sum(to_sum)

        return vals

    def value_iteration(self):
        while True:
            difference = 0
            for i in range(ROWS):
                for j in range(COLS):
                    state = (i, j)
                    self.State.state = state
                    old_V = self.V[state]
                    v = self.updated_action_values()

                    self.policy[state] = np.argmax(v)
                    self.V[state] = self.State.give_reward(state) + self.gamma * np.max(v)

                    difference = max(difference, np.abs(old_V - self.V[state]))
            if difference < self.theta:
                break

        print(self.V)

    def take_action(self, action):
        position = self.State.next_move(action)
        return State(state = position)

    def reset(self):
        self.states = []
        self.State = State()

    def play(self, rounds = 5):
        self.value_iteration()


if __name__ == '__main__':
    agent = Agent()
    agent.play()
    agent.State.visualize(policy=agent.policy)




# knobs to change:
# - size of world
# - where observing states are
# - change reward/where they are
    # frequency of reward
    # distribution of reward
# - obstacles/how to get around them

# Experiment design:
# - For each knob:
    # Run world for each different type of agent
        # vary the 'agent knob' by some amount to observe changes
            # record different policies and behavior.


# **NOTE: how to define the 'optimal' parameters (gamma, transition probs, etc.)**



# ultimate goal:
# in this type of world, you can tell the difference between these types of world
# set up another world, and take the same character sand differentiate AB whereas other world AB look the same


#what world will allow me to differentiate the characters at all
    # go more and more extreme
