# implemented from https://towardsdatascience.com/reinforcement-learning-implement-grid-world-from-scratch-c5963765ebff

import numpy as np
# setting up world
ROWS = 3
COLS = 4
WIN_STATE = (0, 3)
LOSE_STATE = (1, 3)
START = (2, 0)
DETERMINISTIC = True

class State:
    def __init__(self, state = START):
        self.board = np.zeros([ROWS, COLS])

        # wall:
        # add knobs to randomize/scalable
        self.board[1,1] = -1
        self.state = state
        self.atEnd = False
        self.determine = DETERMINISTIC


    def give_reward(self):
        if self.state == WIN_STATE:
            return 1
        if self.state == LOSE_STATE:
            return -1
        else:
            return 0

    def at_end(self):
        if (self.state == WIN_STATE) or (self.state == LOSE_STATE):
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
                    if next_state != (1,1):
                        return next_state

            return self.state

    # visualize
    def print_board(self):
        for i in range(0, ROWS):
            print ('----------------')
            out = '| '
            for j in range(0, COLS):
                if self.board[i, j] == 1:
                    # this is just the start, right?
                    token = '*'
                if self.board[i, j] == -1:
                    token = 'z'
                if self.board[i, j] == 0:
                    token = '0'

                out += token + ' | '
            print(out)
        print ('----------------')

class Agent:
    def __init__(self):
        self.states = []
        self.actions = ['up', 'down', 'left', 'right']
        self.State = State()

        ##
        self.lr = 0.2
        self.explore_rate = 0.3
        ##

        # initial state reward
        self.state_values = {}
        for i in range(ROWS):
            for j in range(COLS):
                # initial value = 0 bc agent knows nothing at first!
                self.state_values[(i, j)] = 0

    def choose_action(self):
        # pick action with greatest expected value
        max_next_reward = 0
        action = ''

        if np.random.uniform(0, 1) <= self.explore_rate:
            action = np.random.choice(self.actions)
        else:
            for possible_action in self.actions:
                next_reward = self.state_values[self.State.next_move(possible_action)]
                if next_reward >= max_next_reward:
                    action = possible_action
                    max_next_reward = next_reward
        return action

    def take_action(self, action):
        position = self.State.next_move(action)
        return State(state = position)

    def reset(self):
        self.states = []
        self.State = State()

    def play(self, rounds = 5):
        current_round = 0
        while current_round < rounds:
            if self.State.atEnd:
                # backprop
                reward = self.State.give_reward()
                self.state_values[self.State.state] = reward
                print('Game End Reward: ', reward)

                for state in reversed(self.states):
                    reward = self.state_values[state] + self.lr * (reward - self.state_values[state])
                    self.state_values[state] = round(reward, 3)
                self.reset()
                current_round += 1
            else:
                action = self.choose_action()
                self.states.append(self.State.next_move(action))
                print('current position {} action {}'.format(self.State.state, action))

                # takes the action, reaches next state:
                self.State = self.take_action(action)

                # mark end
                self.State.at_end()
                print('next state ', self.State.state)
                print ('----------------')
    def show_values(self):
        for i in range(0, ROWS):
            print ('----------------------------')
            out = '| '
            for j in range(0, COLS):
                out += str(self.state_values[(i, j)]).ljust(6) + '| '
            print(out)
        print ('----------------------------')


if __name__ == '__main__':
    agent = Agent()
    agent.play(5)
    print(agent.show_values())




# knobs to change:
# - size of world
# - where observing states are
# - change reward/where they are
    # frequency of reward
# - obstacles/how to get around them


# ultimate goal:
# in this type of world, you can tell the difference between these types of world
# set up another world, and take the same character sand differentiate AB whereas other world AB look the same


#what world will allow me to differentiate the characters at all
    # go more and more extreme
