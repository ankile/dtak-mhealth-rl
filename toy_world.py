# implemented from https://towardsdatascience.com/reinforcement-learning-implement-grid-world-from-scratch-c5963765ebff

import numpy as np
# setting up world
ROWS = 5
COLS = 5
WIN_STATE = (4, 4)
LOSE_STATE = (2,3)
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
        self.board = np.zeros((ROWS, COLS))
        self.board[LOSE_STATE[0], LOSE_STATE[1]] = -1
        self.board[WIN_STATE[0], WIN_STATE[1]] = 10
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
    def print_board(self):
        world = []
        for i in range(ROWS):
            for j in range(COLS):
                if (i, j) == START:
                    world.append('o ')
                elif (i, j) == WIN_STATE:
                    world.append('X ')
                elif (i, j) == LOSE_STATE:
                    world.append('_ ')
                elif self.board[i][j] < 0.0: # is an obstacle
                    world.append('# ')
                else:
                    world.append('. ')
            world.append('\n')

        toDraw = ''.join(world)
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

    # implement Q(s, a) = SUM( P(s'|s, a)*[R(s, a, s') + gamma*U(s')] )
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

        print(self.policy, '\n')
        print(self.V)

    def take_action(self, action):
        position = self.State.next_move(action)
        return State(state = position)

    def reset(self):
        self.states = []
        self.State = State()

    def play(self, rounds = 5):
        current_round = 0
        self.value_iteration()
        # while current_round < rounds:
        #     if self.State.atEnd:
        #         # backprop
        #         reward = self.State.give_reward()
        #         self.state_values[self.State.state] = reward
        #         print('Game End Reward: ', reward)

        #         for state in reversed(self.states):
        #             reward = self.state_values[state] + self.lr * (reward - self.state_values[state])
        #             self.state_values[state] = round(reward, 3)
        #         self.reset()
        #         current_round += 1
        #     else:
        #         action = self.choose_action()
        #         self.states.append(self.State.next_move(action))
        #         print('current position {} action {}'.format(self.State.state, action))

        #         # takes the action, reaches next state:
        #         self.State = self.take_action(action)

        #         # mark end
        #         self.State.at_end()
        #         print('next state ', self.State.state)
        #         print ('----------------')

    # def show_values(self):
    #     for i in range(0, ROWS):
    #         print ('----------------------------')
    #         out = '| '
    #         # for j in range(0, COLS):
    #         #     out += str(self.state_values[(i, j)]).ljust(6) + '| '
    #         print(out)
    #     print ('----------------------------')


if __name__ == '__main__':
    agent = Agent()
    agent.play()
    agent.State.print_board()
    # print(agent.show_values())




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
