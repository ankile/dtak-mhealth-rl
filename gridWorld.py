# implemented from https://towardsdatascience.com/reinforcement-learning-implement-grid-world-from-scratch-c5963765ebff
import numpy as np

world_actions = {
    'up': 0,
    'down': 1,
    'left': 2,
    'right': 3
}

class gridWorld:
    def __init__(self, rows, cols, start_state):
        self.rows = rows
        self.cols = cols
        # only allowing up, down, left, right actions for this gridWorld
        self.action_space_size = 4 
        self.trans_mat = np.ones((self.action_space_size, self.action_space_size)) / self.action_space_size
        self.reward_mat = np.zeros((self.rows, self.cols))
        self.state_space = np.zeros((self.rows, self.cols))
        self.state = start_state

    def setTransMat(self, trans_mat):
        if trans_mat.shape != self.trans_mat.shape:
            raise ValueError("Shape of transition matrix incorrect!")
        self.trans_mat = trans_mat

    def setRewardMat(self, reward_mat):
        if reward_mat.shape != self.reward_mat.shape:
            raise ValueError("Shape of reward matrix incorrect!")
        self.reward_mat = reward_mat

    def step(self, action):
        """
        possible actions include: up, down, left, right
        """
        state = self.state
        if action == 'up':
            next_state = (self.state[0] - 1, self.state[1])
        if action == 'down':
            next_state = (self.state[0] + 1, self.state[1])
        if action == 'left':
            next_state = (self.state[0], self.state[1] - 1)
        if action == 'right':
            next_state = (self.state[0], self.state[1] + 1)

        # checking if the next state is legal:
        if (next_state[0] >= 0) and (next_state[0] <= (self.rows - 1)):
            if (next_state[1] >= 0) and (next_state[1] <= (self.cols - 1)):
                state = next_state

        finished = bool(self.state_space[state[0], state[1]])
        reward = self.reward_mat[state[0], state[1]]
        return state, finished, reward

    # visualize
    def visualize(self, policy=[]):
        if len(policy) > 0:
            grid = []
            for i in range(self.rows):
                for j in range(self.cols):
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

        grid = []
        for i in range(self.rows):
            for j in range(self.cols):
                state = (i, j)
                if self.state_space[state[0], state[1]] == 0:
                    grid.append('. ')
                elif self.state_space[state[0], state[1]] == 1:
                    grid.append('X ')
                elif self.state_space[state[0], state[1]] == -1:
                    grid.append('_ ')

            grid.append('\n')

        toDraw = ''.join(grid)
        print(toDraw)

class Agent:
    def __init__(self):
        self.actions = ['up', 'down', 'left', 'right']
        self.gridWorld = gridWorld(4, 4, (0, 0))

        trans_mat =  np.array([[0.70, 0.10, 0.10, 0.10],
                            [0.10, 0.70, 0.10, 0.10],
                            [0.10, 0.10, 0.70, 0.10],
                            [0.10, 0.10, 0.10, 0.70]])
        reward_mat = np.array([[0, 0, 0, 0], 
                            [0, 0, -1, 0],
                            [0, 0, 0, 0], 
                            [0, 0, 0, 10]])
        
        self.gridWorld.setTransMat(trans_mat)
        self.gridWorld.setRewardMat(reward_mat)
        self.gridWorld.state_space[1][2] = -1
        self.gridWorld.state_space[3][3] = 1


        self.V = np.zeros((self.gridWorld.rows, self.gridWorld.cols))
        self.policy = self.create_initial_pol()
        self.gamma = 0.9
        # tolerance to know when to stop value iteration
        self.theta = 0.0001

    def create_initial_pol(self):
        policy = np.zeros((self.gridWorld.rows, self.gridWorld.cols))
        return policy

    # implement Q(s, a) = SUM( P(s'|s, a)*[R(s, a, s') + gamma*U(s')] )
    def updated_action_values(self):
        vals = np.zeros(len(self.actions))

        for action in self.actions:
            to_sum = []
            for a, prob in enumerate(self.gridWorld.trans_mat[world_actions[action]]):
                state, finished, reward = self.gridWorld.step(self.actions[a])
                to_sum.append(prob * (reward + (self.gamma * self.V[state])))

            vals[world_actions[action]] = sum(to_sum)

        return vals

    def value_iteration(self):
        while True:
            difference = 0
            for i in range(self.gridWorld.rows):
                for j in range(self.gridWorld.cols):
                    state = (i, j)
                    self.gridWorld.state = state
                    old_V = self.V[state]
                    v = self.updated_action_values()

                    self.policy[state] = np.argmax(v)
                    self.V[state] = np.max(v)

                    difference = max(difference, np.abs(old_V - self.V[state]))
            if difference < self.theta:
                break

        print(self.V)

    def reset(self):
        self.gridWorld = gridWorld(4, 4, (0, 0))

    def play(self, rounds = 5):
        self.value_iteration()


if __name__ == '__main__':
    agent = Agent()
    agent.play()
    agent.gridWorld.visualize(policy=agent.policy)