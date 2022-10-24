import toy_world
import numpy as np

class World:
    # Create class instead for an experiment that we know we want to do:

    """
        class Exp1:
            def setReward:
                change reward value, distance, etc based on what we want to look at into the experiment (this might be a reward knob that we're looking at.)

        class Exp2:
            def setSize:
                so on and so on... 
    """

    def __init__(self):
        self.Agent = toy_world.Agent()

    def setWorldSize(self, rows, cols, start):
        self.Agent.State.cols = cols
        self.Agent.State.rows = rows

    def setRewardValues(self, poses, negs):
        pass

    def setDistanceReward(self, distance):
        coord1 = self.Agent.State.cols() / distance 