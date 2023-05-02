# Import enum types
from enum import Enum


class Action(Enum):
    """Action types for the agent"""

    LEFT = 0
    RIGHT = 1
    UP = 2
    DOWN = 3


class TransitionMode(Enum):
    """Transition modes for the MDP"""

    """
    Simple transition mode: the agent moves in the direction it chooses with
    probability p, and stays in place with probability 1-p.
    """
    SIMPLE = "simple"

    """
    Full transition mode: the agent moves in the direction it chooses with
    probability p, and moves in a random direction (excluding the one chosen)
    with probability 1-p. This includes staying in place.
    """
    FULL = "full"
