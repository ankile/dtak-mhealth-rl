# Import enum types
from enum import Enum


class Actions(Enum):
    """Action types for the agent"""
    LEFT = 0
    RIGHT = 1
    UP = 2
    DOWN = 3
