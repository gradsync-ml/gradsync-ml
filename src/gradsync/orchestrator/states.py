from enum import Enum, auto


class NodeState(Enum):
    FOLLOWER  = auto()
    CANDIDATE = auto()
    LEADER    = auto()
