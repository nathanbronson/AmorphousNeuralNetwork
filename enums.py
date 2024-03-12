from enum import Enum, auto

class Role(Enum):
    INPUT     =   auto()
    GROUND    =   auto()
    NETWORK   =   auto()
    OUTPUT    =   auto()

class Metric(Enum):
    ACCURACY        =   auto()
    STEP            =   auto()
    TRAINING_STEP   =   auto()
    NODES           =   auto()
    REPS            =   auto()

class Technique(Enum):
    BACKPROPOGATE      =   auto()
    GAIN_OF_FUNCTION   =   auto()
    BPGOF              =   auto()

class ConnectionStrategy(Enum):
    EVEN          =   auto()
    OUTPUT_HALF   =   auto()
    OUTPUT_75     =   auto()
    OUTPUT_90     =   auto()
    OUTPUT_99     =   auto()

class NodeStrategy(Enum):
    FIXED         =   auto()
    ADD_PLATEAU   =   auto()

class ExtraNodeHandling(Enum):
    SAVE    =   auto()
    RESET   =   auto()

class AttackStrategy(Enum):
    RESTART   =   auto()
    DECAY     =   auto()