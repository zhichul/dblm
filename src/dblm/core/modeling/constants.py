import dataclasses
import enum
from typing import Optional

DEBUG_MODE=True

class TensorInitializer(enum.Enum):
    CONSTANT = enum.auto()
    GAUSSIAN = enum.auto()
    UNIFORM = enum.auto()
    XAVIER = enum.auto()

@dataclasses.dataclass
class UniformlyRandomInitializer:
    min :float = 0.
    max :float = 1.

@dataclasses.dataclass
class GaussianInitializer:
    mean :float = 0.
    std :float = 1.
    min: Optional[float] = None
    max: Optional[float] = None

class DiscreteNoise(enum.Enum):
    UNIFORM = enum.auto()

class SwitchingMode(enum.Enum):
    MIXTURE = enum.auto()
    VARPAIRVAL = enum.auto()

FACTOR_VARIABLES="factor_variables"
FACTOR_FUNCTIONS="factor_functions"
NUMBER_OF_VARIABLES="nvars"
NUMBER_OF_VALUES="nvals"

GRAPH_FILE="graph.json"
FACTORS_FILE="factors.json"
