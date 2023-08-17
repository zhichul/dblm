import enum

DEBUG_MODE=True

class TensorInitializer(enum.Enum):
    CONSTANT = enum.auto()
    GAUSSIAN = enum.auto()
    UNIFORM = enum.auto()
    XAVIER = enum.auto()


FACTOR_VARIABLES="factor_variables"
FACTOR_FUNCTIONS="factor_functions"
NUMBER_OF_VARIABLES="nvars"
NUMBER_OF_VALUES="nvals"

GRAPH_FILE="graph.json"
FACTORS_FILE="factors.json"
