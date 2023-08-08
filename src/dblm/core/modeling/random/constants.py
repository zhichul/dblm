import enum


class TensorInitializer(enum.Enum):
    CONSTANT = enum.auto()
    GAUSSIAN = enum.auto()
    UNIFORM = enum.auto()
