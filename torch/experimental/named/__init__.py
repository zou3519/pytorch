from .torch import *
from .tensor import *


def load_named_lib():
    torch_registry.monkey_patch()
    tensor_registry.monkey_patch()


def unload_named_lib():
    torch_registry.undo_monkey_patch()
    tensor_registry.undo_monkey_patch()
