import torch
from .registry import Registry

tensor_registry = Registry(torch.Tensor)
