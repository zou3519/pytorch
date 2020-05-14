from torch.testing._internal.common_utils import TestCase, run_tests
import torch
from torch import vmap, Tensor
from torch.autograd import gradcheck
import torch.nn.functional as F
import unittest
from collections import namedtuple
import itertools
import copy

class TestVmap(TestCase):
    def test_sum(self):
        pass

if __name__ == '__main__':
    run_tests()
