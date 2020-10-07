import os
import re

DOC_ROOT = '/scratch/rzou/pt/workspace/docs/source'
FILES = {
    'torch': 'torch.rst',
    'tensor': 'tensors.rst',
    'nn.F': 'nn.functional.rst',
}

def get_torch_functions():
    with open(f"{DOC_ROOT}/{FILES['torch']}") as f:
        content = f.readlines()

    # All of the operators have 4 spaces before they begin
    def is_op(line):
        if line[:4] != ' ' * 4:
            return False
        candidate = line[4:]
        if ' ' in candidate:
            return False
        if not re.match(r'^\w+$', candidate):
            return False
        return True

    ops = [line.strip() for line in content if is_op(line)]
    return list(set(ops))


def get_tensor_methods():
    with open(f"{DOC_ROOT}/{FILES['tensor']}") as f:
        content = f.readlines()

    def get_op(line):
        m = re.search(r'automethod:: (\w+)$', line)
        if m is None:
            return None
        return m.group(1)

    ops = [get_op(line) for line in content]
    ops = [op for op in ops if op is not None]
    return ops


def get_nn_functions():
    with open(f"{DOC_ROOT}/{FILES['nn.F']}") as f:
        content = f.readlines()

    def get_op(line):
        m = re.search(r'autofunction:: (\w+)$', line)
        if m is None:
            return None
        return m.group(1)

    ops = [get_op(line) for line in content]
    ops = [op for op in ops if op is not None]
    return ops


def zip_with(lst, value):
    return [(elt, value) for elt in lst]


def get_ops():
    results = zip_with(get_torch_functions(), 'torch')
    results += zip_with(get_tensor_methods(), 'tensor')
    results += zip_with(get_nn_functions(), 'nn')

    ops = {}
    for op, ns in results:
        if op not in ops.keys():
            ops[op] = []
        ops[op].append(ns)
    return ops


def print_csv(ops):
    print('op, namespaces')
    for op in ops.keys():
        namespaces = ','.join(ops[op])
        print(f'{op}, "{namespaces}"')


# Manually update this list with the things from test_vmap.py
OPS_WITH_BATCHING_RULES = set([
    # unary pointwise
    'abs', 'acos', 'asin', 'atan', 'ceil', 'cos', 'cosh', 'digamma',
    'exp', 'expm1', 'floor', 'frac', 'lgamma', 'log', 'log10p', 'log1p',
    'log2', 'neg', 'reciprocal', 'relu', 'round', 'rsqrt', 'sigmoid',
    'sign', 'sin', 'sinh', 'sqrt', 'tan', 'tanh', 'trunc',

    # binary pointwise
    'add', 'sub', 'mul', 'div', 'pow',

    'bmm', 'cat', 'conj', 'chunk', 'diagonal', 'dot', 'expand_as',
    'is_complex', 'movedim', 'mm', 'mv', 'narrow', 'select',
    'stack', 'slice', 'reshape', 'reshape_as', 'result_type',
    'split', 't', 'T',

    # to 
    'to', 'double', 'float', 'int', 'long', 'byte', 'bool', 'bfloat16',
    'cuda', 'cpu', 'half',

    'unfold', 'unbind', 'view', 'view_as',
])


def main():
    ops = get_ops()
    print(f'Number of ops: {len(ops.keys())}')
    inplace = [op for op in ops.keys() if op[-1] == '_']
    print(f'Number of in-place ops: {len(inplace)}')
    print(f'Number of ops with batching rules: {len(OPS_WITH_BATCHING_RULES)}') 


if __name__ == '__main__':
    main()
