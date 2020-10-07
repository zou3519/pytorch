import os
import re
import torch
import warnings
import argparse

def is_op(candidate):
    if hasattr(torch.Tensor, candidate):
        return True
    if hasattr(torch, candidate):
        return True
    if hasattr(torch.nn.functional, candidate):
        return True
    if hasattr(torch.nn, candidate):
        return True
    return False


def find_unsupported_ops(logfile):
    with open(logfile) as f:
        content = f.readlines()

    failed_tests = []
    for line in content:
        m = re.search(r'^F .*::(\w+)$', line)
        if m is None:
            continue
        failed_tests.append(m.group(1))

    def get_op(failed_test):
        # account for dunder methods
        processed_failed_test = failed_test.replace('___', '_')
        is_dunder = len(processed_failed_test) != len(failed_test)
        if is_dunder:
            result = processed_failed_test.split('_')[1]
            return f'__{result}__'

        parts = failed_test.split('_')[1:]
        if len(parts) == 1:
            return parts[0]

        first = parts[0]
        second = parts[1]
        # In-place
        if len(parts) >= 3 and parts[2] == '':
            second = second + '_'
        candidate = f'{first}_{second}'
        if is_op(candidate):
            return candidate
        if is_op(first):
            return first
        warnings.warn(f"could not parse op from {failed_test}")
        return None

    unsupported_ops = set([get_op(test) for test in failed_tests])
    return list(filter(lambda x: x is not None, list(unsupported_ops)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parse logs for unsupported ops')
    parser.add_argument('--logfile', type=str)
    args = parser.parse_args()
    ops = find_unsupported_ops(args.logfile)
    print(f'Unsupported ops: {len(ops)}')
