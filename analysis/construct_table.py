import torch
import warnings
import argparse
from dataclasses import dataclass
from typing import Optional
import json

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


@dataclass
class TestReport:
    op: Optional[str]
    test_name: str
    test_file: str
    error: Optional[str]

def find_unsupported_ops(logfile):
    with open(logfile) as f:
        data = json.load(f)
        content = f.readlines()

    def parse_report(node):
        test_file, _, test_name = node['nodeid'].split('::')
        return TestReport(
            op=get_op(test_name),
            test_name=test_name,
            test_file=test_file,
            error=node['call']['crash']['message'],
        )

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

    failed_tests = [parse_report(test)
                    for test in data['tests'] if test['outcome'] == 'failed']
    return failed_tests

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parse logs for unsupported ops')
    parser.add_argument('--logfile', type=str)
    parser.add_argument('--type', type=str)
    parser.add_argument('--date', type=str)
    args = parser.parse_args()
    reports = find_unsupported_ops(args.logfile)
    for report in reports:
        print(f'{report.op},{report.test_name},{report.test_file},{args.type},{args.date},"{report.error}"')
