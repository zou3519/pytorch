import argparse
import re
import json

def find_missing_batching_rules(logfile):
    with open(logfile) as f:
        data = json.load(f)

    warnings = data['warnings']
    warning_msgs = [warning['message'] for warning in warnings]

    prog = re.compile('not implemented for (.*) falling back')

    def parse_batching_rule(warning_msg):
        result = prog.search(warning_msg)
        if result is None:
            return None
        return result.groups(0)[0]

    matches = [parse_batching_rule(msg) for msg in warning_msgs]
    matches = set([m for m in matches if m is not None])
    return list(matches)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parse logs for unsupported ops')
    parser.add_argument('--logfile', type=str)
    args = parser.parse_args()
    reports = find_missing_batching_rules(args.logfile)
    print('\n'.join(reports))
