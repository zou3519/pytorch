import requests
import subprocess
import sys
import locale
import re
from collections import namedtuple
import json
from os.path import expanduser


def run(command):
    """Returns (return-code, stdout, stderr)"""
    p = subprocess.Popen(command, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, shell=True)
    output, err = p.communicate()
    rc = p.returncode
    enc = locale.getpreferredencoding()
    output = output.decode(enc)
    err = err.decode(enc)
    return rc, output.strip(), err.strip()


def readlines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
        lines = [line for line in lines if len(line) > 0]
        return lines


def commit_body(commit_hash):
    cmd = f'git log -n 1 --pretty=format:%b {commit_hash}'
    ret, out, err = run(cmd)
    return out if ret == 0 else None


def commit_title(commit_hash):
    cmd = f'git log -n 1 --pretty=format:%s {commit_hash}'
    ret, out, err = run(cmd)
    return out if ret == 0 else None


def commit_files_changed(commit_hash):
    cmd = f'git diff-tree --no-commit-id --name-only -r {commit_hash}'
    ret, out, err = run(cmd)
    return out.split('\n') if ret == 0 else None


def parse_pr_number(body, commit_hash, title):
    regex = r'Pull Request resolved: https://github.com/pytorch/pytorch/pull/([0-9]+)'
    matches = re.findall(regex, body)
    if len(matches) == 0:
        if 'revert' not in title.lower() and 'updating submodules' not in title.lower():
            print(f'[{commit_hash}: {title}] Could not parse PR number, ignoring PR')
        return None
    if len(matches) > 1:
        print(f'[{commit_hash}: {title}] Got two PR numbers, using the first one')
        return matches[0]
    return matches[0]


def get_ghstack_token():
    pattern = 'github_oauth = (.*)'
    with open(expanduser('~/.ghstackrc'), 'r+') as f:
        config = f.read()
    matches = re.findall(pattern, config)
    if len(matches) == 0:
        raise RuntimeError("Can't find a github oauth token")
    return matches[0]

token = get_ghstack_token()
headers = {"Authorization": f"token {token}"}

def run_query(query):
    request = requests.post('https://api.github.com/graphql', json={'query': query}, headers=headers)
    if request.status_code == 200:
        return request.json()
    else:
        raise Exception("Query failed to run by returning code of {}. {}".format(request.status_code, query))


def gh_labels(pr_number):
    query = f"""
    {{
      repository(owner: "pytorch", name: "pytorch") {{
        pullRequest(number: {pr_number}) {{
          labels(first: 10) {{
            edges {{
              node {{
                name
              }}
            }}
          }}
        }}
      }}
    }}
    """
    query = run_query(query)
    edges = query['data']['repository']['pullRequest']['labels']['edges']
    return [edge['node']['name'] for edge in edges]

def get_features(commit_hash):
    title, body, files_changed = (
        commit_title(commit_hash),
        commit_body(commit_hash),
        commit_files_changed(commit_hash))
    pr_number = parse_pr_number(body, commit_hash, title)
    labels = []
    if pr_number is not None:
        labels = gh_labels(pr_number)
    return {
        'title': title,
        'body': body,
        'pr_number': pr_number,
        'files_changed': files_changed,
        'labels': labels,
    }


def main():
    commits = readlines('commit_list.txt')

    features = map(get_features, commits)
    data = {commit: feature for commit, feature in zip(commits, features)
            if feature is not None}
    with open('data.json', 'w') as outfile:
        json.dump(data, outfile)


main()
