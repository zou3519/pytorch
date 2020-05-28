import re
import csv
import copy
import json
import argparse
import os
import textwrap
from common import dict_to_features, categories, subcategories, get_features, run

class CommitDataCache:
    def __init__(self, path):
        self.path = path
        self.data = {}
        if os.path.exists(path):
            self.data = self.read_from_disk()

    def get(self, commit):
        if commit not in self.data.keys():
            # Fetch and cache the data
            self.data[commit] = get_features(commit)
            self.write_to_disk()
        return self.data[commit]

    def read_from_disk(self):
        with open(self.path, 'r') as f:
            data = json.load(f)
            data = {commit: dict_to_features(dct)
                    for commit, dct in data.items()}
        return data

    def write_to_disk(self):
        data = {commit: features._asdict() for commit, features in self.data.items()}
        with open(self.path, 'w') as f:
            json.dump(data, f)

class Commit:
    def __init__(self, hash, category):
        self.hash = hash
        self.category = 'Uncategorized' if not category else category

    def serialize(self):
        return [self.hash, self.category]

class CommitList:
    # NB: Private ctor. Use `from_existing` or `create_new`.
    def __init__(self, path, commits):
        self.path = path
        self.commits = commits

    @staticmethod
    def from_existing(path):
        commits = CommitList.read_from_disk(path)
        return CommitList(path, commits)

    @staticmethod
    def create_new(path, base_version, new_version):
        hashes = CommitList.get_commits_between(base_version, new_version)
        commits = [Commit(h, 'Uncategorized') for h in hashes]
        return CommitList(path, commits)

    @staticmethod
    def read_from_disk(path):
        with open(path) as csvfile:
            reader = csv.reader(csvfile)
            rows = list(row for row in reader)
        assert all(len(row) >= 2 for row in rows)
        return [Commit(*row[:2]) for row in rows]

    def write_to_disk(self):
        path = self.path
        rows = [commit.serialize() for commit in self.commits]
        with open(path, 'w') as csvfile:
            writer = csv.writer(csvfile)
            for row in rows:
                writer.writerow(row)

    @staticmethod
    def get_commits_between(base_version, new_version):
        cmd = f'git merge-base {base_version} {new_version}'
        rc, merge_base, _ = run(cmd)
        assert rc == 0

        cmd = f'git log --reverse --oneline {merge_base}..{new_version} | cut -d " " -f 1'
        rc, commits, _ = run(cmd)
        assert rc == 0
        hashes = commits.split('\n')
        return hashes

    def filter(self, category='All'):
        if category == 'All':
            return self.commits
        return [commit for commit in self.commits if commit.category == category]
 
    def update(self, new_version):
        last_hash = self.commits[-1].hash
        hashes = CommitList.get_commits_between(last_hash, new_version)
        self.commits += [Commit(hash, 'Uncategorized') for hash in hashes[1:]]


class Categorizer:
    def __init__(self, path, category='Uncategorized'):
        self.cache = CommitDataCache('results/data.json')
        self.commits = CommitList.from_existing(path)

        # Special categories: 'Uncategorized', 'All'
        # All other categories must be real
        self.category = category

    def categorize(self):
        commits = self.commits.filter(self.category)
        i = 0
        while i < len(commits):
            cur_commit = commits[i]
            next_commit = commits[i + 1] if i + 1 < len(commits) else None
            jump_to = self.handle_commit(cur_commit, i + 1, len(commits))

            # Increment counter
            if jump_to is not None:
                i = jump_to
            elif next_commit is None:
                i = len(commits)
            else:
                i = commits.index(next_commit)

    def features(self, commit):
        return self.cache.get(commit.hash)

    def handle_commit(self, commit, i, total):
        all_categories = categories + subcategories
        features = self.features(commit)
        os.system('clear')
        view = textwrap.dedent(f'''\
[{i}/{total}]
================================================================================
{features.title}

{features.body}

Labels: {features.labels}
Files changed: {features.files_changed}

Current category: {commit.category}

Select from: {', '.join(all_categories)}

        ''')
        print(view)
        choice = None
        while choice is None:
            value = input('category> ')
            if len(value) == 0:
                choice = commit.category
                continue
            if value.isnumeric():
                return int(value) - 1
            choices = [cat for cat in all_categories
                       if cat.startswith(value)]
            if len(choices) != 1:
                print(f'Possible matches: {choices}, try again')
                continue
            choice = choices[0]
        print(f'\nSelected: {choice}')
        self.assign_category(commit, choice)
        return None

    def assign_category(self, commit, category):
        if category == commit.category:
            return
        commit.category = category
        self.commits.write_to_disk()

def main():
    parser = argparse.ArgumentParser(description='Tool to help categorize commits')
    parser.add_argument('--category', type=str, default='Uncategorized',
                        help='Which category to filter by. "Uncategorized", "All", or a category name')
    parser.add_argument('--file', help='The location of the commits CSV',
                        default='results/commitlist.csv')

    args = parser.parse_args()
    categorizer = Categorizer(args.file, args.category)
    categorizer.categorize()


if __name__ == '__main__':
    main()
