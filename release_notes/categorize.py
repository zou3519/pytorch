import re
import csv
import copy
import json
import argparse
import os
import textwrap
from common import dict_to_features, categories, subcategories, get_features

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
        if category == '':
            self.category = 'Uncategorized'
        else:
            self.category = category

    def serialize(self, cache=None):
        title = None
        if cache is not None:
            title = cache.get(self.hash).title
        return [self.hash, self.category, title]

    def has_category(self):
        return self.category is not None

class CommitList:
    def __init__(self, path, cache=None):
        # We assume the data is stored in a csv file in commit_hash,category
        # pairs.
        self.path = path
        self.commits = self.read_from_disk()
        self.cache = cache

    def read_from_disk(self):
        path = self.path
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

    def uncategorized(self):
        return [commit for commit in self.commits if not commit.has_category()]

    def filter(self, category='All'):
        if category == 'All':
            return self.commits
        return [commit for commit in self.commits if commit.category == category]


class Categorizer:
    def __init__(self, path, category='Uncategorized'):
        self.cache = CommitDataCache('data.json')
        self.commits = CommitList(path)

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

Select from: {', '.join(all_categories)}

        ''')
        print(view)
        choice = None
        while choice is None:
            value = input('category> ')
            if len(value) == 0:
                choice = self.category
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

        # Assign the category, write the category out to disk
        commit.category = category
        self.commits.write_to_disk()

def main():
    parser = argparse.ArgumentParser(description='Tool to help categorize commits')
    parser.add_argument('--category', type=str, default='Uncategorized',
                        help='Which category to filter by. "Uncategorized", "All", or a category name')
    parser.add_argument('file', help='The location of the commits CSV')

    args = parser.parse_args()
    categorizer = Categorizer(args.file, args.category)
    categorizer.categorize()


main()
