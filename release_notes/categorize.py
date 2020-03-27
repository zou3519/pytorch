import re
import copy
import json
import argparse
import os
from common import dict_to_features, categories, subcategories, get_features

def readlines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.split('#')[0] for line in lines]
        lines = [line.strip() for line in lines]
        lines = [line for line in lines if len(line) > 0]
        return lines


class Categorizer:
    def __init__(self, path):
        path_pattern = r'results/(.*)\.txt'
        matches = re.findall(path_pattern, path)
        assert len(matches) == 1

        self.path = path
        self.category = matches[0]
        with open('data.json', 'r') as f:
            data = json.load(f)
            self.data = {commit: dict_to_features(dct)
                         for commit, dct in data.items()}

        self.commits = readlines(path)

    def categorize(self):
        i = 0
        while i < len(self.commits):
            cur_commit = self.commits[i]
            next_commit = self.commits[i + 1] if i + 1 < len(self.commits) else None
            jump_to = self.handle_commit(cur_commit, i + 1, len(self.commits))

            # Increment counter
            if jump_to is not None:
                i = jump_to
            elif next_commit is None:
                i = len(self.commits)
            else:
                i = self.commits.index(next_commit)

    def features(self, commit):
        if commit in self.data.keys():
            return self.data[commit]
        else:
            # We didn't preproces it, so just get it now
            # TODO: maybe cache
            return get_features(commit)

    def handle_commit(self, commit, i, total):
        all_categories = categories + subcategories
        features = self.features(commit)
        os.system('clear')
        print(f'[{i}/{total}]')
        print('=' * 80)
        print(features.title)
        print('\n')
        print(features.body)
        print('\n')
        print('Labels:', features.labels)
        print('Files changed:', features.files_changed)
        print('\n')
        print("Select from:", ', '.join(all_categories))
        print('\n')
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
        self.assign_category(commit, choice, features)
        return None

    def assign_category(self, commit, category, features):
        if category == self.category:
            return

        # Write to the category file
        with open(f'results/{category}.txt', 'a') as f:
            metadata = features.title
            f.write(f'{commit}  # {metadata}\n')

        # Remove from the current category file
        self.commits = [com for com in self.commits
                        if not com.startswith(commit)]
        with open(f'{self.path}', 'w+') as f:
            lines = [f'{commit}  # {features.title}' for commit in self.commits]
            f.write('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description='Re-categorize commits from file')
    parser.add_argument('file', help='which file to re-categorize. Must look like results/{category}.txt')
    args = parser.parse_args()
    categorizer = Categorizer(args.file)
    categorizer.categorize()


main()
