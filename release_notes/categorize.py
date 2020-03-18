import re
import copy
import json
import argparse
import os
from common import dict_to_features, categories, subcategories

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
        commits = copy.deepcopy(self.commits)
        for i, commit in enumerate(commits):
            self.handle_commit(commit, i + 1, len(commits))

    def handle_commit(self, commit, i, total):
        all_categories = categories + subcategories
        features = self.data[commit]
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
            choices = [cat for cat in all_categories
                       if cat.startswith(value)]
            if len(choices) != 1:
                print(f'Possible matches: {choices}, try again')
                continue
            choice = choices[0]
        print(f'\nSelected: {choice}')
        self.assign_category(commit, choice)

    def assign_category(self, commit, category):
        if category == self.category:
            return

        # Write to the category file
        with open(f'results/{category}.txt', 'a') as f:
            metadata = self.data[commit].title
            f.write(f'{commit}  # {metadata}\n')

        # Remove from the current category file
        self.commits = [com for com in self.commits
                        if not com.startswith(commit)]
        with open(f'{self.path}', 'w+') as f:
            lines = [f'{commit}  # {self.data[commit].title}' for commit in self.commits]
            f.write('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description='Re-categorize commits from file')
    parser.add_argument('file', help='which file to re-categorize. Must look like results/{category}.txt')
    args = parser.parse_args()
    categorizer = Categorizer(args.file)
    categorizer.categorize()


main()
