import re
import copy
import json
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
        for commit in commits:
            self.handle_commit(commit)

    def handle_commit(self, commit):
        all_categories = categories + subcategories
        features = self.data[commit]
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
        print('\n\n\n\n\n')
        self.assign_category(commit, choice)

    def assign_category(self, commit, category):
        if category == self.category:
            return

        # Write to the category file
        with open(f'results/{category}.txt', 'a+') as f:
            metadata = self.data[commit].title
            f.write(f'{commit}  # {metadata}')

        # Remove from the current category file
        self.commits = [com for com in self.commits
                        if not com.startswith(commit)]
        with open(f'{self.path}', 'w+') as f:
            lines = [f'{commit}  # {self.data[commit].title}' for commit in self.commits]
            f.writelines(lines)


def main():
    path = 'results/python.txt'
    categorizer = Categorizer(path)
    categorizer.categorize()


main()
