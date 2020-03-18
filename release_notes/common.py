from collections import namedtuple

categories = [
    'distributed',
    'rpc',
    'mobile',
    'jit',
    'visualization',
    'onnx',
    'caffe2',
    'quantization',
    'distributions',
    'amd',
    'cpp',
    'python',
    'skip',
]

subcategories = [
    'bc_breaking',
    'new_features',
    'improvements',
    'bug_fixes',
    'deprecations',
    'performance',
    'docs',
    'misc',
]


Features = namedtuple('Features', [
    'title',
    'body',
    'pr_number',
    'files_changed',
    'labels',
])


def dict_to_features(dct):
    return Features(
        title=dct['title'],
        body=dct['body'],
        pr_number=dct['pr_number'],
        files_changed=dct['files_changed'],
        labels=dct['labels'])


def features_to_dict(features):
    return dict(features._asdict())
