import json
import argparse
import numpy as np
from math import inf

from trees.models import Tree

parser = argparse.ArgumentParser()
parser.add_argument(
    'STRATEGY',
    help='If STRATEGY is `json`, create csv file for dtcontrol. If `dot` ' \
    'then convert it to a Tree and export to json'
)
parser.add_argument(
    'SAMPLES',
    nargs='?', default=None,
    help='Use sample file to generate controller'
)
args = parser.parse_args()

HARD_MIN, HARD_MAX = -99999, 99999

def make_from_samples(tree, states, path):
    with open(path, 'w') as f:
        f.write('#NON-PERMISSIVE\n')
        f.write('#BEGIN {} 1\n'.format(len(tree.variables)))
        for state in states:
            action = tree.get(state)
            f.write(','.join(map(str, state)) + f',{float(action)}\n')


def make_from_controller(tree, path):
    with open(path, 'w') as f:
        f.write('#NON-PERMISSIVE\n')
        f.write('#BEGIN {} 1\n'.format(len(tree.variables)))

        min_diff = inf
        for bs in tree.get_bounds():
            for i in range(1, len(bs)):
                diff = abs(bs[i] - bs[i-1])
                if diff < min_diff:
                    min_diff = diff

        eps = np.random.random() * min_diff
        for leaf in tree.get_leaves():
            cs = leaf.state.constraints
            smin, smax = cs[:,0], cs[:,1]
            smax -= eps
            smin[smin == -inf] = HARD_MIN
            smax[smax == inf] = HARD_MAX
            f.write(','.join(map(str, smin)) + f',{float(leaf.action)}\n')
            f.write(','.join(map(str, smax)) + f',{float(leaf.action)}\n')


if __name__ == '__main__':
    fp = args.STRATEGY

    path = fp.split('/')
    i = next(
        (i+1, s) for i, s in enumerate(path) if s == 'automated'
    )[0]
    model_dir = '/'.join(path[:i+1])

    if fp.endswith('.json'):
        tree = Tree.load_from_file(args.STRATEGY)

        path = model_dir + '/samples/dtcontrol_samples.csv'
        if args.SAMPLES is not None:
            samples = np.loadtxt(args.SAMPLES, delimiter=',')
            make_from_samples(tree, samples, path)
        else:
            make_from_controller(tree, path)

        meta = {
            'x_column_names': tree.variables,
        }
        fn = '/'.join(path.split('/')[:-1]) + '/dtcontrol_samples_config.json'
        with open(fn, 'w') as f:
            json.dump(meta, f)

    elif fp.endswith('.dot'):
        with open(model_dir + '/generated/smallest.txt', 'r') as f:
            D = f.readline()

        generated = f'{model_dir}/generated/{D}'
        tree_fp = f'{generated}/trees/dt_original.json'
        tree = Tree.load_from_file(tree_fp)

        dtcontrol_tree = Tree.parse_from_dot(
            fp, { v: v for v in tree.variables }
        )
        dtcontrol_tree.meta = tree.meta

        dtcontrol_tree.save_as(generated + '/trees/dtcontrol_tree.json')
        dtcontrol_tree.export_to_uppaal(
            generated + '/uppaal/dtcontrol_tree.json'
        )
