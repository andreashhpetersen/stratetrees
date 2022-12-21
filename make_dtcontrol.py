import json
import argparse
import numpy as np
from math import inf

from trees.models import Tree

parser = argparse.ArgumentParser()
parser.add_argument(
    'STRATEGY',
    help='The dt strategy to generate actions from'
)
parser.add_argument(
    'SAMPLES',
    help='The sample file'
)
args = parser.parse_args()

HARD_MIN, HARD_MAX = -99999, 99999

def make_data(tree, states, path):
    fn = path + '/dtcontrol_samples.csv'
    with open(fn, 'w') as f:
        f.write('#NON-PERMISSIVE\n')
        f.write('#BEGIN {} 1\n'.format(len(tree.variables)))
        # for state in states:
        #     action = tree.get(state)
        #     f.write(','.join(map(str, state)) + f',{float(action)}\n')
        for leaf in tree.get_leaves():
            cs = leaf.state.constraints
            smin, smax = cs[:,0], cs[:,1]
            smax -= 0.0001
            smin[smin == -inf] = HARD_MIN
            smax[smax == inf] = HARD_MAX
            f.write(','.join(map(str, smin)) + f',{float(leaf.action)}\n')
            f.write(','.join(map(str, smax)) + f',{float(leaf.action)}\n')


    meta = {
        'x_column_names': tree.variables,
    }
    fn = path + '/dtcontrol_samples_config.json'
    with open(fn, 'w') as f:
        json.dump(meta, f)


if __name__ == '__main__':
    tree = Tree.load_from_file(args.STRATEGY)

    samples_fp = args.SAMPLES
    samples = np.loadtxt(samples_fp, delimiter=',')

    path = '/'.join(samples_fp.split('/')[:-1])
    make_data(tree, samples, path)
