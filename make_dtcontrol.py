import json
import argparse
import numpy as np

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

def make_data(tree, states, path):
    fn = path + '/dtcontrol_samples.csv'
    with open(fn, 'w') as f:
        f.write('#NON-PERMISSIVE\n')
        f.write('#BEGIN {} 1\n'.format(len(tree.variables)))
        for state in states:
            action = tree.get(state)
            f.write(','.join(map(str, state)) + f',{action}\n')

    meta = {
        'x_column_names': tree.variables,
        'y_column_names': tree.actions
    }
    fn = path + '/dtcontrol_meta.json'
    json.dump(meta, fn)


if __name__ == '__main__':
    tree = Tree.load_from_file(args.STRATEGY)

    samples_fp = args.SAMPLES
    samples = np.loadtxt(samples_fp, delimiter=',')

    path = '/'.join(samples_fp.split('/')[:-1])
    make_data(tree, samples, path)
