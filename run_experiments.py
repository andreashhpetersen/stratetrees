import os
import re
import csv
import json
import argparse
import numpy as np

from glob import glob
from copy import deepcopy
from time import perf_counter

from trees.advanced import max_parts, boxes_to_tree
from trees.models import Tree
from trees.utils import load_trees, parse_from_sampling_log, count_visits, \
    import_uppaal_strategy

parser = argparse.ArgumentParser()
parser.add_argument(
    'MODEL_DIR',
    help='The directory containing the model to run experiments on.'
)
parser.add_argument(
    '-k', '--repeats',
    nargs='?', type=int, const=1, default=1,
    help='Specify how many times the experiment should run (default 1)'
)
parser.add_argument(
    '-u', '--store-uppaal',
    action='store_true',
    help='Export the constructed strategies to a UPPAAL Stratego format'
)
parser.add_argument(
    '-p', '--print-results',
    action='store_true',
    help='Print results to screen'
)
args = parser.parse_args()

EXPORT_UPPAAL = args.store_uppaal
S_ID, T_ID = 0, 1   # size and time


class performance:
    def __enter__(self):
        self.start = perf_counter()
        return self

    def __exit__(self, *args):
        self.stop = perf_counter()
        self.time = self.stop - self.start


def dump_json(strategy, fp):
    if not EXPORT_UPPAAL:
        return

    with open(fp, 'w') as f:
        json.dump(strategy, f, indent=4)


def write_results(data, model_names, model_dir):
    smallest = np.argmin(data[:,1,S_ID])
    with open(f'{model_dir}/smallest.txt', 'w') as f:
        f.write(f'constructed_{smallest}')

    headers = [
        'model_name', 'time (best)', 'time (avg)', 'time (std)',
        'size (best)', 'size (avg)', 'size (std)'
    ]
    with open(f'{model_dir}/dt_results.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        for i in range(len(model_names)):
            row = [model_names[i]]
            model_d = data[:,i].T
            row.extend([
                model_d[T_ID].min(),
                model_d[T_ID].mean(),
                model_d[T_ID].std(),
                model_d[S_ID].min(),
                model_d[S_ID].mean(),
                model_d[S_ID].std(),
            ])
            writer.writerow(row)


def run_experiment(model_dir, k=10):
    qt_strat_file = f'{model_dir}/qt_strategy.json'
    sample_logs = glob(f'{model_dir}/sample_*.log')

    # qtrees, variables, actions, meta = load_trees(qt_strat_file, verbosity=1)
    loc_qtrees, variables, actions, meta = import_uppaal_strategy(qt_strat_file)

    size = 0
    for loc, qtrees in loc_qtrees.items():
        size += sum([r.size for r in qtrees.values()])

    q_tree_data = np.array([size, 0])

    model_names = [
        'qt_strategy', 'dt_original', 'max_parts', 'dt_max_parts'
    ] + [ 'dt_prune_{}'.format(re.findall(r"\d+", s)[0]) for s in sample_logs ]

    data = []
    for i in range(k):
        store_path = f'{model_dir}/constructed_{i}'

        if EXPORT_UPPAAL:
            try:
                os.mkdir(store_path)
            except FileExistsError:
                pass

        res = run_single_experiment(
            loc_qtrees, variables, actions, meta, sample_logs, store_path
        )
        data.append(np.vstack((q_tree_data, res)))

    data = np.array(data)
    write_results(data, model_names, model_dir)
    return model_names, data


def run_single_experiment(
        loc_qtrees, variables, actions, meta, sample_logs, store_path
):
    results = []
    org_meta = meta

    with performance() as p:
        loc_tree = {}
        for loc, qtrees in loc_qtrees.items():
            loc_tree[loc] = Tree.merge_qtrees(qtrees.values(), variables, actions)

    results.append([sum([t.size for t in loc_tree.values()]), p.time])

    meta = deepcopy(org_meta)
    for loc, tree in loc_tree.items():
        meta = tree.update_meta(meta, loc=loc)
    dump_json(meta, f'{store_path}/dt_original.json')

    with performance() as p:
        loc_leaves = {}
        for loc, tree in loc_tree.items():
            loc_leaves[loc] = max_parts(tree)

    results.append([sum([len(ls) for ls in loc_leaves.values()]), p.time])

    with performance() as p:
        mp_loc_tree = {}
        for loc, leaves in loc_leaves.items():
            mp_loc_tree[loc] = boxes_to_tree(leaves, variables, actions)

    results.append([
        sum([t.size for t in mp_loc_tree.values()]), p.time + results[-1][1]
    ])

    meta = deepcopy(org_meta)
    for loc, tree in mp_loc_tree.items():
        meta = tree.update_meta(meta, loc=loc)
    dump_json(meta, f'{store_path}/dt_max_parts.json')

    for sample_log in sample_logs:
        prune_tree = mp_tree.copy()
        samples = parse_from_sampling_log(sample_log)
        sample_size = len(samples)

        with performance() as p:
            count_visits(prune_tree, samples)
            prune_tree.emp_prune()
            leaves = max_parts(prune_tree)
            prune_tree = boxes_to_tree(leaves, variables, actions)

        results.append([prune_tree.size, p.time])
        prune_tree.export_to_uppaal(
            f'{store_path}/dt_prune_{sample_size}.json', meta
        )

    return np.array(results)


if __name__ == '__main__':
    model_dir, k = args.MODEL_DIR, args.repeats
    model_names, data = run_experiment(model_dir, k=k)
    if args.print_results:
        for i in range(len(model_names)):
            model_d = data[:,i].T
            print(model_names[i])
            print('\tTime:\t\t(avg)\t\t(std)\t\t(best)')
            print('\t     \t\t{:0.2f}\t\t{:0.2f}\t\t{:0.2f}'.format(
                model_d[T_ID].mean(),
                model_d[T_ID].std(),
                model_d[T_ID].min()
            ))
            print('\tSize:\t\t(avg)\t\t(std)\t\t(best)')
            print('\t     \t\t{:0.2f}\t\t{:0.2f}\t\t{}'.format(
                model_d[S_ID].mean(),
                model_d[S_ID].std(),
                int(model_d[S_ID].min())
            ))
