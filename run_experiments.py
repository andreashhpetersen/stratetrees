import os
import re
import csv
import json
import argparse
import numpy as np

from tqdm import tqdm
from glob import glob
from copy import deepcopy
from time import perf_counter

from trees.advanced import max_parts, max_parts3, boxes_to_tree
from trees.models import QTree, DecisionTree
from trees.utils import parse_from_sampling_log

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


def dump_json(tree, fp):
    if not EXPORT_UPPAAL:
        return

    path = fp.split('/')
    tree_path = path[:-1] + ['trees'] + path[-1:]
    uppaal_path = path[:-1] + ['uppaal'] + path[-1:]

    tree.save_as('/'.join(tree_path))
    # tree.save_as(fp.replace('.json', '_dtv.json'))
    tree.export_to_uppaal('/'.join(uppaal_path))


def write_results(data, model_names, model_dir):
    smallest = np.argmin(data[:,1,S_ID])
    with open(f'{model_dir}/generated/smallest.txt', 'w') as f:
        f.write(f'constructed_{smallest}')

    headers = [
        'model_name', 'time (best)', 'time (avg)', 'time (std)',
        'size (best)', 'size (avg)', 'size (std)'
    ]
    with open(f'{model_dir}/generated/dt_results.csv', 'w') as f:
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


def write_trans_results(data, path):
    with open(path, 'w') as f:
        writer = csv.writer(f)
        for d in data:
            writer.writerow(d)


def transitive_max_parts(tree, alg=max_parts3, max_iter=10):
    variables, actions = tree.variables, tree.actions

    with performance() as p:
        boxes = max_parts3(tree)

    time1 = p.time

    with performance() as p:
        ntree = boxes_to_tree(boxes, variables, actions)

    time2 = p.time

    i = 0
    best_n_boxes = len(boxes)
    best_n_tree = ntree.size
    best_tree = ntree
    data = [[i, len(boxes), ntree.size, time1, time2]]
    print(data[0])

    while i < max_iter and \
            (len(boxes) < best_n_boxes + 1 or ntree.size < best_n_tree + 1):

        i += 1

        with performance() as p:
            boxes = alg(ntree)

        time1 = p.time

        with performance() as p:
            ntree = boxes_to_tree(boxes, variables, actions)

        time2 = p.time

        res = [i, len(boxes), ntree.size, time1, time2]
        print(res)
        data.append(res)

        if len(boxes) < best_n_boxes:
            best_n_boxes = len(boxes)

        if ntree.size < best_n_tree:
            best_n_tree = ntree.size
            best_tree = ntree
            best_tree_i = i

    write_trans_results(data, './trans_results.csv')
    return best_tree, (np.array(data), best_tree_i)

def run_experiment(model_dir, k=10):
    qt_strat_file = f'{model_dir}/qt_strategy.json'
    sample_logs = glob(f'{model_dir}/samples/*')

    qtree = QTree(qt_strat_file)
    q_tree_data = np.array([qtree.size, 0])

    model_names = [
        'qt_strategy', 'dt_original',
        'old_max_parts', 'dt_old_max_parts',
        'new_max_parts', 'dt_new_max_parts',
        'new_max_parts_trans', 'dt_new_max_parts_trans',
    ] + [ 'dt_prune_{}'.format(re.findall(r"\d+", s)[0]) for s in sample_logs ]

    data = []
    for i in tqdm(range(k)):
        store_path = f'{model_dir}/generated/constructed_{i}'

        if EXPORT_UPPAAL:
            try:
                os.mkdir(store_path)
            except FileExistsError:
                pass

            try:
                os.mkdir(store_path + '/trees')
            except FileExistsError:
                pass

            try:
                os.mkdir(store_path + '/uppaal')
            except FileExistsError:
                pass

        res = run_single_experiment(qtree, sample_logs, store_path)
        data.append(np.vstack((q_tree_data, res)))

    data = np.array(data)
    write_results(data, model_names, model_dir)
    return model_names, data


def run_single_experiment(
        qtree: QTree, sample_logs: list[str], store_path: str
    ) -> np.ndarray:
    results = []

    with performance() as p:
        tree = qtree.to_decision_tree()

    results.append([tree.size, p.time])
    dump_json(tree, f'{store_path}/dt_original.json')


    # do old max_parts
    mp_tree, (data, best) = transitive_max_parts(tree, alg=max_parts, max_iter=20)
    results.append([data[0,1], data[0,3]])
    results.append([data[0,2], data[0,3] + data[0,4]])

    results.append([data[best,1], data[:best,3:].sum() + data[best,3]])
    results.append([data[best,2], data[:best + 1,3:].sum()])

    # with performance() as p:
    #     leaves = max_parts(tree)

    # results.append([len(leaves), p.time])

    # with performance() as p:
    #     mp_tree = boxes_to_tree(leaves, qtree.variables, qtree.actions)
    #     mp_tree.meta = qtree.meta

    # results.append([ mp_tree.size, p.time + results[-1][1] ])
    mp_tree.meta = qtree.meta
    dump_json(mp_tree, f'{store_path}/dt_old_max_parts.json')


    # do new max_parts
    mp_tree, (data, best) = transitive_max_parts(tree, max_iter=20)
    results.append([data[0,1], data[0,3]])
    results.append([data[0,2], data[0,3] + data[0,4]])

    results.append([data[best,1], data[:best,3:].sum() + data[best,3]])
    results.append([data[best,2], data[:best + 1,3:].sum()])

    mp_tree.meta = qtree.meta
    dump_json(mp_tree, f'{store_path}/dt_new_max_parts_trans.json')

    for sample_log in sample_logs:
        prune_tree = mp_tree.copy()
        samples = parse_from_sampling_log(sample_log)
        sample_size = int(re.findall(r'\d+', sample_log)[0])

        with performance() as p:
            prune_tree.count_visits(samples)
            prune_tree.emp_prune()
            leaves = max_parts(prune_tree)
            prune_tree = boxes_to_tree(leaves, qtree.variables, qtree.actions)
            prune_tree.meta = qtree.meta

        results.append([prune_tree.size, p.time])
        dump_json(prune_tree, f'{store_path}/dt_prune_{sample_size}.json')

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
