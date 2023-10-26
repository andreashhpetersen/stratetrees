import os
import re
import csv
import json
import shutil
import tempfile
import numpy as np

from tqdm import tqdm
from glob import glob
from copy import deepcopy
from time import perf_counter

import gymnasium as gym
import uppaal_gym

from trees.advanced import max_parts, leaves_to_tree, minimize_tree
from trees.loaders import SklearnLoader
from trees.models import QTree, DecisionTree
from trees.utils import parse_from_sampling_log, performance
from viper.viper import viper
from viper.wrappers import ShieldOracleWrapper


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


def run_experiment(model_dir, k=10):
    qt_strat_file = f'{model_dir}/qt_strategy.json'
    sample_logs = glob(f'{model_dir}/samples/*')

    qtree = QTree(qt_strat_file)
    q_tree_data = np.array([qtree.size, 0])

    model_names = [
        'qt_strategy', 'dt_original',
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

    # do new max_parts
    mp_tree, (data, best) = minimize_tree(tree, max_iter=20, verbose=False)
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
            prune_tree = leaves_to_tree(leaves, qtree.variables, qtree.actions)
            prune_tree.meta = qtree.meta

        results.append([prune_tree.size, p.time])
        dump_json(prune_tree, f'{store_path}/dt_prune_{sample_size}.json')

    return np.array(results)
