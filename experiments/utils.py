import os
import re
import json
import shutil
import pathlib
import tempfile
import subprocess
import numpy as np

import gymnasium as gym
import uppaal_gym

from trees.models import QTree
from trees.advanced import max_parts, minimize_tree, leaves_to_tree
from trees.utils import parse_from_sampling_log, performance, \
    convert_uppaal_samples


class Shield:
    def __init__(self, grid, meta):
        self.grid = grid
        self.variables = meta['variables']
        self.env_kwargs = meta['env_kwargs']
        self.bounds = meta['bounds']
        self.granularity = meta['granularity']
        self.n_actions = meta['n_actions']
        self.id_to_actionset = meta['id_to_actionset']
        self.env_id = meta['env_id']


def unpack_shield(path):
    with tempfile.TemporaryDirectory() as tmpdirname:
        shutil.unpack_archive(path, tmpdirname)
        with open(tmpdirname + '/meta.json', 'r') as f:
            shield_meta = json.load(f)

        grid = np.load(tmpdirname + '/grid.npy')

    return Shield(grid, shield_meta)


def get_mean_and_std_matrix(data):
    data = np.array(data)

    _, n_models, n_metrics = data.shape

    exps = data.T.mean(axis=2).T.reshape(-1, n_metrics, 1)
    stds = data.T.std(axis=2, ddof=1).T.reshape(-1, n_metrics, 1)

    return np.dstack((exps, stds)).reshape(n_models, -1)


def run_single_experiment(
        qtree: QTree, sample_logs: list[str], early_stopping=False
    ) -> np.ndarray:
    results = []
    trees = []

    with performance() as p:
        tree = qtree.to_decision_tree()

    results.append([tree.n_leaves, tree.max_depth, tree.min_depth, p.time])
    trees.append(('dt_original', tree))

    mp_tree, (data, best) = minimize_tree(
        tree,
        max_iter=10,
        verbose=False,
        early_stopping=early_stopping
    )

    results.append(
        [data[best,1], data[best,2], data[best,3], data[:best+1].sum()]
    )
    trees.append(('mp_tree', mp_tree))

    mp_tree.meta = qtree.meta

    for sample_log in sample_logs:
        try:
            from_uppaal = not sample_log.endswith('_converted.log')
            samples = parse_from_sampling_log(sample_log, from_uppaal=from_uppaal)
        except IndexError:
            print(f"'{sample_log}' is not in the correct format, skipping...")
            continue

        sample_size = int(re.findall(r'\d+', sample_log)[0])

        prune_tree = mp_tree.copy()
        with performance() as p:
            prune_tree.count_visits(samples)
            prune_tree.emp_prune()
            leaves = max_parts(prune_tree)
            prune_tree = leaves_to_tree(leaves, qtree.variables, qtree.actions)
            prune_tree.meta = qtree.meta

        trees.append((f'empprune_{sample_size}', prune_tree))
        results.append([
            prune_tree.n_leaves,
            prune_tree.max_depth,
            prune_tree.min_depth,
            p.time
        ])

    return trees, np.array(results), data


def parse_uppaal_results(output):
    plain_data = output.split('\n')

    model_ptr = r'([\w]+).json'
    eval_ptr = r'^\([\d]+ runs\) E\([\w]+\) = ([\d]+.[\d]+) ± ([\d]+.[\d]+)'

    check_next = False
    for line in plain_data:
        if check_next:
            assert line == ' -- Formula is satisfied.'

            check_next = False

        if line.startswith('Verifying'):
            check_next = True

        match = re.match(eval_ptr, line)
        if match:
            exp = float(match.groups()[0])
            std = float(match.groups()[1])
            return [exp, std]
        elif '≈ 0' in line:
            return [0., 0.]


def train_uppaal_controller(model_dir, out_name='strategy.json'):
    out_path = pathlib.Path(model_dir, out_name)

    query_file = subprocess.run(
        ( f'{model_dir}/make_training_query.sh', out_path ),
        capture_output=True,
        encoding='utf-8'
    )

    evaluation_args = (
        os.environ['VERIFYTA_PATH'],
        f'{model_dir}/shielded_model.xml',
        query_file.stdout.strip('\n'),
        '--seed',
        str(np.random.randint(0, 10000)),
        '-sWqy'
    )
    subprocess.run(evaluation_args, capture_output=True)

    return out_path


def run_uppaal(model_dir, strategy_name):
    eval_query_args = (
        f'{model_dir}/make_eval_query.sh',
        f'{model_dir}/{strategy_name}'
    )
    eval_query = subprocess.run(
        eval_query_args,
        capture_output=True,
        encoding='utf-8'
    )

    evaluation_args = (
        os.environ['VERIFYTA_PATH'],
        f'{model_dir}/model.xml',
        eval_query.stdout.strip('\n'),
        '--seed',
        str(np.random.randint(0, 10000)),
        '-sWqy'
    )
    evaluation = subprocess.run(
        evaluation_args,
        capture_output=True,
        encoding='utf-8'
    )

    return evaluation.stdout


def compile_shield(path, name):
    path = f'{path}/{name}'
    # make object
    args = (
        'gcc',
        '-c',
        '-fPIC',
        f'{path}.c',
        '-o',
        f'{path}.o'
    )
    subprocess.run(args)

    # make shared object
    args = (
        'gcc',
        '-shared',
        '-o',
        f'{path}.so',
        f'{path}.o'
    )
    subprocess.run(args)


def strategy_is_safe(env_id, strategy, n_trajectories=100, env_kwargs={}):
    env = gym.make(env_id, **env_kwargs)

    for iter in range(n_trajectories):
        obs, _ = env.reset()
        observations = [obs]
        actions = []

        terminated = False
        while not terminated:
            action, _ = strategy.predict(obs)
            if isinstance(action, list):
                action = action[0]
            actions.append(action)
            nobs, reward, terminated, _, _ = env.step(action)
            observations.append(nobs)
            if terminated and not env.unwrapped.is_safe(nobs):
                return False
            obs = nobs

    return True
