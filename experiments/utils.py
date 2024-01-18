import os
import re
import sys
import json
import shutil
import pathlib
import tempfile
import subprocess
import numpy as np

import matplotlib.pyplot as plt
import gymnasium as gym
import uppaal_gym

from trees.models import QTree
from trees.nodes import Node, Leaf
from trees.advanced import max_parts, minimize_tree, leaves_to_tree
from trees.utils import parse_from_sampling_log, performance, \
    convert_uppaal_samples, time_it


class Shield:
    def __init__(self, grid, meta):
        self.grid = grid
        self.variables = meta['variables']
        self.env_kwargs = meta.get('env_kwargs', {})
        self.bounds = meta['bounds']
        self.granularity = meta['granularity']
        self.n_actions = meta['n_actions']
        self.id_to_actionset = meta['id_to_actionset']
        self.env_id = meta['env_id']
        self.bvars = meta['bvars']


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


def strategy_is_safe(env_id, strategy, n_episodes=100, env_kwargs={}, inspect=False):
    env = gym.make(env_id, **env_kwargs)

    deaths = 0
    for episode in range(n_episodes):
        obs, _ = env.reset()
        observations = [obs]
        actions = []
        # shield_acts = []
        rewards = [-1]
        # shield_actions = [-1]

        i = 0
        terminated, trunc = False, False
        while not (terminated or trunc):
            # shield_act = strategy.shield.predict(obs)
            # if shield_act != 2:
            #     action = 1
            # else:
            #     action = 0
            action, _ = strategy.predict(obs)
            if isinstance(action, list):
                action = action[0]
            # shield_acts.append(shield_act)
            actions.append(action)
            nobs, reward, terminated, trunc, _ = env.step(action)
            rewards.append(reward)
            observations.append(nobs)
            if terminated or trunc: #and not env.unwrapped.is_safe(nobs):
                # observations = np.array(observations)
                # shield_acts.append(99)
                # actions.append(99)
                # actions = np.array([actions]).T
                # shield_acts = np.array([shield_acts]).T
                # info = np.hstack((observations, actions, shield_acts))
                # info = np.hstack((observations, actions))
                rewards = np.array(rewards)
                if not env.unwrapped.is_safe(nobs):
                    deaths += 1

                if inspect:
                    import ipdb; ipdb.set_trace()
                #     return False, info
                # else:
                #     return False
            obs = nobs
            i += 1

    # return (True, []) if inspect else True
    return deaths


def confidence(sample, z=1.96):
    return z * (sample.std() / np.sqrt(len(sample)))


def my_evaluate_policy(policy, env, n_eval_episodes=100, inspect=False):
    all_rews = []
    all_obs = []
    all_deaths = []
    for episode in range(n_eval_episodes):
        obs, _ = env.reset()
        ep_rews, ep_obs, ep_deaths = [], [obs], 0

        terminated, trunc = False, False
        while not (terminated or trunc):
            action, _ = policy.predict(obs)
            if isinstance(action, list):
                action = action[0]
            nobs, reward, terminated, trunc, _ = env.step(action)
            ep_rews.append(reward)
            ep_obs.append(nobs)
            obs = nobs

            # if not env.unwrapped.is_safe(obs):
            if terminated and len(ep_obs) < 400:
                ep_deaths += 1
                env.unwrapped.p = 8 + np.random.uniform(0,2)
                env.unwrapped.v = 0
                # env.unwrapped.x1 = 0.35
                # env.unwrapped.v = 15.0
                terminated = False
                if inspect:
                    import ipdb; ipdb.set_trace()

        all_rews.append(np.sum(ep_rews))
        all_obs.append(np.vstack(ep_obs))
        all_deaths.append(ep_deaths)

    rews = np.array(all_rews)
    obs = all_obs
    deaths = np.array(all_deaths)
    info = {
        'observations': obs,
        'deaths': deaths ,
        'rewards': rews
    }
    return rews.mean(), confidence(rews), info


def estimate_sizeof(node):
    node_handler = lambda n: [n.var_id, n.bound, n.low, n.high]
    leaf_handler = lambda l: [l.action]

    handlers = { Node: node_handler, Leaf: leaf_handler }

    seen = set()
    default_size = sys.getsizeof(0)

    def sizeof(o):
        if id(o) in seen:
            return 0
        seen.add(id(o))
        s = sys.getsizeof(o, default_size)

        for typ, handler in handlers.items():
            if isinstance(o, typ):
                s += sum(map(sizeof, handler(o)))
                break
        return s

    return sizeof(node)


def time_predict(tree, bounds, n=100_000):
    dims = len(bounds[0])
    sample = np.random.uniform(*bounds, size=(n, dims))
    _, tm = time_it(lambda t, xs: len([t.predict(s) for s in xs]), tree, sample)
    return tm
