import os
import re
import subprocess
import numpy as np

from trees.models import QTree
from trees.advanced import max_parts, minimize_tree, leaves_to_tree
from trees.utils import parse_from_sampling_log, performance, \
    convert_uppaal_samples


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
