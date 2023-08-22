import os
import re
import json
import glob
import pathlib
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from glob import glob
from tqdm import tqdm

from trees.advanced import max_parts, minimize_tree, leaves_to_tree
from trees.models import QTree, DecisionTree
from trees.utils import parse_from_sampling_log, performance


def dump_json(tree, fp):
    path = fp.split('/')
    tree_path = path[:-1] + ['trees'] + path[-1:]
    uppaal_path = path[:-1] + ['uppaal'] + path[-1:]

    tree.save_as('/'.join(tree_path))
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
            try:
                assert line == ' -- Formula is satisfied.'
            except AssertionError:
                import ipdb; ipdb.set_trace()

            check_next = False

        if line.startswith('Verifying'):
            check_next = True

        match = re.match(eval_ptr, line)
        if match:
            exp = float(match.groups()[0])
            std = float(match.groups()[1])
            return [exp, std]


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


def make_samples(model_dir, clear_first=True):
    if clear_first:
        for p in pathlib.Path(f'{model_dir}/samples/').glob('*.log'):
            p.unlink()

    args = (f'{model_dir}/make_samples.sh',)
    subprocess.run(args)


def get_mean_and_std_matrix(data):
    data = np.array(data)

    _, n_models, n_metrics = data.shape

    exps = data.T.mean(axis=2).T.reshape(-1, n_metrics, 1)
    stds = data.T.std(axis=2, ddof=1).T.reshape(-1, n_metrics, 1)

    return np.dstack((exps, stds)).reshape(n_models, -1)


def main(model_dir, k=10, early_stopping=False):
    qt_strat_file = f'{model_dir}/qt_strategy.json'
    sample_logs = glob(f'{model_dir}/samples/*_converted.log')

    results_columns = [
        'leaves', 'max depth', 'min depth',
        'construction time', 'performance', 'performance std'
    ]
    mean_results_columns = [
        'leaves', 'leaves (std)',
        'max depth', 'max depth (std)',
        'min depth', 'min depth (std)',
        'construction time', 'construction time (std)'
    ]
    maxparts_columns = [
        'regions found', 'regions found (std)',
        'leaves in tree', 'leaves in tree (std)',
        'max depth', 'max depth (std)',
        'min depth', 'min depth (std)',
        'construction time', 'construction time (std)'
    ]

    results, model_names = [], []

    print('\nEvaluate original Qtree-strategy')
    qtree = QTree(qt_strat_file)
    uppaal_res = parse_uppaal_results(run_uppaal(model_dir, 'qt_strategy.json'))

    results.append(
        [qtree.n_leaves, -1, -1, -1, uppaal_res[0], uppaal_res[1]]
    )
    model_names.append('qt_strategy')

    all_res, all_mp_data= [], []
    best_trees, res_i = [], 0
    smallest = np.inf

    print(f'\nGenerate minimized trees (best of {k} attempts)')
    for i in tqdm(range(k)):
        trees, res, mp_data = run_single_experiment(
            qtree,
            sample_logs,
            early_stopping=early_stopping
        )

        all_res.append(res)
        all_mp_data.append(mp_data)

        # if the originally constructed dt is smaller, save output trees as best
        if res[0,0] < smallest:
            smallest = res[0,0]
            best_trees = trees
            best_res = res

    store_path = lambda x: f'{model_dir}/generated/{x}/'

    pathlib.Path(store_path('trees')).mkdir(parents=True, exist_ok=True)
    pathlib.Path(store_path('uppaal')).mkdir(parents=True, exist_ok=True)
    pathlib.Path(store_path('results')).mkdir(parents=True, exist_ok=True)

    print(f'\nEvaluate minimized trees')
    for (model, tree), res in zip(tqdm(best_trees), best_res):
        model_names.append(model)
        filename = f'{model}.json'

        tree.save_as(f'{store_path("trees")}{filename}')
        tree.export_to_uppaal(f'{store_path("uppaal")}{filename}')

        evaluation = parse_uppaal_results(
            run_uppaal(model_dir, f'generated/uppaal/{filename}')
        )
        results.append(np.concatenate((res, evaluation)))

    if early_stopping:
        maxlen = max([len(d) for d in all_mp_data])
        all_mp_data = [d[:maxlen] for d in all_mp_data]

    all_mp = get_mean_and_std_matrix(all_mp_data)
    all_res = get_mean_and_std_matrix(all_res)

    # export results
    print(f'\nSaving results to {model_dir}/generated/')
    results_df = pd.DataFrame(
        np.array(results), index=model_names, columns=results_columns
    )
    results_df.to_csv(f'{store_path("results")}results.csv')

    mean_results_df = pd.DataFrame(
        all_res, index=model_names[1:], columns=mean_results_columns
    )
    mean_results_df.to_csv(f'{store_path("results")}mean_results.csv')

    maxparts_df = pd.DataFrame(
        all_mp, index=range(len(all_mp)), columns=maxparts_columns
    )
    maxparts_df.to_csv(f'{store_path("results")}maxparts_results.csv')

    regs, leaves = 'regions found', 'leaves in tree'
    maxparts_df[[regs, leaves]].plot(color=['#CC4F1B', '#1B2ACC'])
    plt.fill_between(
        range(len(maxparts_df)),
        maxparts_df[regs] - maxparts_df[regs + ' (std)'],
        maxparts_df[regs] + maxparts_df[regs + ' (std)'],
        facecolor='#FF9848'
    )
    plt.fill_between(
        range(len(maxparts_df)),
        maxparts_df[leaves] - maxparts_df[leaves + ' (std)'],
        maxparts_df[leaves] + maxparts_df[leaves + ' (std)'],
        facecolor='#089FFF'
    )
    ticks = np.linspace(
        0, len(maxparts_df),
        int(np.ceil((len(maxparts_df) + 1) / 2))
    )
    plt.xticks(
        ticks=ticks, labels=np.int_(ticks+1)
    )

    plt.xlabel('# maxpartition episodes')
    plt.ylabel('# leaves/regions')

    plt.savefig(f'{store_path("results")}maxparts_plot.png')
