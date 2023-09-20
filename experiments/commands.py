import pathlib
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from glob import glob
from tqdm import tqdm

from experiments.utils import run_single_experiment, run_uppaal, \
    parse_uppaal_results, get_mean_and_std_matrix
from trees.models import QTree, DecisionTree
from trees.utils import convert_uppaal_samples


def make_samples(model_dir, clear_first=True):
    samples_path = f'{model_dir}/samples/'
    if clear_first:
        for p in pathlib.Path(samples_path).glob('*.log'):
            p.unlink()

    args = (f'{model_dir}/make_samples.sh',)
    subprocess.run(args)

    for file in glob(f'{samples_path}*_uppaal.log'):
        convert_uppaal_samples(file)


def run_experiments(model_dir, k=10, early_stopping=False):
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
