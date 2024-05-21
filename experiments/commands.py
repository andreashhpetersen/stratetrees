import pathlib
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import gymnasium as gym
import uppaal_gym

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env

from viper.viper import viper
from viper.wrappers import ShieldOracleWrapper, SB3Wrapper

from glob import glob
from tqdm import tqdm

from experiments.utils import run_single_experiment, run_uppaal, \
    parse_uppaal_results, get_mean_and_std_matrix, compile_shield, \
    unpack_shield, train_uppaal_controller, strategy_is_safe, \
    estimate_sizeof, time_predict, my_evaluate_policy

from trees.advanced import minimize_tree
from trees.loaders import SklearnLoader
from trees.models import QTree, DecisionTree
from trees.utils import convert_uppaal_samples
from trees.wrappers import ShieldedTree


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


def make_viper(env_id, oracle, variables, actions, env_kwargs, verbose=1):
    """
    Set `verbose=2' to get VIPER status info
    """

    if verbose:
        print('generating VIPER policy...')
    policy = viper(
        oracle, env_id, n_iter=10, env_kwargs=env_kwargs, verbose=verbose > 1
    )
    import ipdb; ipdb.set_trace()
    loader = SklearnLoader(policy.tree, variables, actions)
    tree = DecisionTree(loader.root, loader.variables, loader.actions)

    return tree


def make_wrapped_shield(sdata, store_path, try_load=False):
    if try_load:
        try:
            shield = DecisionTree.load_from_file(store_path)
        except FileNotFoundError:
            pass
    else:
        print('build tree from shield...')
        tree = DecisionTree.from_grid(
            sdata.grid,
            sdata.variables,
            np.arange(sdata.n_actions),
            sdata.granularity,
            np.array(sdata.bounds).T,
            bvars=sdata.bvars
        )

        print('minimize tree...')
        shield, (data, best_i) = minimize_tree(tree, max_iter=20)
        shield.save_as(store_path)

    # build action map (what shield action corresponds to which allowed
    # policy actions?)
    amap = [0] * len(sdata.id_to_actionset)
    for i, acts in sdata.id_to_actionset.items():
        amap[int(i)] = tuple(np.argwhere(acts).T[0])

    return ShieldOracleWrapper(shield, amap, sdata.n_actions)


def shield_experiment(model_dir, env_kwargs={}, callback=None, verbose=False):
    results = []

    # load shield
    sdata = unpack_shield(model_dir + 'shields/synthesized.zip')

    # construct shield
    wrapped_shield = make_wrapped_shield(
        sdata, model_dir + 'shields/mp_shield.json', try_load=True
    )
    import ipdb; ipdb.set_trace()

    # construct viper shield
    viper_tree = make_viper(
        sdata.env_id, wrapped_shield, sdata.variables,
        np.arange(sdata.n_actions), env_kwargs, verbose=verbose
    )
    import ipdb; ipdb.set_trace()

    # evaluate viper
    _, _, v_info = my_evaluate_policy(
        SB3Wrapper(viper_tree),
        gym.make(sdata.env_id, **env_kwargs),
        n_eval_episodes=1000,
        callback=callback
    )

    # evaluate maxpartitions
    _, _, mp_info = my_evaluate_policy(
        wrapped_shield,
        gym.make(sdata.env_id, **env_kwargs),
        callback=callback
    )

    # store results
    results = [
        np.product(sdata.grid.shape),       # org size
        wrapped_shield.shield.n_leaves,     # maxparts size
        mp_info['deaths'].sum() == 0,       # maxparts safe?
        (mp_info['deaths'] > 0).sum(),      # maxparts unsafe runs
        mp_info['deaths'].sum(),            # maxparts deaths
        viper_tree.n_leaves,                # viper size
        v_info['deaths'].sum() == 0,        # viper safe?
        (v_info['deaths'] > 0).sum(),       # viper unsafe runs
        v_info['deaths'].sum(),             # viper deaths

    ]
    return results, wrapped_shield, sdata


def cont_exp(
    wrapped_shield, model_dir, sdata, strat_name,
    env_kwargs={}, callback=None, verbose=False
):

    # shield as tree
    stree = wrapped_shield.shield

    # get path to oracle strategy from UPPAAL
    if pathlib.Path(model_dir + strat_name).is_file():
        oracle_path = model_dir + strat_name
    else:
        print('train in uppaal...\n')
        params = ', '.join([f'double {v}' for v in stree.variables])
        c_path = model_dir + '/shields/shield.c'
        stree.export_to_c_code(signature=f'shield({params})', out_fp=c_path)
        compile_shield(model_dir + '/shields/', 'shield')
        oracle_path = train_uppaal_controller(
            model_dir, out_name=strat_name
        )

    # convert UPPAAL oracle to decision tree
    qtree = QTree(oracle_path)
    tree = qtree.to_decision_tree()

    # minimize with maxpartitions
    ntree, (data, best_i) = minimize_tree(tree, max_iter=20, verbose=False)

    # minimize with VIPER
    v_tree = make_viper(
        sdata.env_id, SB3Wrapper(qtree),
        stree.variables, stree.actions, env_kwargs, verbose=verbose
    )

    # evaluate VIPER
    v_perf, v_conf, v_info = my_evaluate_policy(
        SB3Wrapper(v_tree),
        gym.make(sdata.env_id, **env_kwargs),
        n_eval_episodes=1000,
        callback=callback
    )

    # evaluate maxpartitions
    mp_perf, mp_conf, mp_info = my_evaluate_policy(
        SB3Wrapper(ntree),
        gym.make(sdata.env_id),
        callback=callback
    )

    # store results
    results = [
        tree.n_leaves,                  # size of oracle

        ntree.n_leaves,                 # maxparts size
        (mp_info['deaths'] > 0).sum(),  # maxparts unsafe runs
        mp_info['deaths'].sum(),        # maxparts violations
        mp_perf,                        # maxparts performance
        mp_conf,                        # maxparts confidence

        v_tree.n_leaves,                # viper size
        (v_info['deaths'] > 0).sum(),   # viper unsafe runs
        v_info['deaths'].sum(),         # viper violations
        v_perf,                         # viper performance
        v_conf,                         # viper confidence
    ]

    return results, v_tree


def combined_exp(strategy, wrapped_shield, sdata):

    # combine strategy and shield
    shielded_strat = ShieldedTree(
        strategy, wrapped_shield.shield, wrapped_shield.act_map
    )

    # evaluate on antagonistic environment
    evil_perf, evil_conf, evil_info = my_evaluate_policy(
        SB3Wrapper(shielded_strat), gym.make(sdata.env_id, unlucky=True),
    )

    # store corrections and reset
    evil_corrections = shielded_strat.corrections
    shielded_strat.corrections = 0

    # evaluate on normal environment
    norm_perf, norm_conf, norm_info = my_evaluate_policy(
        SB3Wrapper(shielded_strat), gym.make(sdata.env_id)
    )
    norm_corrections = shielded_strat.corrections

    return [
        evil_perf, evil_conf, evil_corrections, evil_info['deaths'].sum(),
        norm_perf, norm_conf, norm_corrections, norm_info['deaths'].sum(),
    ]
