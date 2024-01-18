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


def viper_exp(env_id, oracle, variables, actions, env_kwargs):
    policy = viper(oracle, env_id, n_iter=10, env_kwargs=env_kwargs)
    loader = SklearnLoader(policy.tree, variables, actions)
    tree = DecisionTree(loader.root, loader.variables, loader.actions)

    # viper_wrapped = ShieldOracleWrapper(viper_tree, amap, shield.n_actions)
    perf, conf, info = my_evaluate_policy(
        SB3Wrapper(tree),
        gym.make(env_id, **env_kwargs),
        n_eval_episodes=1000,
    )

    return tree, (perf, conf), info


def make_wrapped_shield(sdata, store_path, try_load=False):
    if try_load:
        try:
            shield = DecisionTree.load_from_file(store_path)
        except FileNotFoundError:
            pass
    else:
        tree = DecisionTree.from_grid(
            sdata.grid,
            sdata.variables,
            np.arange(sdata.n_actions),
            sdata.granularity,
            np.array(sdata.bounds).T,
            bvars=sdata.bvars
        )

        print('minimize tree...\n')
        shield, (data, best_i) = minimize_tree(tree, max_iter=20)
        shield.save_as(store_path)

    # build action map (what shield action corresponds to which allowed
    # policy actions?)
    amap = [0] * len(sdata.id_to_actionset)
    for i, acts in sdata.id_to_actionset.items():
        amap[int(i)] = tuple(np.argwhere(acts).T[0])

    return ShieldOracleWrapper(shield, amap, sdata.n_actions)


def shield_experiment(model_dir):
    results = []

    print('load shield..\n')
    sdata = unpack_shield(model_dir + 'shields/synthesized.zip')

    variables, actions = sdata.variables, np.arange(sdata.n_actions)
    env_id, env_kwargs = sdata.env_id, sdata.env_kwargs
    org_size = np.product(sdata.grid.shape)

    print('build tree..\n')
    store_path = model_dir + 'shields/mp_shield.json'
    wrapped_shield = make_wrapped_shield(sdata, store_path, try_load=True)

    print('run viper...\n')
    viper_tree, _, info = viper_exp(
        env_id, wrapped_shield, variables, actions, {}
    )

    mp_deaths = strategy_is_safe(env_id, wrapped_shield, env_kwargs=env_kwargs)
    mp_is_safe = mp_deaths == 0

    if (info['deaths'] > 0).sum() == 0:
        import ipdb; ipdb.set_trace()

    print('\nepisode deaths: ', (info['deaths'] > 0).sum(), '\n')

    print(f'MP shield is{" not " if not mp_is_safe else " "}safe')
    print(f'VIPER shield is{" not " if not info["deaths"].sum() == 0 else " "}safe')

    # ADD TO RESULTS
    results = [
        org_size,
        wrapped_shield.shield.n_leaves,
        mp_is_safe,
        mp_deaths,
        viper_tree.n_leaves,
        info['deaths'].sum() == 0,
        info['deaths'].sum(),

    ]
    return results, wrapped_shield, sdata


def cont_exp(wrapped_shield, model_dir, sdata):
    stree = wrapped_shield.shield

    strat_name = 'shielded_strategy.json'
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

    qtree = QTree(oracle_path)
    tree = qtree.to_decision_tree()
    ntree, (data, best_i) = minimize_tree(tree, max_iter=20, verbose=False)

    v_tree, (v_perf, v_conf), v_info = viper_exp(
        sdata.env_id,
        SB3Wrapper(qtree),
        stree.variables,
        stree.actions,
        {}
    )
    mp_perf, mp_conf, mp_info = my_evaluate_policy(
        SB3Wrapper(ntree),
        gym.make(sdata.env_id),
    )

    mp_deaths = mp_info['deaths'].sum()
    mp_safe = mp_deaths == 0

    v_deaths = v_info['deaths'].sum()
    v_safe = v_deaths == 0

    results = [
        tree.n_leaves,

        ntree.n_leaves,
        mp_perf,
        mp_conf,
        mp_safe,
        mp_deaths,

        v_tree.n_leaves,
        v_perf,
        v_conf,
        v_safe,
        v_deaths
    ]

    return results, v_tree


def combined_exp(strategy, wrapped_shield, sdata):
    shielded_strat = ShieldedTree(
        strategy,
        wrapped_shield.shield,
        wrapped_shield.act_map
    )

    evil_perf, evil_conf, evil_info = my_evaluate_policy(
        SB3Wrapper(shielded_strat),
        gym.make(sdata.env_id, unlucky=True),
        inspect=True
    )
    print(f'evil corrections: {shielded_strat.corrections}')

    norm_perf, norm_conf, norm_info = my_evaluate_policy(
        SB3Wrapper(shielded_strat),
        gym.make(sdata.env_id)
    )
    print(f'norm corrections: {shielded_strat.corrections}')

    evil_deaths = evil_info['deaths'].sum()
    evil_safe = evil_deaths == 0

    norm_deaths = norm_info['deaths'].sum()
    norm_safe = norm_deaths == 0

    return [
        evil_perf, evil_conf, evil_deaths, evil_safe,
        norm_perf, norm_conf, norm_deaths, norm_safe
    ]


def controller_experiment(model_dir):
    results = []

    print('load shield..\n')
    shield_dir = model_dir + 'shields/'
    shield = unpack_shield(shield_dir + 'synthesized.zip')

    variables = shield.variables
    actions = np.arange(shield.n_actions)
    env_id = shield.env_id
    env_kwargs = shield.env_kwargs

    print('load shield tree..\n')
    stree = DecisionTree.load_from_file(shield_dir + 'mp_shield.json')

    # build action map (what shield action corresponds to which allowed
    # policy actions?)
    amap = [0]*len(shield.id_to_actionset)
    for i, acts in shield.id_to_actionset.items():
        amap[int(i)] = tuple(np.argwhere(acts).T[0])

    strat_name = 'new_strategy.json'
    if pathlib.Path(model_dir + strat_name).is_file():
        oracle_path = model_dir + strat_name
    else:
        # export shield to c and train a controller in UPPAAL
        print('train in uppaal...\n')
        params = ', '.join([f'double {v}' for v in stree.variables])
        c_path = shield_dir + 'shield.c'
        stree.export_to_c_code(signature=f'shield({params})', out_fp=c_path)
        compile_shield(shield_dir, 'shield')
        oracle_path = train_uppaal_controller(
            model_dir, out_name=strat_name
        )

    print('minimize with MaxPartitions...\n')
    # minimize strategy with MaxParts and VIPER, respectively
    qtree = QTree(oracle_path)
    tree = qtree.to_decision_tree()
    ntree, (data, best_i) = minimize_tree(tree, max_iter=20, verbose=False)

    print('minimize with VIPER..\n')
    # viper_pol = viper(SB3Wrapper(qtree), env_id, n_iter=10, env_kwargs=env_kwargs)
    viper_pol = viper(SB3Wrapper(qtree), env_id, n_iter=3)
    loader = SklearnLoader(viper_pol.tree, variables, actions)
    viper_tree = DecisionTree(loader.root, loader.variables, loader.actions)

    # CHECK SAFETY OF STRATEGIES
    # mp_deaths = strategy_is_safe(env_id, SB3Wrapper(ntree), env_kwargs=env_kwargs)
    # mp_deaths = strategy_is_safe(env_id, SB3Wrapper(ntree), inspect=None, env_kwargs=env_kwargs)
    # mp_is_safe = mp_deaths == 0

    # viper_deaths = strategy_is_safe(env_id, SB3Wrapper(viper_tree), env_kwargs=env_kwargs)
    # viper_deaths = strategy_is_safe(env_id, SB3Wrapper(viper_tree), env_kwargs=env_kwargs)
    # viper_is_safe = viper_deaths == 0

    mp_perf, mp_std, mp_info = my_evaluate_policy(SB3Wrapper(ntree), gym.make(env_id), 500)
    viper_perf, viper_std, viper_info = my_evaluate_policy(SB3Wrapper(viper_tree), gym.make(env_id), 500)

    mp_deaths = mp_info['deaths'].sum()
    viper_deaths = viper_info['deaths'].sum()
    mp_is_safe = mp_deaths == 0
    viper_is_safe = viper_deaths == 0

    print(f'MP controller is{" not " if not mp_is_safe else " "}safe')
    print(f'VIPER controller is{" not " if not viper_is_safe else " "}safe')

    # ADD TO RESULTS
    results.append(tree.n_leaves)
    results.append(ntree.n_leaves)
    results.append(int(mp_is_safe))
    results.append(mp_deaths)
    results.append(mp_perf)
    results.append(mp_std)

    results.append(viper_tree.n_leaves)
    results.append(int(viper_is_safe))
    results.append(viper_deaths)
    results.append(viper_perf)
    results.append(viper_std)

    return results


def old_shield_experiment(model_dir):

    results = []

    print('load shield..\n')
    shield_dir = model_dir + 'shields/'
    shield_path = shield_dir + 'synthesized.zip'
    shield = unpack_shield(shield_path)

    variables = shield.variables
    actions = np.arange(shield.n_actions)
    env_id = shield.env_id
    env_kwargs = shield.env_kwargs
    bvars = shield.bvars

    meta_data = dict(
        env_id=env_id,
        variables=variables,
        n_actions=shield.n_actions,
        env_kwargs=env_kwargs,
        granularity=shield.granularity,
        dimensions=shield.grid.shape,
        columns=[
            'shield_org_n_leaves',
            # 'shield_org_sizeof', 'shield_org_predict_time',
            'shield_mp_n_leaves',
            # 'shield_mp_iterations', 'shield_mp_sizeof', 'shield_mp_predict_time',
            'shield_mp_safe', 'shield_mp_deaths',

            'shield_viper_n_leaves',
            # 'shield_viper_sizeof', 'shield_viper_predict_time',
            'shield_viper_safe', 'shield_viper_deaths'

            # 'strat_org_n_leaves', 'strat_org_sizeof', 'strat_org_predict_time',
            # 'strat_mp_n_leaves', 'strat_mp_sizeof',
            # 'strat_mp_predict_time', 'strat_mp_safe',
            # 'strat_viper_n_leaves', 'strat_viper_sizeof',
            # 'strat_viper_predict_time', 'strat_viper_safe'
        ]
    )

    print('build tree..\n')
    tree = DecisionTree.from_grid(
        shield.grid,
        variables,
        actions,
        shield.granularity,
        np.array(shield.bounds).T
    )
    print('minimize tree...\n')
    ntree, (data, best_i) = minimize_tree(tree, max_iter=20)
    ntree.save_as(shield_dir + '/mp_shield.json')

    # build action map (what shield action corresponds to which allowed
    # policy actions?)
    amap = [0]*len(shield.id_to_actionset)
    for i, acts in shield.id_to_actionset.items():
        amap[int(i)] = tuple(np.argwhere(acts).T[0])

    # let viper take a shot at minimizing shield
    print('run viper...\n')
    wrapped = ShieldOracleWrapper(ntree, amap, shield.n_actions)
    viper_pol = viper(wrapped, env_id, n_iter=50, env_kwargs=env_kwargs)

    # convert viper policy to a tree
    loader = SklearnLoader(viper_pol.tree, variables, actions)
    viper_tree = DecisionTree(loader.root, loader.variables, loader.actions)


    # CHECK SAFETY OF MINIMIZED SHIELDS
    mp_deaths = strategy_is_safe(env_id, wrapped, env_kwargs=env_kwargs)
    mp_is_safe = mp_deaths == 0
    print(f'MP shield is{" not " if not mp_is_safe else " "}safe')

    viper_wrapped = ShieldOracleWrapper(viper_tree, amap, shield.n_actions)
    viper_deaths = strategy_is_safe(
        env_id,
        viper_wrapped,
        n_episodes=1000,
        env_kwargs=env_kwargs
    )
    viper_is_safe = viper_deaths == 0
    print(f'VIPER shield is{" not " if not viper_is_safe else " "}safe')


    # ADD TO RESULTS
    results.append(tree.n_leaves)
    # results.append(estimate_sizeof(tree.root))
    # results.append(time_predict(tree, shield.bounds))

    results.append(ntree.n_leaves)
    # results.append(best_i + 1)
    # results.append(estimate_sizeof(ntree.root))
    # results.append(time_predict(ntree, shield.bounds))
    results.append(int(mp_is_safe))
    results.append(mp_deaths)

    results.append(viper_tree.n_leaves)
    # results.append(estimate_sizeof(viper_tree.root))
    # results.append(time_predict(viper_tree, shield.bounds))
    results.append(int(viper_is_safe))
    results.append(viper_deaths)

    return meta_data, results
