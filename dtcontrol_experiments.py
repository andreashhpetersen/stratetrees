import os
import csv
import json
import pickle
import pathlib
import numpy as np

import gymnasium as gym
import uppaal_gym

from glob import glob

from dtcontrol.benchmark_suite import BenchmarkSuite
from dtcontrol.decision_tree.decision_tree import DecisionTree as DtcDecisionTree
from dtcontrol.decision_tree.impurity.entropy import Entropy
from dtcontrol.decision_tree.splitting.axis_aligned import AxisAlignedSplittingStrategy

from trees.advanced import minimize_tree
from trees.models import DecisionTree
from trees.nodes import Leaf, Node, State
from trees.utils import cut_overlaps

from experiments.utils import unpack_shield, my_evaluate_policy, bb_callback, op_callback

from viper.wrappers import ShieldOracleWrapper


def get_state_action_pairs(samples, tree, amap=None, NO_ACTIONS=9, zipped=True):
    """
    Get a (zipped) tuple of state/action pairs generated by predicting an action
    in `tree` for every entry in `samples`. If an action map `amap` is provided,
    the decision of the `tree` is used as a key to get a set of actions from
    `amap`, and a separate state/action pair will be added for each of these
    actions (this is used when the input tree is a shield).
    """
    states, actions = [], []
    for state in samples:
        decision = tree.predict(state)

        # no action map, just add the decision of the tree
        if amap is None:
            states.append(state)
            actions.append(decision)

        else:
            action_set = amap[decision]

            # no action is prescribed by the tree, add special NO_ACTIONS
            if len(action_set) == 0:
                states.append(state)
                actions.append(NO_ACTIONS)

            else:
                # add a line for each allowed action in this state
                for action in action_set:
                    states.append(state)
                    actions.append(action)

    # convert to numpy arrays
    states = np.array(states)
    actions = np.array(actions)

    # make actions conform to expected shape
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, axis=1)

    # return iterator or tuple
    return zip(states, actions) if zipped else (states, actions)


def write_dtcontrol_csv(
    name, pairs, nvars, nacts=1,
    config_dict=None, dataset_folder='./dtcontrol/clean_examples/'
):
    path = pathlib.Path(dataset_folder + name)

    with open(path, 'w') as f:
        f.write('#PERMISSIVE\n')
        f.write(f'#BEGIN {nvars} {nacts}\n')
        for state, actions in pairs:
            try:
                f.write('{},{}\n'.format(
                    ','.join(map(str, state)),
                    ','.join(map(str, actions))
                ))
            except TypeError:
                import ipdb; ipdb.set_trace()


    if config_dict is not None:
        conf_path = pathlib.Path(
            dataset_folder + name.replace('.csv', '_config.json')
        )
        with open(conf_path, 'w') as f:
            json.dump(config_dict, f)


def run_dtcontrol_suite(data_dir, include=None):
    # instantiate the benchmark suite with a timeout of 2 hours
    # rest of the parameters behave like in CLI
    suite = BenchmarkSuite(
        timeout=60*60*2,
        save_folder='./dtcontrol/experiments',
        benchmark_file='./dtcontrol/experiments/benchmark',
        rerun=True
    )

    # Add the 'examples' directory as the base where
    # the different controllers will be searched for
    # You can also choose to only include specific files
    # in the directory with the 'include' and 'exclude' list
    suite.add_datasets(data_dir, include=include)

    # select the DT learning algorithms we want to run and give them names
    classifier = DtcDecisionTree([AxisAlignedSplittingStrategy()], Entropy(), 'CART')

    # finally, execute the benchmark
    suite.benchmark([classifier])


def parse_json(data, vmap, action_map):
    if data['split'] is None:
        acts = data['actual_label']

        try:
            action = action_map[tuple(acts)]
        except KeyError:
            import ipdb; ipdb.set_trace()

        # replace with map parameter
        # if len(acts) > 1:
        #     action = 3
        # elif acts[0] == '2':
        #     action = 0
        # elif acts[0] == '1':
        #     action = 1
        # else:
        #     action = 2
        return Leaf(action)

    low = parse_json(data['children'][0], vmap, action_map)
    high = parse_json(data['children'][1], vmap, action_map)
    varname = data['split']['lhs']['var']
    return Node(varname, vmap[varname], data['split']['rhs'], low, high)


def import_dtcontrol_tree(path, variables, actions, action_map):
    with open(path, 'rb') as f:
        dt = pickle.load(f)

    json_data = json.loads(dt.toJSON({'variables': variables}, {}))
    root = parse_json(json_data, { v: a for a, v in enumerate(variables) }, action_map)
    return DecisionTree(root, variables, actions)


def write_evenly_spaced_points(
    base_name, granularities, tree, bounds,
    config_dict=None, amap=None
):
    """
    For each granularity given in `granularities`, generate that many evenly
    spaced points in the space defined by `bounds`. Use `tree` to predict the
    corresponding actions and write to a dtcontrol input csv file.
    """
    K = bounds.shape[0]
    names = []
    for g in granularities:
        name = f'{base_name}_{g}.csv'
        names.append(name.replace('.csv',''))
        samples = np.linspace(*bounds, num=g)
        pairs = get_state_action_pairs(samples, tree, amap)
        write_dtcontrol_csv(name, pairs, K, config_dict=config_dict)

    return names


def write_randomly_sampled_points_from_partitions(
    base_name, samples_per_leaf, tree, bounds,
    config_dict=None, amap=None
):
    """
    For each number `n` in `samples_per_leaf`, sample `n` points from the region
    of every leaf in the tree `tree`.
    """
    K = bounds.T.shape[0]
    leaves = tree.leaves()
    names = []
    for ns in samples_per_leaf:
        name = f'{base_name}_partitions_{ns}.csv'
        names.append(name.replace('.csv',''))

        # init empty samples array
        samples = np.empty((len(leaves) * ns, K))

        i = 0
        for leaf in leaves:
            # get the bounded state that the leaf represents
            state = leaf.state.bounded(*bounds)

            # sample `ns` points from this region randomly and add to samples
            samples[i:i+ns] = np.random.uniform(*state.T, size=(ns, K))

            # increment iterator
            i += ns

        # convert to pairs and write to dtcontrol csv file
        pairs = get_state_action_pairs(samples, tree, amap)
        write_dtcontrol_csv(
            name, pairs, K,
            config_dict=config_dict
        )

    return names


def write_centered_points_from_partitions(
    base_name, tree, bounds, decimals=6, config_dict=None, amap=None
):
    """
    For each leaf in `tree`, write the center point and corresponding actions
    to a csv file in the dtcontrol format.
    """
    K = bounds.T.shape[0]
    low, high = bounds[0], bounds[1]
    name = f'{base_name}_partitions_centered.csv'
    names = [name.replace('.csv','')]

    # init empty samples array
    leaves = tree.leaves()
    samples = np.empty((len(leaves), K))

    for i, leaf in enumerate(leaves):
        # get center and round to `decimals`
        center = np.round(
            leaf.state.center(min_limit=low, max_limit=high), decimals
        )

        # update samples
        samples[i] = center

    # convert to pairs and write to dtcontrol csv file
    pairs = get_state_action_pairs(samples, tree, amap)
    write_dtcontrol_csv(
        name, pairs, K,
        config_dict=config_dict
    )

    return names


def make_data(
    base_name, tree, bounds, amap=None, variables=None,
    granularities=[], samples_per_leaf=[]
):
    """
    Make data for dtcontrol with different heuristics
    """
    config_dict = None if variables is None else { 'column_names': variables }

    names1 = write_evenly_spaced_points(
        base_name, granularities, tree, bounds,
        config_dict=config_dict, amap=amap
    )

    names2 = write_randomly_sampled_points_from_partitions(
        base_name, samples_per_leaf, tree, bounds,
        config_dict=config_dict, amap=amap
    )

    names3 = write_centered_points_from_partitions(
        base_name, tree, bounds,
        config_dict=config_dict, amap=amap
    )
    return names1 + names2 + names3


def get_saved_controllers(base_dir, include=None):
    paths, names = [], []
    directories = list(os.walk(base_dir))[1:]
    directories.sort(key=lambda x: x[0])

    for subdir, _, files in directories:
        name = subdir.split('/')[-1]

        if include is not None and name not in include:
            continue

        names.append(name)
        controller = sorted(files)[-1]
        paths.append(pathlib.Path(subdir + '/' + controller))

    return names, paths


def volume(box):
    if not box.shape[-1] == 2:
        box = box.T
    assert box.shape[-1] == 2
    return np.product(box[:,1] - box[:,0])


def get_overlap(b1, b2):
    overlap = np.zeros(b1.shape)
    for d, ((min1, max1), (min2, max2)) in enumerate(zip(b1, b2)):
        if not max2 <= min1 and not max1 <= min2:
            if min2 <= min1 <= max2:
                overlap[d, 0] = min1
            elif min1 <= min2 <= max1:
                overlap[d, 0] = min2

            if min2 <= max1 <= max2:
                overlap[d, 1] = max1
            elif min1 <= max2 <= max1:
                overlap[d, 1] = max2

    return overlap



def compare_trees(tree1, tree2, bounds):
    total_volume_difference = 0
    for leaf1 in tree1.leaves():
        s1 = leaf1.state.bounded(*bounds)
        intersects = tree2.get_for_region(s1[:,0], s1[:,1], actions=False)

        for leaf2 in intersects:
            if leaf1.action == leaf2.action:
                continue

            s2 = leaf2.state.bounded(*bounds)
            overlap = get_overlap(s1, s2)
            total_volume_difference += volume(overlap)
            # cut = np.array(cut_overlaps(s1.T, s2.T, allow_multiple=True)).T
            # total_volume_difference += volume(s2) - volume(cut)

    return total_volume_difference


def get_shield_path(model):
    return f'./experiments/automated/{model}/shields/synthesized.zip'

def get_tree_path(model):
    return f'./experiments/automated/{model}/shields/mp_shield.json'

def get_includes(model, base_dir='./dtcontrol/clean_examples/'):
    return [p.split('/')[-1].replace('.csv', '') for p in glob(base_dir + f'{model}*.csv')]


if __name__ == '__main__':

    config = [
        # ["random_walk", None, { ('9',): 0, ('1',): 1, ('0','1'): 2 }, True],
        ["bouncing_ball", bb_callback, { ('9',): 0, ('1',): 1, ('0',): 2, ('0','1'): 3 }, True],
        # ["cruise", None, { ('9',): 0, ('0',): 1, ('0','2'): 2, ('0','1','2'): 3 }, False],
        # ["oil_pump", None, { ('9',): 0, ('1',): 1, ('0',): 2, ('0','1'): 3 }, True]
    ]

    result_columns = [
        'model', 'size', 'safety violations', 'failed runs (perc)', 'equivalence'
    ]

    for model, callback, action_map, unlucky in config:
        shield_path = get_shield_path(model)
        tree_path = get_tree_path(model)
        shield = unpack_shield(shield_path)
        shield.tree = DecisionTree.load_from_file(tree_path)
        include = get_includes(model)

        # run_dtcontrol_suite('./dtcontrol/clean_examples/', include=include)
        names, paths = get_saved_controllers('./dtcontrol/experiments/CART/', include=include)

        all_res = []
        actions = sorted(map(int, list(shield.id_to_actionset.keys())))
        for mname, path in zip(names, paths):
            dtc_tree = import_dtcontrol_tree(path, shield.variables, actions, action_map)
            dtc_shield = ShieldOracleWrapper(dtc_tree, shield.amap, shield.n_actions)

            _, _, info = my_evaluate_policy(
                dtc_shield,
                gym.make(shield.env_id, unlucky=unlucky),
                callback=callback,
                n_eval_episodes=1000
            )

            safety_violations = np.sum(info['deaths'])
            failed_runs = np.round(np.sum(info['deaths'] > 0) / len(info['deaths']) * 100, 2)

            volume_diff = compare_trees(shield.tree, dtc_tree, shield.bounds)
            total_volume = np.product(shield.bounds[1] - shield.bounds[0])
            volume_diff_perc = np.round(volume_diff / total_volume, 6)

            result = [ mname, dtc_tree.n_leaves, safety_violations, failed_runs, 1 - volume_diff_perc ]
            all_res.append(result)

        with open(f'./results_dtcontrol_{model}.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(result_columns)
            for row in all_res:
                writer.writerow(row)



    # shield_path = './experiments/automated/bouncing_ball/shields/synthesized.zip'
    # tree_path = './experiments/automated/bouncing_ball/shields/mp_shield.json'

#     shield_path = './experiments/automated/cruise/shields/synthesized.zip'
#     tree_path = './experiments/automated/cruise/shields/mp_shield.json'
#     base_name = 'cruise_shield'
#     callback = None
#     action_map = {
#         ('3',): 0,
#         ('0',): 1,
#         ('0', '2'): 2,
#         ('0', '1', '2'): 3
#     }

#     shield = unpack_shield(shield_path)
#     shield.tree = DecisionTree.load_from_file(tree_path)
#     # shield.make_tree(minimize=False)

#     include = make_data(
#         base_name, shield.tree, np.array(shield.bounds), shield.amap, shield.variables,
#         # granularities=[1000, 10_000, 100_000, 1_000_000],
#         samples_per_leaf=[1, 2, 4, 8, 16, 32, 64]
#     )

#     print('run dtcontrol')
#     run_dtcontrol_suite('./dtcontrol/examples/', include=include)

#     names, paths = get_saved_controllers(
#         './dtcontrol/experiments/CART/',
#         include=include
#     )

#     result_columns = [
#         'model', 'size', 'safety violations', 'failed runs (perc)', 'equivalent'
#     ]
#     all_res = []

#     for model, path in zip(names, paths):
#         print(model)
#         dtc_tree = import_dtcontrol_tree(path, shield.variables, [0, 1, 2, 3], action_map)
#         dtc_shield = ShieldOracleWrapper(dtc_tree, shield.amap, 4)

#         _, _, info = my_evaluate_policy(
#             dtc_shield,
#             gym.make(shield.env_id, unlucky=True),
#             callback=callback
#         )

#         safety_violations = np.sum(info['deaths'])
#         failed_runs = np.round(np.sum(info['deaths'] > 0) / 100, 2)
#         volume_diff = compare_trees(shield.tree, dtc_tree, shield.bounds)
#         total_volume = np.product(shield.bounds[1] - shield.bounds[0])

#         volume_diff_perc = np.round(volume_diff / total_volume, 6)

#         result = [
#             model,
#             dtc_tree.n_leaves,
#             safety_violations,
#             failed_runs,
#             1 - volume_diff_perc
#         ]
#         all_res.append(result)


#     with open(f'./results_dtcontrol_{base_name}.csv', 'w') as f:
#         writer = csv.writer(f)
#         writer.writerow(result_columns)
#         for row in all_res:
#             writer.writerow(row)


#  partitions_centered, 3342, 0, 0.0, 1.0
