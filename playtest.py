import numpy as np
import matplotlib.pyplot as plt

from trees.advanced import max_parts3, boxes_to_tree, init_bounds
from trees.models import DecisionTree, MpTree
from trees.nodes import State
from trees.utils import get_edge_vals, set_edges, calc_volume, \
    leaves_to_state_constraints, plot_voxels

strategy_path = './automated/cartpole/generated/constructed_0/trees/dt_original.json'


def compare_volumes(ls1, ls2):
    edges = get_edge_vals(ls1, leaves=True, broadcast=False)
    bs = set_edges(ls2, edges=edges, leaves=True)

    vol1 = calc_volume(ls1, leaves=True)
    vol2 = calc_volume(bs)

    return np.isclose(vol1, vol2), (vol1, vol2)


def is_consistent(tree, ntree, verbose=False):
    bad_leaves = []
    for leaf in ntree.leaves():
        action = leaf.action
        state = leaf.state.constraints
        if not tree.get_for_region(state[:,0], state[:,1]) == set([action]):
            if verbose:
                bad_leaves.append(leaf)
            else:
                return False
    return len(bad_leaves) == 0, bad_leaves if verbose else True


def generate_minimal_tree(path):
    tree = DecisionTree.load_from_file(path)
    variables, actions = tree.variables, tree.actions

    todos = [
        DecisionTree(tree.root.high, variables, actions),
        DecisionTree(tree.root.low, variables, actions)
    ]
    best_tree = todos[-1].copy()

    found_one = True
    count = 0

    print('start looking for minimal tree.. patience is advised!')
    while len(todos) > 0:
        tree = todos.pop()
        prev_tree = None

        done = False
        while not done and not tree.root.is_leaf:
            print(count, tree.size, len(todos))

            tree.root.set_state(State(variables))
            boxes = max_parts3(tree, seed=42)
            equal_vol, (tree_vol, box_vol) = compare_volumes(tree.leaves(), boxes)

            ntree = boxes_to_tree(boxes, tree.variables, actions=tree.actions)
            consistent = is_consistent(tree, ntree)

            done = consistent

            if not done:
                count += 1
                # bs = leaves_to_state_constraints(tree.leaves())
                # if np.isinf(np.hstack((bs.T))).all(axis=1).any():
                #     found_one = True
                #     prev_tree = tree.copy()

                prev_tree = tree.copy()
                todos.append(DecisionTree(tree.root.high, variables, actions))
                tree = DecisionTree(tree.root.low, variables, actions)
            else:
                count += tree.size
                if found_one and prev_tree is not None and prev_tree.size < best_tree.size:
                    best_tree = prev_tree.copy()
                    print(f'new minimal tree of size {best_tree.size}')
                    # import ipdb; ipdb.set_trace()

    if not found_one:
        print('did not find any less than 4d')
    else:
        print(f'best tree has size {best_tree.size}')
        return best_tree


def transform_bounds(bounds, n_bounds, pad=2):
    upper = n_bounds.max() * pad
    bounds_map = {}
    for dim in range(len(n_bounds)):
        for i in range(n_bounds[dim] + 1):
            bounds_map[bounds[dim][i]] = upper if i == n_bounds[dim] else i * pad

    return bounds_map


def rec(node, bmap):
    if node.is_leaf:
        return

    node.bound = bmap[node.bound]
    rec(node.low, bmap)
    rec(node.high, bmap)
    return


def set_parents(node):
    if node.is_leaf:
        return

    node.low.parent = node
    node.high.parent = node

    set_parents(node.low)
    set_parents(node.high)


def nuke_leaves(tree):

    node = tree.root
    low_child = node.low
    high_child = node.high

    if low_child.low.is_leaf:
        leaf = low_child.low
        node.low = node.low.high


def plot_bars(bs, labels):
    cols = 3
    rows = int(np.ceil(bs.shape[0] / 3))
    fig, axs = plt.subplots(rows, cols)
    colors = ['blue', 'red', 'green', 'black']

    y = np.arange(4)[::-1]
    x_ticks = np.arange(11)

    b_id = 0
    for r in range(rows):
        for c in range(cols):
            if b_id >= len(bs):
                continue

            b = bs[b_id]
            b[b == np.inf] = 10
            b[b == -np.inf] = 0
            axs[r][c].barh(
                y, b[:,1] - b[:,0],
                left=b[:,0], color=colors
            )
            axs[r][c].set_xticks(x_ticks)
            b_id += 1
    plt.show()



if __name__ == '__main__':
    try:
        tree = DecisionTree.load_from_file('./min_problem_less_than_4d.json')
        print('found existing tree')
    except FileNotFoundError:
        print('generating new minimal tree')
        tree = generate_minimal_tree(strategy_path)
        tree.save_as('./min_problem_less_than_4d.json')


    _, bounds, n_bounds = init_bounds(MpTree(tree).partitions(), 4)
    bmap = transform_bounds(bounds, n_bounds, pad=2)

    rec(tree.root, bmap)
    tree.set_state()
    boxes = max_parts3(tree, seed=42)

    bs_box = np.array([b.state.constraints.copy() for b in boxes])
    bs_tree = np.array([b.state.constraints.copy() for b in tree.leaves()])


    # plot bars
    # labels = list(map(lambda x: x.split('.')[-1], tree.variables))
    # plot_bars(bs_tree, labels)


    # plot in 3d
    plot_voxels(bs_box, [l.action for l in boxes])


    # boxes = max_parts3(tree, seed=42)
    # edges = get_edge_vals(tree.leaves(), leaves=True, broadcast=False)

    # bs_tree = leaves_to_state_constraints(tree.leaves())
    # bs_box = leaves_to_state_constraints(boxes)

    # tree_vol = calc_volume(tree.leaves(), leaves=True)
    # box_vol = calc_volume(boxes, leaves=True)

# specific to one tree (./minimal_problem_tree_problem2.json)
# _, bounds, _ = init_bounds(MpTree(tree).partitions(), 4)
# missing = np.array([
#     [bounds[0][1], np.inf],
#     [bounds[1][1], np.inf],
#     [-np.inf, np.inf],
#     [-np.inf, bounds[3][1]]
# ])
# assert calc_volume(set_edges(missing, edges=edges)) + box_vol == tree_vol


# import ipdb; ipdb.set_trace()
