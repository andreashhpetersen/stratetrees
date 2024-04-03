import json
import math
import time
import tqdm
import heapq
import random
import numpy as np

from matplotlib.patches import Rectangle

from copy import deepcopy
from decimal import *
from time import perf_counter
from collections import defaultdict
from stratetrees.models import Node, Leaf, State, DecisionTree, MpTree
from stratetrees.utils import has_overlap, cut_overlaps, in_box, in_list, make_state,\
    make_leaf, breaks_box, plot_voxels, leaves_to_state_constraints, \
    draw_graph, performance, draw_partitioning


class SearchHeuristics:
    @classmethod
    def pre_order_low_first(cls, n, s):
        low_state, high_state = s.split(n.var_id, n.bound)
        return n.low, low_state, n.high, high_state

    @classmethod
    def pre_order_high_first(cls, n, s):
        low_state, high_state = s.split(n.var_id, n.bound)
        return n.high, high_state, n.low, low_state

    @classmethod
    def pre_order_random(cls, n, s):
        low_state, high_state = s.split(n.var_id, n.bound)
        if np.random.random() > 0.5:
            return n.low, low_state, n.high, high_state
        else:
            return n.high, high_state, n.low, low_state

    @classmethod
    def choose_max_depth(cls, n, s):
        low_state, high_state = s.split(n.var_id, n.bound)
        if n.low._max_depth >= n.high._max_depth:
            return n.low, low_state, n.high, high_state
        else:
            return n.high, high_state, n.low, low_state

    @classmethod
    def choose_min_depth(cls, n, s):
        low_state, high_state = s.split(n.var_id, n.bound)
        if n.low._min_depth <= n.high._min_depth:
            return n.low, low_state, n.high, high_state
        else:
            return n.high, high_state, n.low, low_state

    @classmethod
    def choose_max_size(cls, n, s):
        low_state, high_state = s.split(n.var_id, n.bound)
        if n.low.size >= n.high.size:
            return n.low, low_state, n.high, high_state
        else:
            return n.high, high_state, n.low, low_state

    @classmethod
    def choose_min_size(cls, n, s):
        low_state, high_state = s.split(n.var_id, n.bound)
        if n.low.size <= n.high.size:
            return n.low, low_state, n.high, high_state
        else:
            return n.high, high_state, n.low, low_state

    @classmethod
    def greedy_then_low(cls, n, s):
        low_state, high_state = s.split(n.var_id, n.bound)
        if n.low.is_leaf and n.low.action is None:
            return n.low, low_state, n.high, high_state
        elif n.high.is_leaf and n.high.action is None:
            return n.high, high_state, n.low, low_state
        else:
            return n.low, low_state, n.high, high_state


def is_explored(min_state, max_state, tree):
    """
    Returns True, if any part of the region given by `min_state` and `max_state`
    is assigned an action in `track`.
    """
    if tree.root is None:
        return False
    return len(tree.get_for_region(min_state, max_state)) > 0


### Functions for new max parts algorithm ###


def init_bounds(partitions, K):
    lmap = defaultdict(list)
    bound_to_nodes = [defaultdict(list) for _ in range(K)]

    # create mapping from a bound value to a list of node (ids) that has
    # that bound as part of its state
    # K x N operation
    for node_id, state in partitions:
        for i in range(K):
            lmap[node_id].append([None, None])
            bound_to_nodes[i][state[i,0]].append(node_id)
            bound_to_nodes[i][state[i,1]].append(node_id)

    # sort bounds in each dimension i
    # K x M operation (where M is the max number of bounds in any dimension)
    sorted_bounds = [sorted(list(bound_to_nodes[i].keys())) for i in range(K)]

    # make lmap map from a node_id to an index pointer representation of the
    # state of the node. Also get number of bounds in each dimension.
    # K x M x 2 operation
    n_bounds = np.zeros((K,), dtype=np.int32)
    for i in range(K):
        n_bounds[i] = len(sorted_bounds[i]) - 1
        for idx, bound in enumerate(sorted_bounds[i]):
            for node_id in bound_to_nodes[i][bound]:
                node_bounds = lmap[node_id]

                # check wether to set min or max bound
                m = 0 if node_bounds[i][0] is None else 1
                node_bounds[i][m] = idx

    for k, v in lmap.items():
        lmap[k] = np.array(v).T

    M = n_bounds.max()
    for i in range(K):
        sorted_bounds[i] = np.pad(
            sorted_bounds[i],
            (0, int(M - n_bounds[i])),
            mode='empty'
        )

    return lmap, np.array(sorted_bounds), n_bounds


def get_unexhausted_dim(exhausted):
    return np.random.choice(np.arange(exhausted.shape[0])[exhausted == 0])


def update_track_tree(track, state):
    leaf = make_leaf(1, track.variables, state)
    if track.root is None:
        track.root = Node.make_node_from_leaf(leaf, track.variables)
    else:
        track.put_leaf(leaf, prune=True)


def update_region_bounds(p_state, state, tree, lmap):
    """
    Parameters
    ----------
    lmap : dictionary
        The region bounds to be updated
    """
    parts = tree.predict_for_region(state[:,0], state[:,1], node_ids=True)
    for nid in parts:
        lmap[nid] = cut_overlaps(p_state, lmap[nid])


def find_unexplored_state_dfs(n, state, func):
    child1, state1, child2, state2 = func(n, state)

    if not child1.is_leaf:
        return find_unexplored_state_dfs(child1, state1, func)

    elif child1.action is None:
        return state1

    if not child2.is_leaf:
        return find_unexplored_state_dfs(child2, state2, func)

    elif child2.action is None:
        return state2

    return None


def max_parts(tree, seed=None, return_info=False, \
               heuristic_func=SearchHeuristics.pre_order_low_first,
               animate=False, draw_dims=[], min_v=0, max_v=None):

    if seed is not None:
        np.random.seed(seed)

    org_tree = tree
    tree = MpTree(tree)
    K = tree.n_features

    partitions = tree.partitions()
    lmap, bounds, n_bounds = init_bounds(partitions, K)

    if animate:
        patches = []

        if len(draw_dims) == 0:
            draw_dims = np.arange(K)

        if max_v is None:
            max_v = int(bounds[np.arange(K), n_bounds - 1].max() + 1)

    # used to gather track sizes (if return_info=True)
    ts = []

    regions = []
    track = DecisionTree.empty_tree(tree.variables, tree.actions)

    mstate = make_state(tuple((0,) * K for _ in range(2)), bounds)
    while True:

        if track.root is None:
            node_id = tree.predict_node_id(mstate[:,0], cheat=True)
        else:
            mstate = State(tree.variables)
            mstate = find_unexplored_state_dfs(
                track.root,
                mstate,
                heuristic_func
            )
            if mstate is None:
                break
            node_id = tree.predict_node_id(mstate[:,0], cheat=True)

        # get updated pmin and pmax
        pmin, pmax = lmap[node_id]

        # make state and check if it is explored
        state = make_state((pmin, pmax), bounds)
        min_state, max_state = state[:,0], state[:,1]

        # predict action
        action = tree.predict(max_state)

        # init array to keep track of exhausted dimensions and exhaust where
        # we are already on the edge of the state space
        exhausted = np.zeros((K,))
        exhausted[np.array(pmax) == n_bounds] = 1

        # main loop
        healing = False
        cand_pmax = pmax.copy()
        while exhausted.sum() < K:

            # if we aren't healing, do an incremental expansion
            if not healing:
                dim = get_unexhausted_dim(exhausted)
                cand_pmax[dim] += 1

                diff_state = min_state.copy()
                diff_state[dim] = bounds[dim][pmax[dim]]

            # make candidate state
            cand_state = make_state(cand_pmax, bounds)

            # get partitions intersecting with current state (diff_state, cand_state)
            parts = tree.predict_for_region(diff_state, cand_state, node_ids=True)

            # invariants to be checked
            explored = is_explored(diff_state, cand_state, track)
            actions = set([tree.values[nid] for nid in parts])
            broken = [nid for nid in parts if breaks_box(lmap[nid], (pmin, cand_pmax))]

            # we either encountered explored territory or different actions
            if explored or set([action]) != actions:
                healing = False
                exhausted[dim] = 1
                cand_pmax[dim] = pmax[dim]

            # we broke one or more regions in more than 2 pieces
            elif len(broken) > 0:
                healing = True
                cand_update = max([lmap[nid][1][dim] for nid in broken])

                # terminal case, where no expansion in dim can heal the broken
                if cand_update <= cand_pmax[dim]:
                    healing = False
                    cand_pmax[dim] = pmax[dim]
                    exhausted[dim] = 1
                else:
                    cand_pmax[dim] = cand_update

            # we made a completely legit expansion in dim
            else:
                healing = False
                pmax = cand_pmax.copy()

                # we have reached the edge of dim
                if cand_pmax[dim] == n_bounds[dim]:
                    exhausted[dim] = 1

        # make new region and update the track tree
        reg = make_state((pmin, pmax), bounds)
        regions.append(make_leaf(action, tree.variables, reg))

        update_track_tree(track, reg)
        update_region_bounds((pmin, pmax), reg, tree, lmap)

        if return_info:
            ts.append(track.n_leaves)

        if animate:
            if len(tree.variables) == 2:
                xymin, xymax = reg[:,0], reg[:,1]
                xymin[xymin == -np.inf] = 0
                xymax[xymax == np.inf] = max_v

                w, h = xymax[0] - xymin[0], xymax[1] - xymin[1],
                hatch = '\\\\' if action == 0 else '//'
                color = '#ffffb3' if action == 0 else '#bebada'
                patches.append(Rectangle(xymin, w, h, color=color, lw=0))
                patches.append(Rectangle(xymin, w, h, fill=None, hatch=hatch))
                draw_partitioning(
                    org_tree.leaves(), 'x', 'y', [0,4], [0,4],
                    cmap=['#ffffb3', '#bebada'], extra_patches=patches,
                    labels=None, show=True, lw=0.4,
                    xticks=[0,1,2,3,4], yticks=[0,1,2,3,4], out_fp='iterative.png'
                )

            if len(tree.variables) == 3:
                bs = leaves_to_state_constraints(regions)[:,draw_dims,:]
                acts = [l.action for l in regions]

                if isinstance(mstate, State):
                    bs = np.vstack((bs, [mstate.constraints]))
                    acts.append(2)

                plot_voxels(bs, acts, max_v=max_v)

    if return_info:
        info = { 'track_sizes': np.array(ts) }
        return regions, info

    return regions


def minimize_tree(tree, max_iter=10, \
                  verbose=True, early_stopping=True, return_trace=True):
    """
    Run max_parts on `tree` and convert the result into a new decision tree.
    If `max_iter` is bigger than 0, repeat the process for that many times.
    If `early_stopping` is True, stop if neither the amount of leaves found by
    max_parts nor the size of the new tree has decreased since last step.
    If `return_trace` is True, the function returns a tuple with data on the
    intermediate steps as well as the index of the best results in addition to
    returning the minimal tree.
    """

    variables, actions = tree.variables, tree.actions

    if verbose:
        print(f'minimizing original tree with {tree.n_leaves} leaves')

    with performance() as p:
        leaves = max_parts(tree)
        ntree = leaves_to_tree(leaves, variables, actions)

    ntree.meta = tree.meta

    if verbose:
        print(f'found {len(leaves)} leaves')
        print(f'constructed new tree of with {ntree.n_leaves} leaves')
        print(f'max/min depth: {ntree.max_depth}/{ntree.min_depth}\n')

    best_n_leaves, best_n_tree = len(leaves), ntree.size
    best_tree, best_tree_i = ntree, 0

    data = [[len(leaves), ntree.n_leaves, ntree.max_depth, ntree.min_depth, p.time]]

    i = 1
    while i < max_iter:
        with performance() as p:
            leaves = max_parts(ntree)
            ntree = leaves_to_tree(leaves, variables, actions)

        if verbose:
            print(f'found {len(leaves)} leaves')
            print(f'constructed new tree of with {ntree.n_leaves} leaves')
            print(f'max/min depth: {ntree.max_depth}/{ntree.min_depth}\n')

        data.append(
            [len(leaves), ntree.n_leaves, ntree.max_depth, ntree.min_depth, p.time]
        )

        progress = len(leaves) < best_n_leaves or ntree.size < best_n_tree
        if early_stopping and not progress:
            if verbose:
                print(f'stopping early as no progress was seen')
            break

        if len(leaves) < best_n_leaves:
            best_n_leaves = len(leaves)

        if ntree.size < best_n_tree:
            best_n_tree = ntree.size
            best_tree = ntree
            best_tree_i = i

        i += 1

    return best_tree, (np.array(data), best_tree_i)


### Functions for reconstructing a DecisionTree from a list of leaves ###


def split_leaf(leaf, v, c):
    vmin, vmax = leaf.state.min_max(v)

    if vmin < c < vmax:
        lstate = leaf.state.copy()
        hstate = leaf.state.copy()

        lstate.less_than(v, c)
        hstate.greater_than(v, c)

        low = Leaf(leaf.action, state=lstate)
        high = Leaf(leaf.action, state=hstate)

    elif vmax <= c:
        low, high = leaf, None

    else:  # vmin > c
        low, high = None, leaf

    return low, high


def split_leaves_list(leaves, variables):

    # map each variable to a sorted list of indicies of the leaves in `leaves'
    # where the sorting is done according to maximum value of the respective
    # variable for that leaf
    max_sorted = {
        # the variable is key, the list is the value
        v: sorted(
            # this just generates indicies from [0, 1, ..., len(leaves) - 1]
            range(len(leaves)),

            # this sorts the above indicies in ascending order according to the
            # maximum value of v in the box that the index corresponds to
            key=lambda x: list(leaves[x].state.min_max(v))[::-1]
        )

        # iterate the variables
        for v in variables
    }

    # store potential cuts here
    cuts = []

    # go through each variable
    for v in variables:

        # list of indicies of entries in `leaves' sorted according to their max
        # value for the variable `v'
        curr_l = max_sorted[v]

        # go through each index in the sorted list
        for i in range(len(curr_l) - 1):

            # this is the lowest max value for v in the rest of the list (ie.
            # leaves[curr_l[i]].state.max(v) <= leaves[curr_l[j]].state.max(v)
            # for all j > i)
            max_v = leaves[curr_l[i]].state.max(v)

            if max_v == leaves[curr_l[i+1]].state.max(v):
                continue

            # ideally, we'd split at i == len(leaves) / 2, so we start by
            # defining impurity as the difference between `i' and the optimal
            # value (the halfway point)
            impurity = abs((len(leaves) / 2) - (i + 1))

            # look at the rest of curr_l (with large max values for v)
            for j in range(i + 1, len(curr_l)):

                # if the min value for v in curr_l[j] is less that max_v, we
                # know that the box will be cut in 2, since it's max value for
                # v by design must be greater than max_v
                if leaves[curr_l[j]].state.min(v) < max_v:
                    impurity += 1

            # add the triplet to our list of possible cuts
            cuts.append((v, i, impurity))

    # all leaves had same action
    if len(cuts) == 0:
        return Leaf(leaves[0].action)

    # sort according to impurity and take the 'most pure' cut
    v, b_id, _ = sorted(cuts, key=lambda x: x[2])[0]

    # grab that optimal value
    max_val = leaves[max_sorted[v][b_id]].state.max(v)

    # separate the leaves into list of those that are lower than our cut, those
    # that are higher or both if it falls on both sides of the cut
    low, high = [], []
    for b in leaves:
        l, h = split_leaf(b, v, max_val)
        if l is not None:
            low.append(l)
        if h is not None:
            high.append(h)

    # something went wrong if we end up here
    assert not (len(low) == 0 or len(high) == 0)

    # create the new branch node with a cut on v <= max_val
    return (v, max_val), low, high


def find_cut_impurity(idx, leaves, sorted_idxs):
    # this is the lowest max value for v in the rest of the list (ie.
    # leaves[curr_l[i]].state.max(v) <= leaves[curr_l[j]].state.max(v)
    # for all j > i)
    max_v = leaves[sorted_idxs[i]].state.max(v)

    if max_v == leaves[sorted_idxs[i+1]].state.max(v):
        return None

    # ideally, we'd split at i == len(leaves) / 2, so we start by
    # defining impurity as the difference between `i' and the optimal
    # value (the halfway point)
    impurity = abs((len(leaves) / 2) - (i + 1))

    # look at the rest of curr_l (with large max values for v)
    n_cuts = 0
    for j in range(i + 1, len(sorted_idxs)):

        # if the min value for v in curr_l[j] is less that max_v, we
        # know that the box will be cut in 2, since it's max value for
        # v by design must be greater than max_v
        if leaves[sorted_idxs[j]].state.min(v) < max_v:
            n_cuts += 1

    impurity = abs(((len(leaves) + n_cuts) / 2) - (i + 1 + n_cuts)) + n_cuts
    return impurity


def split_leaves_list2(leaves, variables):

    # map each variable to a sorted list of indicies of the leaves in `leaves'
    # where the sorting is done according to maximum value of the respective
    # variable for that leaf
    max_sorted = {
        # the variable is key, the list is the value
        v: sorted(
            # this just generates indicies from [0, 1, ..., len(leaves) - 1]
            range(len(leaves)),

            # this sorts the above indicies in ascending order according to the
            # maximum value of v in the box that the index corresponds to
            key=lambda x: list(leaves[x].state.min_max(v))[::-1]
        )

        # iterate the variables
        for v in variables
    }

    # store potential cuts here
    cuts = []

    # go through each variable
    for v in variables:

        # list of indicies of entries in `leaves' sorted according to their max
        # value for the variable `v'
        curr_l = max_sorted[v]

        # go through each index in the sorted list
        for i in range(len(curr_l) - 1):
            impurity = find_cut_impurity(i, leaves, curr_l)

            # add the triplet to our list of possible cuts (if impurity was
            # found)
            if impurity is not None:
                cuts.append((v, i, impurity))

    # all leaves had same action
    if len(cuts) == 0:
        return Leaf(leaves[0].action)

    # sort according to impurity and take the 'most pure' cut
    v, b_id, _ = sorted(cuts, key=lambda x: x[2])[0]

    # grab that optimal value
    max_val = leaves[max_sorted[v][b_id]].state.max(v)

    # separate the leaves into list of those that are lower than our cut, those
    # that are higher or both if it falls on both sides of the cut
    low, high = [], []
    for b in leaves:
        l, h = split_leaf(b, v, max_val)
        if l is not None:
            low.append(l)
        if h is not None:
            high.append(h)

    # something went wrong if we end up here
    assert not (len(low) == 0 or len(high) == 0)

    # create the new branch node with a cut on v <= max_val
    return (v, max_val), low, high


def make_branch_node(leaves, variables, vmap):

    # only one leaf, return it
    if len(leaves) == 1:
        return Leaf(leaves[0].action)

    split = split_leaves_list(leaves, variables)

    # all leaves had same action, so just return leaf
    if isinstance(split, Leaf):
        return split

    # split on v <= bound
    (v, bound), low_ls, high_ls = split

    # make child nodes
    low = make_branch_node(low_ls, variables, vmap)
    high = make_branch_node(high_ls, variables, vmap)

    # return branch node
    return Node(v, vmap[v], bound, low, high)


def leaves_to_tree(leaves, variables, actions=[]):
    vmap = leaves[0].state.var2id
    root = make_branch_node(leaves, variables, vmap)
    return DecisionTree(root.prune(), variables, actions)
