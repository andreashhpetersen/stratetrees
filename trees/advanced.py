import json
import math
import time
import tqdm
import heapq
import random
import numpy as np

from copy import deepcopy
from decimal import *
from time import perf_counter
from collections import defaultdict
from trees.models import Node, Leaf, State, DecisionTree, MpTree
from trees.utils import has_overlap, cut_overlaps, in_box, in_list, make_state,\
    make_leaf, breaks_box


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
    n_bounds = np.zeros((K,), dtype=np.int)
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


def update_track_tree(track, state, action):
    assert action is not None
    leaf = make_leaf(action, track.variables, state)
    if track.root is None:
        track.root = track.make_root_from_leaf(leaf)
    else:
        track.put_leaf(leaf)


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


def add_points(p_state, points, n_bounds):
    pmin, pmax = p_state
    K = len(pmin)
    for i in range(K):
        if pmax[i] == n_bounds[i]:
            continue

        new_point = pmin.copy()
        new_point[i] = pmax[i]
        heapq.heappush(points, tuple(new_point))


def max_parts3(tree, seed=None, return_track_tree=False):
    if seed is not None:
        np.random.seed(seed)

    tree = MpTree(tree)
    K = tree.n_features

    partitions = tree.partitions()
    lmap, bounds, n_bounds = init_bounds(partitions, K)

    regions = []
    points = [tuple([0 for _ in range(K)])]
    track = DecisionTree.empty_tree(tree.variables, tree.actions)

    while len(points) > 0:

        # pop lexicographically smallest point and predict node
        opmin = heapq.heappop(points)
        node_id = tree.predict_node_id(make_state(opmin, bounds), cheat=True)

        # get updated pmin and pmax
        pmin, pmax = lmap[node_id]

        # make state and check if it is explored
        state = make_state((pmin, pmax), bounds)
        min_state, max_state = state[:,0], state[:,1]
        if is_explored(min_state, max_state, track):
            continue

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

        update_track_tree(track, reg, action)
        update_region_bounds((pmin, pmax), reg, tree, lmap)
        add_points((pmin, pmax), points, n_bounds)

    # used in testing to assert everything is explored
    if return_track_tree:
        return regions, track

    return regions


### Functions for old max parts algorithm ###


def grow(p, bs, max_i):
    """
    Non-deterministically choose an unexhausted dimension in which to increment
    `p` by one.

    p:  (K,) array of indicies
    bs: (K, M) array of bounds, where M is the padded max number of bounds
    max_i: (K,) array of max number of bounds for each dimension
    """
    idxs = np.arange(len(p))
    np.random.shuffle(idxs)
    i = idxs[np.logical_and(bs[idxs,-1] == 0, p[idxs] < max_i[idxs])][0]
    p[i] += 1
    return p, i


def max_parts(tree, min_vals=None, max_vals=None, padding=1):
    """
    Args:
        root: the root `Node` from which to extract boxes
        variables: the list of variables in `root`
        max_vals: (optional) a list of maximum values for the variables. If not
                  provided, the maximum values will all be set to infinity
        min_vals: (optional) a list of minimum values for the variables. If not
                  provided, the minimum values will be calculated by subtracting
                  1 from the smallest bound for each variable
    """

    # the number of state dimensions
    K = len(tree.variables)

    if min_vals is None:
        min_vals = [None for _ in range(K)]

    if max_vals is None:
        max_vals = [None for _ in range(K)]

    # each element in vbounds is a list of constraints for a specific dimension
    vbounds = tree.get_bounds()  # assumed to be sorted in ascending order

    # store the number of constraints for any dimension i (for indexing)
    max_i = np.zeros((K,), dtype=int)

    # build list of constraints
    bounds = []
    for i in range(K):
        bs = np.array(vbounds[i])

        # add min and max values if there are none
        if min_vals[i] is None:
            if bs.shape[0] == 0:
                min_vals[i] = -math.inf
            else:
                min_vals[i] = np.amin(bs) - padding

        if max_vals[i] is None:
            if bs.shape[0] == 0:
                max_vals[i] = math.inf
            else:
                max_vals[i] = np.amax(bs) + padding

        # make list of constraints in range [min_val, max_val] (inclusive)
        bs = np.hstack((
            min_vals[i],
            bs[np.logical_and(bs > min_vals[i], bs < max_vals[i])],
            max_vals[i]
        ))

        # register number of constraints minus 1 (ie. the index of the last
        # constraint)
        max_i[i] = bs.shape[0] - 1

        # add the array to the list of bounds
        bounds.append(bs)

    # arrange the constraints in a (K,M) matrix, where M is the length of the
    # largest list of constraints + 1. The + 1 is for marking the dimension as
    # exhausted
    max_len = np.amax(max_i) + 2
    for i in range(K):
        bounds[i] = np.pad(bounds[i], (0, max_len - bounds[i].shape[0]))
    bounds = np.vstack(bounds)

    # start the list of points from the minimum values
    points = [[0 for _ in range(K)]]

    # we use a heap to process the points in a lexical order
    heapq.heapify(points)

    # define a tree to keep track of what has been covered
    track = DecisionTree.empty_tree(tree.variables, tree.actions)

    # this is what we return in the end - lets start!
    regions = []

    # we keep going for as long as there are points
    while len(points) > 0:

        # p_min and p_max contains indicies of the current constraints
        p_min = np.array(heapq.heappop(points), dtype=int)
        p_max = p_min.copy() + 1

        # define the region spanned by min_state and max_state
        min_state = bounds[np.arange(K), p_min]
        max_state = bounds[np.arange(K), p_max]

        # check if we have already explored this state
        if is_explored(min_state, max_state, track):
            continue

        # get the action that the entire region must agree on
        action = tree.predict(max_state)

        # reset exhausted variables
        bounds[:,-1] = 0
        bounds[:,-1][p_max == max_i] = 1

        while np.sum(bounds[:,-1]) < K:
            # grow in dimension i and update p_max and max_state
            p_max, i = grow(p_max, bounds, max_i)
            max_state[i] = bounds[i][p_max[i]]

            # state to express difference between previous max_state and current
            diff_state = min_state.copy()
            diff_state[i] = bounds[i][p_max[i] - 1]

            # check if new region is explored
            explored = is_explored(diff_state, max_state, track)

            # check if our new region returns a more than one action
            actions = tree.get_for_region(diff_state, max_state)

            if (actions != set([action]) or explored or p_max[i] == max_i[i]):

                # mark variable as exhausted
                bounds[i,-1] = 1

                # roll back to last state
                if actions != set([action]) or explored:
                    p_max[i] -= 1
                    max_state[i] = diff_state[i]

        # create the region as a leaf with a state spanned by min_state and
        # max_state
        state = State(tree.variables, np.vstack((min_state, max_state)).T)
        leaf = Leaf(cost=0, action=action, state=state)
        regions.append(leaf)

        # create the root of the tracking tree if we haven't done so yet
        if track.root is None:
            track.root = track.make_root_from_leaf(leaf)
        # or add to the tree, so we know not to explore this part again
        else:
            track.put_leaf(leaf)

        # add new points from which to start later
        for i in range(K):
            if p_max[i] < max_i[i]:

                new_p = [idx for idx in p_min]
                new_p[i] = p_max[i]
                heapq.heappush(points, new_p)

    return regions


### Functions for reconstructing a DecisionTree from a list of leaves ###


def split_box(leaf, v, c):
    vmin, vmax = leaf.state.min_max(v)

    if vmin < c < vmax:
        lstate = leaf.state.copy()
        hstate = leaf.state.copy()

        lstate.less_than(v, c)
        hstate.greater_than(v, c)

        low = Leaf(0, action=leaf.action, state=lstate)
        high = Leaf(0, action=leaf.action, state=hstate)

    elif vmax <= c:
        low, high = leaf, None

    else:  # vmin > c
        low, high = None, leaf

    return low, high


def find_best_cut(boxes, variables, vmap):

    # map each variable to a sorted list of indicies of the leaves in `boxes'
    # where the sorting is done according to maximum value of the respective
    # variable for that leaf
    max_sorted = {
        # the variable is key, the list is the value
        v: sorted(
            # this just generates indicies from [0, 1, ..., len(boxes) - 1]
            range(len(boxes)),

            # this sorts the above indicies in ascending order according to the
            # maximum value of v in the box that the index corresponds to
            key=lambda x: list(boxes[x].state.min_max(v))[::-1]
        )

        # iterate the variables
        for v in variables
    }

    # store potential cuts here
    cuts = []

    # go through each variable
    for v in variables:

        # list of indicies of entries in `boxes' sorted according to their max
        # value for the variable `v'
        curr_l = max_sorted[v]

        # go through each index in the sorted list
        for i in range(len(curr_l) - 1):

            # this is the lowest max value for v in the rest of the list (ie.
            # boxes[curr_l[i]].state.max(v) <= boxes[curr_l[j]].state.max(v)
            # for all j > i)
            max_v = boxes[curr_l[i]].state.max(v)

            if max_v == boxes[curr_l[i+1]].state.max(v):
                continue

            # ideally, we'd split at i == len(boxes) / 2, so we start by
            # defining impurity as the difference between `i' and the optimal
            # value (the halfway point)
            impurity = abs((len(boxes) / 2) - (i + 1))

            # look at the rest of curr_l (with large max values for v)
            for j in range(i + 1, len(curr_l)):

                # if the min value for v in curr_l[j] is less that max_v, we
                # know that the box will be cut in 2, since it's max value for
                # v by design must be greater than max_v
                if boxes[curr_l[j]].state.min(v) < max_v:
                    impurity += 1

            # add the triplet to our list of possible cuts
            cuts.append((v, i, impurity))

    if len(cuts) == 0:
        assert len(set([l.action for l in boxes]))
        leaf = Leaf(0, action=boxes[0].action)
        leaf.id = boxes[0].id
        return leaf

    # sort according to impurity and take the 'most pure' cut
    v, b_id, _ = sorted(cuts, key=lambda x: x[2])[0]

    # grab that optimal value
    max_val = boxes[max_sorted[v][b_id]].state.max(v)

    # separate the boxes into list of those that are lower than our cut, those
    # that are higher or both if it falls on both sides of the cut
    low, high = [], []
    for b in boxes:
        l, h = split_box(b, v, max_val)
        if l is not None:
            low.append(l)
        if h is not None:
            high.append(h)

    # something went wrong if we end up here
    if len(low) == 0 or len(high) == 0:
        import ipdb; ipdb.set_trace()

    # create the new branch node with a cut on v <= max_val
    node = Node(v, vmap[v], max_val)

    return node, low, high


def cut_to_node(boxes, variables, vmap):
    if len(boxes) == 1:
        leaf = Leaf(0.0, action=boxes[0].action)
        return leaf

    res = find_best_cut(boxes, variables, vmap)
    if isinstance(res, Leaf):
        return res

    # else
    node, low, high = res

    node.low = cut_to_node(low, variables, vmap)
    node.high = cut_to_node(high, variables, vmap)
    return node


def boxes_to_tree(boxes, variables, actions=[]):
    tree = DecisionTree.empty_tree(variables, actions)
    root = cut_to_node(boxes, variables, tree.var2id).prune()
    root.set_state(State(variables))
    tree.root = root
    tree.size = root.size
    return tree
