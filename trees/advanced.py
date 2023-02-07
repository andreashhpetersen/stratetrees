import json
import math
import heapq
import random
import numpy as np

from decimal import *
from time import perf_counter
from collections import defaultdict
from trees.models import Node, Leaf, State, DecisionTree


def build_state(intervals):
    variables = list(intervals.keys())
    state = State(variables)
    for var, (min_v, max_v) in intervals.items():
        state.greater_than(var, min_v)
        state.less_than(var, max_v)
    return state


def grow(p, bs, max_i):
    """
    Finds the next dimension `i' from point `p' that minimizes the difference
    between bounds `bs_i[p_i]' and `bs_i[p_i + 1]'. Only unexhausted dimensions
    (ie. where `bs[i][-1] != 0') that are not already at max (ie. `where p[i] +
    1 <= max_i[i]') are considered. Returns the new point and the update
    dimension.

    p:  (K,) array of indicies
    bs: (K, M) array of bounds, where M is the padded max number of bounds
    max_i: (K,) array of max number of bounds for each dimension
    """
    idxs = np.arange(len(p))
    np.random.shuffle(idxs)
    i = idxs[np.logical_and(bs[idxs,-1] == 0, p[idxs] < max_i[idxs])][0]
    p[i] += 1
    return p, i


def is_explored(min_state, max_state, tree):
    if tree.root is None:
        return False
    return len(tree.get_for_region(min_state, max_state)) > 0


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
    track = DecisionTree.empty_tree(tree.variables, [])  # actions arg doesn't matter

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
        action = tree.get(max_state)

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
            track.root = DecisionTree.make_root_from_leaf(track, leaf)
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


def split_box(leaf, v, c):
    vmin, vmax = leaf.state.min_max(v)
    if vmin < c < vmax:
        lstate = leaf.state.copy()
        lstate.less_than(v, c)
        hstate = leaf.state.copy()
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
        return Leaf(0, action=boxes[0].action)

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
        return Leaf(0.0, action=boxes[0].action)

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


def prune_subtree(root, action):
    """
    Attempts to find redundant nodes that imposes conditions that are satisfied
    further `down' the tree. For example:

       x<4
      /   \
     /     \
    a1    x<5
         /   \
        /     \
       a1     a2

    can be reduced to:

       x<5
      /   \
     /     \
    a1     a2
    """

    # both children are leaves
    #
    #     x<b
    #    /   \
    #   /     \
    # a1       a2
    if root.low.is_leaf and root.high.is_leaf:
        if root.low.action == action:
            return root, root.high
        else:
            return root, root.low

    # left child is leaf, right is new subtree
    #
    #     x<b
    #    /   \
    #   /     \
    # a1      y<c
    if root.low.is_leaf and not root.high.is_leaf:
        if root.low.action == action:
            subtree, leaf = prune_subtree(root.high, action)

        else:
            subtree, leaf = prune_subtree(root.high, root.low.action)

        if leaf is None or leaf.action == root.low.action:
            root.high = subtree
            return root, None

        try:
            if leaf.state.min[root.variable] > root.bound:
                return subtree, leaf
            else:
                root.high = subtree
                return root, leaf
        except:
            import ipdb; ipdb.set_trace()

    # right child is leaf, lef is new subtree
    #
    #     x<b
    #    /   \
    #   /     \
    # y<c     a1
    if root.high.is_leaf and not root.low.is_leaf:
        if root.high.action == action:
            subtree, leaf = prune_subtree(root.low, action)

        else:
            subtree, leaf = prune_subtree(root.low, root.high.action)

        if leaf is None or leaf.action == root.high.action:
            root.low = subtree
            return root, None

        if leaf.state.max[root.variable] < root.bound:
            return subtree, leaf
        else:
            root.low = subtree
            return root, leaf

    # both children are new subtrees
    #
    #     x<b
    #    /   \
    #   /     \
    # y<c     y<d
    new_low, leaf1 = prune_subtree(root.low, None)
    new_high, leaf2 = prune_subtree(root.high, None)

    root.low = new_low
    root.high = new_high

    new_leaf = None
    if not (leaf1 is None or leaf2 is None) and leaf1.action == leaf2.action:
        state1, state2 = leaf1.state, leaf2.state
        state = State(state1.variables)
        for v in state.variables:
            state.greater_than(v, min(state1.min[v], state2.min[v]))
            state.less_than(v, max(state1.max[v], state2.max[v]))
        new_leaf = Leaf(0, action=leaf1.action, state=state)

    # here we have potential for further optimization
    # if root.variable == 'Ball[0].v' and round(root.bound, 2) == -11.74:
    #     import ipdb; ipdb.set_trace()

    return root, new_leaf
