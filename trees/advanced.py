import json
import math
import drawSvg as draw
from trees.models import Node, Leaf, State


def build_state(intervals):
    variables = list(intervals.keys())
    state = State(variables)
    for var, (min_v, max_v) in intervals.items():
        state.greater_than(var, min_v)
        state.less_than(var, max_v)
    return state


def get_boxes(root, variables, eps=0.001, max_vals=None, min_vals=None):
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
    var2id = { v: i for i, v in enumerate(variables) }
    id2var = { i: v for i, v in enumerate(variables) }

    if max_vals is None:
        max_vals = [math.inf for _ in variables]
    max_var_vals = { v: m for v, m in zip(variables, max_vals) }
    min_var_vals = {} if min_vals is None else {
        v: m for v, m in zip(variables, min_vals)
    }

    # build the list of constraints
    constraints = []
    all_bounds = root.get_bounds()  # assumed to be sorted in ascending order
    for var, bounds in all_bounds.items():
        if min_vals is None:
            min_var_vals[var] = bounds[0] - 1
        constraints += [
            (var2id[var], b) for b in bounds
        ] + [(var2id[var], max_var_vals[var])]

    # start the list of points from the minimum values
    points = [tuple(min_var_vals[v] for v in variables)]

    # store the boxes
    boxes = []

    # define a tree to keep track of what has been covered
    tree = None

    while len(points) > 0:

        # reset exhausted variables
        exhausted = []

        # sort points 'from left to right' and get first one
        points.sort()
        p = points.pop(0)

        # add epsilon to 'enter' the box
        p = tuple(x + eps for x in p)

        # define the state
        state = { v: p[var2id[v]] for v in variables }

        # check if we have already explored this state
        if tree is not None:
            leaf = tree.get_leaf(state)
            if leaf.action is not None:
                continue

        # sort constraints according to our current point
        constraints.sort(key=lambda x: -(p[x[0]] - x[1]))

        # get action at state
        action = root.get_leaf(state).action

        # go through constrains
        for i in range(len(constraints)):

            # extract constraint values (variable and bound)
            var, bound = constraints[i]

            # since we don't remove constraints, we should skip those, that we have
            # already checked previously (ie. those smaller than our current point)
            if bound <= p[var]:
                continue

            # continue if var is exhausted
            if var in exhausted:
                continue

            # remember our old state, if we have to roll back
            old_val = state[variables[var]]

            # update the state with the bound of the new constraint
            state[variables[var]] = bound

            # construct a symbolic state
            sym_state = build_state({
                v: (p[var2id[v]], state[v]) for v in variables
            })

            # check if our new state returns a different action
            leafs = root.get_leafs_at_symbolic_state(sym_state, pairs=[])
            state_actions = set([a for a, l in leafs])

            if len(state_actions) > 1 or bound == math.inf:

                # roll back to last state
                if len(state_actions) > 1:
                    state[variables[var]] = old_val

                # mark variable as exhausted
                exhausted.append(var)

                # continue if we still have unexhausted variables
                if len(exhausted) < len(variables):
                    continue

                # otherwise, store the big box as a leaf
                box = build_state({
                    v: (p[var2id[v]] - eps, state[v]) for v in variables
                })
                leaf = Leaf(cost=0, action=action, state=box)
                boxes.append(leaf)

                # add to the tree, so we know not to explore this part again
                if tree is not None:
                    tree.put_leaf(leaf, State(variables))
                # or create the tree if we haven't done so yet
                else:
                    tree = Node.make_root_from_leaf(leaf)

                for v in variables:
                    if p[var2id[v]] != state[v] and state[v] < max_var_vals[v]:
                        points.append(tuple(
                            p[var2id[w]] - eps if w != v else state[w]
                            for w in variables
                        ))
                break

    return boxes
