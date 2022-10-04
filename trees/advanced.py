import json
import math
import random
import drawSvg as draw
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from decimal import *
from collections import defaultdict
from trees.models import Node, Leaf, State
from matplotlib.patches import Rectangle


def build_state(intervals):
    variables = list(intervals.keys())
    state = State(variables)
    for var, (min_v, max_v) in intervals.items():
        state.greater_than(var, min_v)
        state.less_than(var, max_v)
    return state


def max_parts(root, variables, eps=0.001, max_vals=None, min_vals=None):
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

    # for numerically safe addition/subtraction
    eps = Decimal(str(eps))

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
        p = tuple(Decimal(str(x)) + eps for x in p)

        # define the state
        state = { v: p[var2id[v]] for v in variables }

        # check if we have already explored this state
        if tree is not None:
            leaf = tree.get_leaf(state)
            if leaf.action is not None:
                continue

        # sort constraints according to our current point
        constraints.sort(key=lambda x: -(p[x[0]] - Decimal(str(x[1]))))

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

            # check if we have already explored this state
            explored = False
            if tree is not None:
                explored = [
                    l for l in tree.get_leafs_at_symbolic_state(sym_state, pairs=[])
                    if l.action is not None
                ]

            # check if our new state returns a different action
            state_actions = []
            if not explored:  # but skip
                leafs = root.get_leafs_at_symbolic_state(sym_state, pairs=[])
                state_actions = set([l.action for l in leafs])

            if len(state_actions) > 1 or explored or bound == max_var_vals[variables[var]]:

                # roll back to last state
                if len(state_actions) > 1 or explored:
                    state[variables[var]] = old_val

                # mark variable as exhausted
                exhausted.append(var)

                # continue if we still have unexhausted variables
                if len(exhausted) < len(variables):
                    continue

                # otherwise, store the big box as a leaf
                box = build_state({
                    v: (float(p[var2id[v]] - eps), state[v])
                    for v in variables
                })
                leaf = Leaf(cost=0, action=action, state=box)
                boxes.append(leaf)

                # add to the tree, so we know not to explore this part again
                if tree is not None:
                    tree.put_leaf(leaf, State(variables))
                # or create the tree if we haven't done so yet
                else:
                    tree = Node.make_root_from_leaf(leaf)

                # add new points from which to start later
                for v in variables:
                    if p[var2id[v]] != state[v] and state[v] < max_var_vals[v]:
                        points.append(tuple(
                            float(p[var2id[w]] - eps)
                            if w != v else state[w]
                            for w in variables
                        ))
                break

    return boxes


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


def find_best_cut_2(boxes, variables):

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
            key=lambda x: (boxes[x].state.max[v], boxes[x].state.min[v])
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
            # boxes[curr_l[i]].state.max[v] <= boxes[curr_l[j]].state.max[v]
            # for all j > i)
            max_v = boxes[curr_l[i]].state.max[v]

            if max_v == boxes[curr_l[i+1]].state.max[v]:
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
                if boxes[curr_l[j]].state.min[v] < max_v:
                    impurity += 1

            # add the triplet to our list of possible cuts
            cuts.append((v, i, impurity))

    if len(cuts) == 0:
        assert len(set([l.action for l in boxes]))
        return Leaf(0, action=boxes[0].action)

    # sort according to impurity and take the 'most pure' cut
    v, b_id, _ = sorted(cuts, key=lambda x: x[2])[0]

    # grab that optimal value
    max_val = boxes[max_sorted[v][b_id]].state.max[v]

    # separate the boxes into list of those that are lower than our cut, those
    # that are higher or both if it falls on both sides of the cut
    low, high = [], []
    for b in boxes:
        l, h = split_box(b, v, max_val)
        if l is not None:
            low.append(l)
        if h is not None:
            high.append(h)
        # if b.state.min[v] < max_val:
        #     low.append(b)

        # if b.state.max[v] > max_val:
        #     high.append(b)

    # something went wrong if we end up here
    if len(low) == 0 or len(high) == 0:
        import ipdb; ipdb.set_trace()

    # create the new branch node with a cut on v <= max_val
    node = Node(v, max_val)

    return node, low, high


def find_best_cut(boxes, variables):
    max_sorted = {
        v: sorted(range(len(boxes)), key=lambda x: boxes[x].state.max[v])
        for v in variables
    }

    cuts = []
    for v in variables:
        curr_l = max_sorted[v]
        for i in range(len(curr_l)):
            max_v = boxes[curr_l[i]].state.max[v]
            cut = True
            for j in range(i + 1, len(curr_l)):
                if boxes[curr_l[j]].state.min[v] < max_v:
                    cut = False
                    break

            if cut and i < len(curr_l) - 1:
                cuts.append((v, i))

    if len(cuts) == 0:
        v = variables[random.randint(0, len(variables) - 1)]
        b_id = (len(boxes) // 2) - 1

    else:
        v, b_id = sorted(
            cuts,
            key=lambda x: abs((len(boxes) / 2) - (x[1] + 1))
        )[0]

    low, high = max_sorted[v][:b_id + 1], max_sorted[v][b_id + 1:]
    low, high = [boxes[i] for i in low], [boxes[i] for i in high]
    node = Node(v, boxes[max_sorted[v][b_id]].state.max[v])
    return node, low, high


def cut_to_node(boxes, variables, lvl):
    if len(boxes) == 1:
        return Leaf(0.0, action=boxes[0].action)

    res = find_best_cut_2(boxes, variables)
    if isinstance(res, Leaf):
        return res

    # else
    node, low, high = res
    if len(low) == 3 and len(high) == 1 and lvl == 843:
        import ipdb; ipdb.set_trace()
    node.low = cut_to_node(low, variables, lvl+1)
    node.high = cut_to_node(high, variables, lvl+1)
    return node


def boxes_to_tree(boxes, variables):
    return cut_to_node(boxes, variables, 1).prune()


def draw_grid_lines(d, xk, yk, min_x, min_y, max_x, max_y, conv_x, conv_y, lines=True):
    """
    d: a Drawing object
    """

    x_ticks = [
        x for x in range(min_x + (xk - min_x % xk), max_x - (xk - max_x % xk), xk)
    ]
    for x in x_ticks:
        l = draw.Line(
            conv_x(x), conv_y(min_y), conv_x(x), conv_y(max_y) if lines else conv_y(0.1),
            stroke='black', stroke_width=0.25 if lines else 2
        )
        d.append(l)

    y_ticks = [
        y for y in range(min_y + (yk - min_y % yk), max_y - (yk - max_y % yk), yk)
    ]
    for y in y_ticks:
        l = draw.Line(
            conv_x(min_x), conv_y(y), conv_x(max_x) if lines else conv_x(min_x + 0.1), conv_y(y),
            stroke='black', stroke_width=0.25 if lines else 2
        )
        d.append(l)

    xax = draw.Line(
        conv_x(min_x), conv_y(min_y), conv_x(max_x), conv_y(min_y),
        stroke='black', stroke_width=2
    )
    yax = draw.Line(
        conv_x(0), conv_y(min_y), conv_x(0), conv_y(max_y),
        stroke='black', stroke_width=2
    )
    d.append(draw.Text('velocity', 26, x=(max_x - min_x) / 2, y=2, center=True,
                       fill='blue'))
    d.append(draw.Text('VELOCITY', 10, path=xax, center=True, fill='black'))
    d.append(draw.Text('JUST SOMETHING', 15, x=0, y=100, center=True))
    d.append(xax)
    d.append(yax)


def draw_partition(
    leaves, x_var, y_var, xlim, ylim, cmap,
    dpi=100, lw=0, show=False, out_fp='./tmp.svg'
):
    min_x, max_x = xlim
    min_y, max_y = ylim

    fig, ax = plt.subplots()

    for l in leaves:
        s = l.state
        x_start, x_end = s.min_max(x_var, min_limit=min_x, max_limit=max_x)
        y_start, y_end = s.min_max(y_var, min_limit=min_y, max_limit=max_y)
        width = x_end - x_start
        height = y_end - y_start
        c = cmap[l.action]
        ax.add_patch(
            Rectangle(
                (x_start, y_start), width, height, color=c, ec='black', lw=lw
            )
        )

    plt.xlim(xlim)
    plt.ylim(ylim)

    plt.xlabel(x_var)
    plt.ylabel(y_var)

    if show:
        plt.show()

    if out_fp is not None:
        plt.savefig(out_fp, dpi=dpi)

    plt.close()


def draw_2d_partition(
        leafs, varX, varY,
        min_xy=(0,0), max_padding=1, stroke_width=1, draw_grid=True,
        actions=None, can_width=2400, can_height=1900, out_fp='./tmp.svg'
):
    if actions is None:
        actions = set([l.action for l in leafs])

    boxes_pairs = []
    min_x, min_y, max_x, max_y = math.inf, math.inf, -math.inf, -math.inf
    for leaf in leafs:
        s = leaf.state
        x1, x2 = s.min_max(varX)
        y1, y2 = s.min_max(varY)
        boxes_pairs.append(((x1,y1), (x2,y2)))

        min_x = x1 if x1 > -math.inf and x1 < min_x else min_x
        min_y = y1 if y1 > -math.inf and y1 < min_y else min_y
        max_x = x2 if x2 < math.inf and x2 > max_x else max_x
        max_y = y2 if y2 < math.inf and y2 > max_y else max_y

    min_x, min_y = min_xy
    max_x, max_y = max_x + max_padding, max_y + max_padding
    max_x, max_y = 14.2, 14.5

    boxes = []
    for i, ((x1, y1), (x2, y2)) in enumerate(boxes_pairs):
        x1 = min_x if x1 == -math.inf else x1
        y1 = min_y if y1 == -math.inf else y1

        x2 = max_x if x2 == math.inf else x2
        y2 = max_y if y2 == math.inf else y2

        width = x2 - x1
        height = y2 - y1
        boxes.append((x1, y1, width, height, leafs[i].action))

    state_width = max_x - min_x
    state_height = max_y - min_y

    conv_x = lambda x: (x / state_width) * can_width
    conv_y = lambda y: (y / state_height) * can_height

    origin = (conv_x(min_x), conv_y(min_y))
    d = draw.Drawing(can_width, can_height, origin=origin, displayInline=False)

    a_colors = { a: c for a,c in zip(actions, ['red', 'green', 'blue']) }
    for x, y, w, h, action in boxes:
        cx, cw = map(conv_x, (x, w))
        cy, ch = map(conv_y, (y, h))

        fill = 'green' if action == '1' else 'white'
        fill = a_colors[action]
        r = draw.Rectangle(
            cx, cy, cw, ch,
            fill=fill, stroke='black', stroke_width=stroke_width
        )
        d.append(r)

    if draw_grid:
        draw_grid_lines(
            d, 1, 1, min_x, min_y, math.ceil(max_x), math.ceil(max_y),
            conv_x, conv_y, lines=True
        )
    d.saveSvg(out_fp)


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

    # both children are leafs
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
