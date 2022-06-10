import json
import math
import random
import drawSvg as draw
from decimal import *
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
        p = tuple(Decimal(str(x)) + Decimal(str(eps)) for x in p)

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

            # check if our new state returns a different action
            leafs = root.get_leafs_at_symbolic_state(sym_state, pairs=[])
            state_actions = set([l.action for l in leafs])

            # check if we have already explored this state
            explored = False
            if tree is not None:
                # try:
                explored = [
                    l for l in tree.get_leafs_at_symbolic_state(sym_state, pairs=[])
                    if l.action is not None
                ]
                # except:
                    # import ipdb; ipdb.set_trace()

            if len(state_actions) > 1 or explored or bound == math.inf:

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
                    v: (float(p[var2id[v]] - Decimal(str(eps))), state[v]) for v in variables
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
                            float(p[var2id[w]] - Decimal(str(eps))) if w != v else state[w]
                            for w in variables
                        ))
                break

    return boxes


def find_best_cut_2(boxes, variables):
    max_sorted = {
        v: sorted(
            range(len(boxes)),
            key=lambda x: (boxes[x].state.max[v], boxes[x].state.min[v])
        )
        for v in variables
    }

    cuts = []
    for v in variables:
        curr_l = max_sorted[v]
        for i in range(len(curr_l)):
            max_v = boxes[curr_l[i]].state.max[v]
            impurity = abs((len(boxes) / 2) - (i + 1))
            for j in range(i + 1, len(curr_l)):
                if boxes[curr_l[j]].state.min[v] < max_v:
                    impurity += 1

            cuts.append((v, i, impurity))

    v, b_id, _ = sorted(cuts, key=lambda x: x[2])[0]

    max_val = boxes[max_sorted[v][b_id]].state.max[v]

    low, high = [], []
    for b in boxes:
        if b.state.min[v] < max_val:
            low.append(b)

        if b.state.max[v] > max_val:
            high.append(b)

    if len(low) == 0 or len(high) == 0:
        import ipdb; ipdb.set_trace()

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


def cut_to_node(boxes, variables):
    if len(boxes) == 1:
        return Leaf(0.0, action=boxes[0].action)

    node, low, high = find_best_cut_2(boxes, variables)
    node.low = cut_to_node(low, variables)
    node.high = cut_to_node(high, variables)
    return node


def boxes_to_tree(boxes, variables):
    return cut_to_node(boxes, variables)


def draw_grid(d, xk, yk, min_x, min_y, max_x, max_y, conv_x, conv_y, lines=True):
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
    d.append(xax)
    d.append(yax)


def draw_2d_partition(
        leafs, varX, varY,
        min_xy=(0,0), max_padding=1, stroke_width=1,
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

    draw_grid(
        d, 1, 1, min_x, min_y, math.ceil(max_x), math.ceil(max_y),
        conv_x, conv_y, lines=True
    )
    d.saveSvg(out_fp)
