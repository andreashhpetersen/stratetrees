import csv
import json
import pydot
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import smc2py

from copy import deepcopy
from collections import defaultdict
from matplotlib.patches import Rectangle

from trees.models import Node, Leaf, State

####### Functions to add color gradients #######

def hex_to_RGB(hex):
    """
    #FFFFFF -> [255,255,255]
    """

    # Pass 16 to the integer function for change of base
    return [int(hex[i:i+2], 16) for i in range(1,6,2)]

def RGB_to_hex(RGB):
    """
    [255,255,255] -> #FFFFFF
    """
    # Components need to be integers for hex to make sense
    RGB = [int(x) for x in RGB]
    return "#"+"".join(["0{0:x}".format(v) if v < 16 else
              "{0:x}".format(v) for v in RGB])

def color_gradient(weight, start="#FFFFFF", end="#000000"):
    s = hex_to_RGB(start)
    f = hex_to_RGB(end)
    return RGB_to_hex([
        int(s[j] + (weight * (f[j] - s[j]))) for j in range(3)
    ])


####### Functions to draw a tree and save it to png #######

def draw_leaf(
    graph, leaf, n,
    print_action=False, print_cost=False, print_ratio=False
):
    label = ''
    font_color = 'black'
    color = 'white'

    if print_action:
        if hasattr(leaf, 'verbose_action'):
            label += f'Action: {leaf.verbose_action}\n'
        else:
            label += f'Action: {leaf.action}\n'
    if print_cost:
        label += f'Cost: {round(leaf.cost, 2)}'
    if print_ratio and hasattr(leaf, 'ratio'):
        label += f'\nFreq: {round(leaf.ratio * 100, 2)}'
        color = color_gradient(leaf.ratio, start="#FFFFFF", end="#FF0000")
        if leaf.ratio == 0:
            color = 'black'
    if hasattr(leaf, 'best_ratio'):
        label += f'\nbest: {round(leaf.best_ratio * 100, 2)}'

    node = pydot.Node(
        str(n), label=label,
        shape='circle', fillcolor=color, fontcolor=font_color, style='filled'
    )
    graph.add_node(node)
    return node, n

def draw_node(graph, root, n, print_action=False, print_cost=False):
    if root.is_leaf:
        return draw_leaf(
            graph, root, n, print_action=print_action, print_cost=print_cost
        )

    low_node, low_n = draw_node(
        graph, root.low, n, print_action=print_action, print_cost=print_cost
    )
    high_node, high_n = draw_node(
        graph, root.high, low_n + 1, print_action=print_action,
        print_cost=print_cost
    )

    new_n = high_n + 1
    label = f'{root.variable}: {round(root.bound, 2)}'
    node = graph.add_node(
        pydot.Node(str(new_n), label=label, shape='square')
    )


    graph.add_edge(pydot.Edge(str(new_n), str(low_n), label='low'))
    graph.add_edge(pydot.Edge(str(new_n), str(high_n), label='high'))
    return node, new_n

def draw_graph(
    trees, labels=None, out_fp='graph_drawing.png', print_action=True,
    print_cost=False
):
    graph = pydot.Dot(graph_type='digraph')
    root = pydot.Node('root')
    iterator = zip(labels, trees) if labels is not None else enumerate(trees)

    n = 0
    for label, tree in iterator:
        node, n = draw_node(
            graph, tree, n, print_action=print_action, print_cost=print_cost
        )
        if len(trees) > 1:
            graph.add_edge(pydot.Edge('action', str(n), label=label))
            n += 1

    if out_fp.endswith('.dot'):
        graph.write_dot(out_fp)
    elif out_fp.endswith('.png'):
        graph.write_png(out_fp)
    elif out_fp.endswith('.svg'):
        graph.write_svg(out_fp)
    else:
        print('format not supported')

def draw_partitioning(
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



####### Functions to load and add statistics to a tree #######

def parse_from_sampling_log(filepath, as_numpy=True):
    """
    Return data as a list (or as a `np.array` if `as_numpy=True`) of floats
    parsed from a log file (of the format [timestep, var1, var2, ...])
    """
    # import ipdb; ipdb.set_trace()
    # data = smc2py.parseEngineOuput(filepath)
    with open(filepath, 'r') as f:
        data = f.readlines()

    data = [list(map(float, s.strip().split(','))) for s in data]
    if as_numpy:
        data = np.array(data)
    # print(data)
    # exit(0);
    return data

def test_equivalence(tree, forest, data, variables, step=1):
    for i in range(0, len(data), step):
        state = { var: val for var, val in zip(variables, data[i][1:]) }
        a1 = tree.get_leaf(state).action
        a2 = sorted(
            [r.get_leaf(state) for r in forest], key=lambda x: x.cost
        )[0].action
        if not a1 == a2:
            print(f'Inconsistent at {state}')
            print(f'Tree chose {a1}, forest chose {a2}')
            return False
    return True

def count_visits(tree, data, step=1):
    """
    Evaluate every state given by `data`, and mark the resulting leaf in `root`
    as visited. Counts the total number of visits and the frequency each leaf
    was visited with. Note that the variables in every entry in `data` must
    occur in the same order as the do in `variables`.

    - params:
        - `root`, the root `Node` of the strategy
        - `data`, a list of lists, where each entry is of the form
            `[t, var1_t, ..., varN_t]` for a setting with N variables in
            the state at each time `t`
        - `variables`, a list of the variable names in the order they occur in
            each of `data[t]`
        - `step` (default 1), the stepsize when iterating `data`

    - returns:
        a dictionary counting the times each action was chosen
    """

    leaves = tree.get_leaves()
    for l in leaves:
        l.visits = 0
        l.ratio = 0

    total = 0
    actions = defaultdict(int)
    last_a = None
    for i in range(0, len(data), step):
        state = data[i]
        leaf = tree.get(state, leaf=True)
        leaf.visits += 1
        actions[leaf.action] += 1
        total += 1
        last_a = leaf.action

    for l in leaves:
        l.ratio = l.visits / total

    return actions

def add_stats(trees, stats, variables, max_ts, step_sz):
    """

    params:
        stat -  a list of statistics, where each statistic should consist of
                a list of tuples of (timestep, value) for each variable in the
                statistic
    """
    actions = defaultdict(int)
    leaves = [l for ls in [t.get_leaves() for t in trees] for l in ls]
    for l in leaves:
        l.visits = 0
        l.ratio = 0
        l.best = 0
        l.best_ratio = 0

    for stat in stats:
        var_ptrs = [0 for _ in range(len(variables))]

        for ts in range(0, max_ts, step_sz):
            state = {}
            for var in range(len(variables)):
                var_data = stat[var]

                # increase pointer if next timestep is larger
                if not ts < var_data[var_ptrs[var]][0]:
                    var_ptrs[var] += 1

                # update state with value at pointer
                state[variables[var]] = var_data[var_ptrs[var]][1]

            # increase visited
            visited = sorted(
                [tree.get_leaf(state) for tree in trees],
                key=lambda x: x.cost
            )
            best = visited[0]
            visited[0].best += 1
            actions[visited[0].action] += 1

            for l in visited:
                l.visits += 1

    total = len(stats) * (max_ts / step_sz)
    for l in leaves:
        try:
            l.ratio = l.visits / total
        except ZeroDivisionError:
            l.ratio = 0

        try:
            l.best_ratio = l.best / l.visits
        except ZeroDivisionError:
            l.best_ratio = 0

    return actions


####### Function to load and build tree #######

def build_tree(tree, a, variables, S=0):
    if isinstance(tree, float) or isinstance(tree, int):
        return Leaf(float(tree), action=a)

    return Node(
        variables[tree['var'] + S],
        tree['var'] + S,  # var_id
        tree['bound'],
        low=build_tree(tree['low'], a, variables, S),
        high=build_tree(tree['high'], a, variables, S)
    )

def get_uppaal_data(data):
    for reg in data['regressors']:
        data['regressors'][reg]['regressor'] = {}
    return data

def load_trees(fp, loc='(1)', verbosity=0):
    """
    If `verbosity=1`, also return a dict with the UPPAAL specific strategy data
    (relevant for later exporting back to the UPPAAL forma). Default 0.
    """
    with open(fp, 'r') as f:
        data = json.load(f)

    variables = data['pointvars']
    roots = []
    trees = data['regressors'][loc]['regressor']
    actions = []
    for action, tree in trees.items():
        root = build_tree(tree, action, variables)
        root.set_state(State(variables))
        roots.append(root)
        actions.append(action)

    if verbosity > 0:
        misc = get_uppaal_data(data)
        return roots, variables, actions, misc

    return roots, variables, actions

def import_uppaal_strategy(fp):
    with open(fp, 'r') as f:
        data = json.load(f)

    variables = data['pointvars']
    actions = list(data['actions'].keys())
    regressors = data['regressors']
    locations = regressors.keys()

    trees_at_location = {}
    for location in locations:
        qtrees = regressors[location]['regressor']
        roots = {}
        for action, qtree in qtrees.items():
            root = build_tree(qtree, action, variables)
            root.set_state(State(variables))
            roots[action] = root

        trees_at_location[location] = roots

    meta = get_uppaal_data(data)
    return trees_at_location, variables, actions, meta

def import_uppaal_strategy2(fp):
    with open(fp, 'r') as f:
        data = json.load(f)

    actions = list(data['actions'].keys())
    variables = data['statevars'] + data['pointvars']
    S = len(data['statevars'])

    locations = data['regressors'].keys()

    locs = [ list(map(int, l[1:-1].split(','))) for l in locations ]
    locs.sort()

    root = None
    for loc in locs:
        root = put_loc(root, loc, data['statevars'], 0)

    org_root = deepcopy(root)

    # map locations to a list of possible actions
    loc2actions = {
        loc: list(data['regressors'][loc]['regressor'])
        for loc in locations
    }

    # map actions to a dictionary of q-trees at each location
    act2loc_trees = {
        a: {
            loc: data['regressors'][loc]['regressor'][a]
        } for loc in locations for a in loc2actions[loc]
    }

    roots = []
    for action, loc_trees in act2loc_trees.items():
        root = deepcopy(org_root)
        i = 0
        for location in loc_trees:
            tree = build_tree(loc_trees[location], action, variables, S)
            loc = list(map(int, location[1:-1].split(',')))

            put_tree(root, loc, Leaf(0, action=tree))
            i += 1

        fix_tree(root, action)
        root.set_state(State(variables))
        roots.append(root)

    meta = get_uppaal_data(data)
    meta['statevars'] = []
    meta['pointvars'] = variables
    meta['regressors'] = {
        '(1)': {
            'type': 'act->point->val',
            'representation': 'simpletree',
            'minimize': 1,
            'regressor': {}
        }
    }
    return roots, variables, actions, meta

def fix_tree(node, action):
    if not node.low.is_leaf:
        node.low = fix_tree(node.low, action)
    elif isinstance(node.low.action, Node):
        node.low = node.low.action

    if not node.high.is_leaf:
        node.high = fix_tree(node.high, action)
    elif isinstance(node.high.action, Node):
        node.high = node.high.action

    if node.low.is_leaf and node.high.is_leaf:
        return Leaf(np.inf, action=action)
    else:
        return node

def put_loc(node, loc, names, i):
    if node is None or node.is_leaf:
        if i < len(loc):
            node = Node(names[i], i, loc[i], low=Leaf(np.inf), high=Leaf(np.inf))
            return put_loc(node, loc, names, i + 1)
        else:
            return node

    if loc[node.var_id] <= node.bound:
        node.low = put_loc(node.low, loc, names, max(node.var_id, i))
    else:
        node.high = put_loc(node.high, loc, names, max(node.var_id, i))

    return node

def put_tree(node, loc, tree):
    if node.is_leaf:
        return tree

    if loc[node.var_id] <= node.bound:
        node.low = put_tree(node.low, loc, tree)
    else:
        node.high = put_tree(node.high, loc, tree)

    return node
