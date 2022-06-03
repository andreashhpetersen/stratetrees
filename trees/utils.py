import pydot
import csv
import json
import numpy as np
from collections import defaultdict

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

def draw_leaf(graph, leaf, n, print_action=False):
    label = ''
    font_color = 'black'
    color = 'white'

    if print_action:
        label += f'Action: {leaf.action}\n'
    label += f'Cost: {round(leaf.cost, 2)}'
    if hasattr(leaf, 'ratio'):
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

def draw_node(graph, root, n, print_action=False):
    if isinstance(root, Leaf):
        return draw_leaf(graph, root, n, print_action=print_action)

    low_node, low_n = draw_node(graph, root.low, n, print_action=print_action)
    high_node, high_n = draw_node(graph, root.high, low_n + 1, print_action=print_action)

    new_n = high_n + 1
    label = f'{root.variable}: {round(root.bound, 2)}'
    node = graph.add_node(
        pydot.Node(str(new_n), label=label, shape='square')
    )


    graph.add_edge(pydot.Edge(str(new_n), str(low_n), label='low'))
    graph.add_edge(pydot.Edge(str(new_n), str(high_n), label='high'))
    return node, new_n

def draw_graph(trees, labels=None, out_fp='graph_drawing.png', print_action=False):
    graph = pydot.Dot(graph_type='digraph')
    root = pydot.Node('root')
    iterator = zip(labels, trees) if labels is not None else enumerate(trees)

    n = 0
    for label, tree in iterator:
        node, n = draw_node(graph, tree, n, print_action=print_action)
        graph.add_edge(pydot.Edge('actions', str(n), label=label))
        n += 1

    graph.write_png(out_fp)


####### Functions to load and add statistics to a tree #######

def parse_from_sampling_log(filepath, as_numpy=True):
    """
    Return data as a list (or as a `np.array` if `as_numpy=True`) of floats
    parsed from a log file (of the format [timestep, var1, var2, ...])
    """
    with open(filepath, 'r') as f:
        data = f.readlines()

    data = [list(map(float, s.strip().split(' '))) for s in data]
    if as_numpy:
        data = np.array(data)

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

def count_visits(root, data, variables, step=1):
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

    leafs = root.get_leafs()
    for l in leafs:
        l.visits = 0
        l.ratio = 0

    total = 0
    actions = defaultdict(int)
    last_a = None
    for i in range(0, len(data), step):
        state = { var: val for var, val in zip(variables, data[i][1:]) }
        leaf = root.get_leaf(state)
        leaf.visits += 1
        actions[leaf.action] += 1
        total += 1
        # if leaf.action == '1':
        #     print('Sample: {}\nState: {}\nAction: {}'.format(
        #         data[i], state, leaf.action
        #     ))
            # import ipdb; ipdb.set_trace()
        last_a = leaf.action

    for l in leafs:
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
    leafs = [l for ls in [t.get_leafs() for t in trees] for l in ls]
    for l in leafs:
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
    for l in leafs:
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

def build_tree(tree, a, variables=None):
    if isinstance(tree, float) or isinstance(tree, int):
        return Leaf(float(tree), action=a)

    return Node(
        tree['var'] if variables is None else variables[tree['var']],
        tree['bound'],
        low=build_tree(tree['low'], a, variables),
        high=build_tree(tree['high'], a, variables)
    )

def get_uppaal_data(data):
    for reg in data['regressors']:
        data['regressors'][reg]['regressor'] = {}
    return data

def load_tree(fp, loc='(1)', verbosity=0):
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
