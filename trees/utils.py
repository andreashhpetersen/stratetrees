import pydot
import csv
import json
import numpy as np
from collections import defaultdict

from trees.models import Node, Leaf, State

####### Functions to add color gradients #######

def hex_to_RGB(hex):
  """ "#FFFFFF" -> [255,255,255] """
  # Pass 16 to the integer function for change of base
  return [int(hex[i:i+2], 16) for i in range(1,6,2)]

def RGB_to_hex(RGB):
  """ [255,255,255] -> "#FFFFFF" """
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

def _draw_graph(graph, tree, n, print_action=False):
    if isinstance(tree, Leaf):
        label = ''
        font_color = 'black'
        color = 'white'
        if print_action:
            label += f'Action: {tree.action}\n'
        label += f'Cost: {round(tree.cost, 2)}'
        if hasattr(tree, 'ratio'):
            label += f'\nFreq: {round(tree.ratio * 100, 2)}'
            color = color_gradient(tree.ratio, start="#FFFFFF", end="#FF0000")
            if tree.ratio == 0:
                color = 'black'
        if hasattr(tree, 'best_ratio'):
            label += f'\nbest: {round(tree.best_ratio * 100, 2)}'
        node = pydot.Node(
            str(n), label=label,
            shape='circle', fillcolor=color, fontcolor=font_color, style='filled'
        )
        graph.add_node(node)
        return node, n

    try:
        low_node, low_n = _draw_graph(graph, tree.low, n, print_action=print_action)
    except AttributeError:
        import ipdb; ipdb.set_trace()
        pass
    high_node, high_n = _draw_graph(graph, tree.high, low_n + 1, print_action=print_action)

    new_n = high_n + 1
    label = f'{tree.variable}: {round(tree.bound, 2)}'
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
        node, n = _draw_graph(graph, tree, n, print_action=print_action)
        graph.add_edge(pydot.Edge('actions', str(n), label=label))
        n += 1

    graph.write_png(out_fp)


####### Functions to load and add statistics to a tree #######

def load_stat(fp):
    with open(fp, 'r') as f:
        reader = csv.reader(f)
        variables, data = [], []
        next(reader)

        for row in reader:
            if len(row) == 1:
                # TODO: get actual var name
                variables.append(row[0])
                data.append([])
            else:
                data[-1].append((float(row[0]), float(row[1])))

    for i in range(len(data)):
        data[i] = np.array(data[i])

    return data, variables

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
    if isinstance(tree, float):
        return Leaf(tree, action=a)

    return Node(
        tree['var'] if variables is None else variables[tree['var']],
        tree['bound'],
        low=build_tree(tree['low'], a, variables),
        high=build_tree(tree['high'], a, variables)
    )

def load_tree(fp, loc='(1)', stats=[]):
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

    if len(stats) > 0:
        add_stats(roots, stats)

    return roots, variables, actions
