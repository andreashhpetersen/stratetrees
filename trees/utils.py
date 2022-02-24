import pydot
import csv
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.cluster import KMeans

from trees.models import Node, Leaf, State

def build_tree(tree, a, variables=None):
    if isinstance(tree, float):
        return Leaf(tree, action=a)

    return Node(
        tree['var'] if variables is None else variables[tree['var']],
        tree['bound'],
        low=build_tree(tree['low'], a, variables),
        high=build_tree(tree['high'], a, variables)
    )

def make_state(tree, state, leafs):
    if isinstance(tree, Leaf):
        tree.state = state
        leafs.append(tree)
        return leafs

    state_low = state.copy()
    state_high = state.copy()

    state_low.less_than(tree.variable, tree.bound)
    leafs = make_state(tree.low, state_low, leafs)

    state_high.greater_than(tree.variable, tree.bound)
    return make_state(tree.high, state_high, leafs)

def combine_leafs(l1, l2):
    s1, s2 = l1.state, l2.state
    variables = s1.variables

    s = State(variables)
    for var in variables:
        s.less_than(var, min(s1.max[var], s2.max[var]))
        s.greater_than(var, max(s1.min[var], s1.min[var]))

    return (s, (l1, l2))

def find_compatible(l1, ls):
    out = []
    for l2 in ls:
        if l1.state.compatible(l2.state):
            out.append(l2)

    return out

def plot_clusters(data, k=4):
    fig, axs = plt.subplots(len(data), 1)
    for ax, var in enumerate(data.keys()):
        X = np.array(data[var]['bounds']).reshape(-1,1)
        model = KMeans(n_clusters=k)
        model.fit(X)

        clusters = model.predict(X)
        for c in range(k):
            axs[ax].scatter(
                X[clusters == c],
                [10 for _ in range(len(X[clusters == c]))],
                label=var,
                marker='x'
            )
        axs[ax].get_yaxis().set_visible(False)
        axs[ax].grid(axis='x')
        axs[ax].set_title(var)
    plt.tight_layout()
    plt.show()

def _draw_graph(graph, tree, n):
    if isinstance(tree, Leaf):
        label = f'Cost: {round(tree.cost, 2)}'
        if hasattr(tree, 'ratio'):
            label += f'\nFreq: {round(tree.ratio * 100, 2)}'
        if hasattr(tree, 'best_ratio'):
            label += f'\nbest: {round(tree.best_ratio * 100, 2)}'
        node = pydot.Node(
            str(n), label=label, shape='circle'
        )
        graph.add_node(node)
        return node, n

    low_node, low_n = _draw_graph(graph, tree.low, n)
    high_node, high_n = _draw_graph(graph, tree.high, low_n + 1)

    new_n = high_n + 1
    label = f'{tree.variable}: {round(tree.bound, 2)}'
    node = graph.add_node(
        pydot.Node(str(new_n), label=label, shape='square')
    )

    graph.add_edge(pydot.Edge(str(new_n), str(low_n), label='low'))
    graph.add_edge(pydot.Edge(str(new_n), str(high_n), label='high'))
    return node, new_n

def draw_graph(trees, labels=None, out_fp='graph_drawing.png'):
    graph = pydot.Dot(graph_type='digraph')
    root = pydot.Node('root')
    iterator = zip(labels, trees) if labels is not None else enumerate(trees)

    n = 0
    for label, tree in iterator:
        node, n = _draw_graph(graph, tree, n)
        graph.add_edge(pydot.Edge('actions', str(n), label=label))
        n += 1

    graph.write_png(out_fp)

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

def merge_state(s1, s2, maximize=True):
    assert s1.variables == s2.variables
    new_state = State(s1.variables)
    for var in new_state.variables:
        if maximize:
            new_state.less_than(var, max(s1.max[var], s2.max[var]))
            new_state.greater_than(var, min(s1.min[var], s2.min[var]))
        else:
            new_state.less_than(var, min(s1.max[var], s2.max[var]))
            new_state.greater_than(var, max(s1.min[var], s2.min[var]))

    return new_state

def intersects(s1, s2):
    assert s1.variables == s2.variables
    variables = s1.variables

    for var in variables:
        if s1.min[var] < s2.max[var] < s1.max[var]:
            return True

        if s1.min[var] < s2.min[var] < s1.max[var]:
            return True
    return False

def cut_leaf(l1, l2):
    """
    Cuts the intersecting part of `l2.state` out of `l1.state` and returns the
    list of new leafs

    We probably should build a tree within the space defined by the state of
    `l1`. Make a root Node with state `l1.state` and then `.put` values from
    `l2.state`
    """
    s1, s2 = l1.state, l2.state
    if not intersects(s1, s2):
        return []

    variables = s1.variables

    nodes = [Node(None, None, state=s1.copy())]
    for var in variables:
        if s1.min[var] < s2.min[var] < s1.max[var]:
            nodes[len(nodes)-1].variable = var
            nodes[len(nodes)-1].bound = s2.min[var]

            new_state = s1.copy()
            new_state.less_than(s2.min[var])
            nodes[len(nodes)-1].low = Leaf(l1.cost, action=l1.action, state=new_state)

        if s1.min[var] < s2.max[var] < s1.max[var]:
            pass

def prune_leafs(leafs):
    """
    Prunes all leafs in `leafs` that are dominated by a lower cost leaf

    params:
        leafs:  a list of `Leaf`s, sorted by their cost
    """
    leafs = leafs.copy()
    start = 0
    while start < len(leafs):
        p = leafs[start]
        for leaf in leafs[start+1:]:
            if p.state.contains(leaf.state):
                leafs.remove(leaf)

        start += 1

    return leafs

def deep_pruning(leafs):
    """
    Supposed to go backward through a list of leafs sorted according to cost,
    and check each leaf if its entirely covered by lower valued leafs. Expects
    that for any two leafs, l_1 and l_2, it holds that if cost(l_1) < cost(l_2)
    then l_2 is not covered by l_1.

    However, it currently does not work
    """

    leafs = leafs.copy()

    start = len(leafs) - 1
    while start > 0:
        p = leafs[start]
        istate = None

        for i in range(start - 1, -1, -1):
            leaf = leafs[i]
            if intersects(leaf.state, p.state):
                if istate is None:
                    istate = leaf.state.copy()
                else:
                    istate = merge_state(leaf.state, istate)

            if istate is not None and istate.contains(p.state):
                leafs.remove(p)
                break

        start -= 1

    return leafs
