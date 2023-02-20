import json
import numpy as np

from joblib import load
from copy import deepcopy
from collections import defaultdict

from trees.nodes import Node, Leaf, State


class UppaalLoader:

    @classmethod
    def load(cls, path: str) -> (list[Node], list[str], list[str], str):
        with open(path, 'r') as f:
            data = json.load(f)

        locations = sorted(data['regressors'].keys())
        loc_state = {
            location: list(map(int, location[1:-1].split(',')))
            for location in locations
        }
        variables = data['statevars'] + data['pointvars']
        S = len(data['statevars'])

        action_location_roots = defaultdict(dict)
        for location, loc_trees in data['regressors'].items():
            for action, tree in loc_trees['regressor'].items():
                root = cls._build_tree(tree, variables, S)
                root.set_state(State(variables))
                action_location_roots[action][location] = root

        actions = sorted(action_location_roots.keys())
        if S == 0:
            roots = [ action_location_roots[a]['(1)'] for a in actions ]
            return roots, actions, variables, cls._get_uppaal_data(data)

        org_root = None
        for loc in locations:
            org_root = cls._put_loc(org_root, loc_state[loc], data['statevars'], 0)

        roots = []
        for action, location_roots in action_location_roots.items():
            root = deepcopy(org_root)
            for loc, tree in location_roots.items():
                cls._put_tree(root, loc_state[loc], Leaf(0, action=tree))

            cls._fix_tree(root, action)
            root.set_state(State(variables))
            roots.append(root)

        return roots, actions, variables, cls._get_uppaal_data(data)

    @classmethod
    def _fix_tree(cls, node, action):
        if not node.low.is_leaf:
            node.low = cls._fix_tree(node.low, action)
        elif isinstance(node.low.action, Node):
            node.low = node.low.action

        if not node.high.is_leaf:
            node.high = cls._fix_tree(node.high, action)
        elif isinstance(node.high.action, Node):
            node.high = node.high.action

        if node.low.is_leaf and node.high.is_leaf:
            return Leaf(np.inf, action=action)
        else:
            return node

    @classmethod
    def _put_loc(cls, node, loc, names, i):
        if node is None or node.is_leaf:
            if i < len(loc):
                node = Node(names[i], i, loc[i], low=Leaf(np.inf), high=Leaf(np.inf))
                return cls._put_loc(node, loc, names, i + 1)
            else:
                return node

        if loc[node.var_id] <= node.bound:
            node.low = cls._put_loc(node.low, loc, names, max(node.var_id, i))
        else:
            node.high = cls._put_loc(node.high, loc, names, max(node.var_id, i))

        return node

    @classmethod
    def _put_tree(cls, node, loc, tree):
        if node.is_leaf:
            return tree

        if loc[node.var_id] <= node.bound:
            node.low = cls._put_tree(node.low, loc, tree)
        else:
            node.high = cls._put_tree(node.high, loc, tree)

        return node

    @classmethod
    def _get_uppaal_data(cls, data):
        for reg in data['regressors']:
            data['regressors'][reg]['regressor'] = {}

        data['pointvars'] = data['statevars'] + data['pointvars']
        data['statevars'] = []
        data['regressors'] = {
            '(1)': {
                'type': 'act->point->val',
                'representation': 'simpletree',
                'minimize': 1,
                'regressor': {}
            }
        }
        return data

    @classmethod
    def _build_tree(cls, tree, variables, S=0):
        if isinstance(tree, float) or isinstance(tree, int):
            return Leaf(cost=float(tree))

        return Node(
            variables[tree['var'] + S],
            tree['var'] + S,  # var_id
            tree['bound'],
            low=cls._build_tree(tree['low'], variables, S),
            high=cls._build_tree(tree['high'], variables, S)
        )


class SklearnLoader:

    def __init__(self, path: str, actions, variables):
        self.variables = variables
        self.actions = actions

        clf = self.load_classifier(path)
        self.children_left = clf.tree_.children_left
        self.children_right = clf.tree_.children_right
        self.features = clf.tree_.feature
        self.bounds = clf.tree_.threshold
        self.values = clf.tree_.value

        self.root = self.build_root(0)
        self.root.set_state(State(variables))

    def build_root(self, node_id: int):
        is_branch = self.children_left[node_id] != self.children_right[node_id]
        if is_branch:
            return Node(
                self.variables[self.features[node_id]],
                self.features[node_id],
                self.bounds[node_id],
                low=self.build_root(self.children_left[node_id]),
                high=self.build_root(self.children_right[node_id]),
            )
        else:
            act_id = np.argmax(self.values[node_id])
            return Leaf(0, action=self.actions[act_id], act_id=act_id)

    def load_classifier(self, path):
        return load(path)

    @classmethod
    def load(cls, path, actions, variables):
        loader = SklearnLoader(path, actions, variables)
        return loader.root
