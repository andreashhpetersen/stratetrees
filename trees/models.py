import json
import pydot
import numpy as np

from copy import deepcopy
from random import shuffle

from trees.loaders import UppaalLoader
from trees.nodes import Node, Leaf, State


class Tree:
    def __init__(self, root, variables, actions, size=None, meta={}):
        self.root = root
        self.variables = variables
        self.var2id = { v: i for i, v in enumerate(variables) }
        self.actions = actions
        self.act2id = { a: i for i, a in enumerate(actions) }
        self.size = size
        self.meta = meta

    def set_depth(self):
        self.root.set_depth()
        leaves = self.get_leaves()
        self.max_depth = max([l.depth for l in leaves])
        self.avg_depth = sum([l.depth for l in leaves]) / len(leaves)

    def get(self, state, leaf=False):
        """
        Get the action/decision associated with `state'. If `leaf=True', the
        entire `Leaf' is returned.
        """
        l = self.root.get(state)
        if leaf:
            return l
        return l.action

    def get_for_region(self, min_state, max_state, actions=True):
        return self.root.get_for_region(
            min_state, max_state, actions=actions, collect=set()
        )

    def get_bounds(self):
        bounds = self.root.get_bounds([set() for _ in self.variables])
        return [sorted(list(vbounds)) for vbounds in bounds]

    def get_branches(self):
        nodes = []
        self.root.get_branches(nodes)
        return nodes

    def get_leaves(self):
        return self.root.get_leaves()

    def put_leaf(self, leaf):
        self.root.put_leaf(leaf, State(self.variables))

    def emp_prune(self, sub_action=None):
        self.root = Node.emp_prune(self.root, sub_action=sub_action)
        self.size = self.root.size

    def save_as(self, filepath):
        """
        Save tree as json to a file at `filepath`.
        """
        data = {
            'variables': self.variables,
            'actions': self.actions,
            'root': self.root.as_dict(),
            'meta': self.meta
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)

    def get_uppaal_meta(self, loc='(1)'):
        roots = self.root.to_q_trees(self.actions)
        meta = self.meta.copy()
        for action, root in roots:
            try:
                meta['regressors'][loc]['regressor'][action] = root.to_uppaal(
                    self.var2id
                )
            except:
                import ipdb; ipdb.set_trace()

        return meta

    def export_to_uppaal(self, filepath, loc='(1)'):
        with open(filepath, 'w') as f:
            json.dump(self.get_uppaal_meta(loc=loc), f, indent=4)

    def copy(self):
        return deepcopy(self)

    def __copy__(self):
        return type(self)(
            self.root, self.variables, self.actions, self.size, self.meta
        )

    def __deepcopy__(self, memo):
        id_self = id(self)
        _copy = memo.get(id_self)
        if _copy is None:
            _copy = type(self)(
                deepcopy(self.root, memo),
                deepcopy(self.variables, memo),
                deepcopy(self.actions, memo),
                deepcopy(self.size, memo),
                deepcopy(self.meta, memo),
            )
            memo[id_self] = _copy
        return _copy

    @classmethod
    def empty_tree(cls, variables, actions):
        """
        Initialize and return an empty tree with given `variables' and
        `actions'
        """
        return Tree(None, variables, actions)

    @classmethod
    def get_all_leaves(cls, roots, sort=True):
        leaves = [l for ls in [root.get_leaves() for root in roots] for l in ls]
        if sort:
            leaves.sort(key=lambda x: x.cost)
        return leaves

    @classmethod
    def build_from_leaves(cls, leaves, variables, actions):
        tree = Tree.empty_tree(variables, actions)
        leaves.sort(key=lambda x: x.cost)
        root = Tree.make_root_from_leaf(tree, leaves[0])
        for i in range(1, len(leaves)):
            root = root.put_leaf(leaves[i], State(variables))

        # import ipdb; ipdb.set_trace()
        tree.root = root.prune()
        tree.size = root.size
        return tree

    @classmethod
    def merge_qtrees(cls, qtrees, variables, actions):
        return cls.build_from_leaves(
            cls.get_all_leaves(qtrees), variables, actions
        )

    @classmethod
    def make_root_from_leaf(cls, tree, leaf):
        root = None
        last = None
        for var in tree.variables:
            var_min, var_max = leaf.state.min_max(
                var, min_limit=None, max_limit=None
            )
            if var_min is not None:
                new_node = Node(var, tree.var2id[var], var_min)
                new_node.low = Leaf(np.inf, action=None)

                if last is None:
                    last = new_node
                else:
                    if last.low is None:
                        last.low = new_node
                    else:
                        last.high = new_node

                    last = new_node

                if root is None:
                    root = last

            if var_max is not None:
                new_node = Node(var, tree.var2id[var], var_max)
                new_node.high = Leaf(np.inf, action=None)

                if last is None:
                    last = new_node
                else:
                    if last.low is None:
                        last.low = new_node
                    else:
                        last.high = new_node

                    last = new_node

                if root is None:
                    root = last

        if last.low is None:
            last.low = leaf
        else:
            last.high = leaf

        root.set_state(State(tree.variables))
        return root

    @classmethod
    def load_from_file(cls, filepath):
        with open(filepath, 'r') as f:
            data = json.load(f)

        variables = data['variables']
        actions = data['actions']

        var2id = { v: i for i, v in enumerate(variables) }
        act2id = { a: i for i, a in enumerate(actions) }

        root = Node.build_from_dict(data['root'], var2id, act2id)
        root.set_state(State(variables))
        return Tree(
            root,
            variables,
            actions,
            size=root.size,
            meta=data.get('meta', '')
        )

    @classmethod
    def parse_from_dot(cls, filepath, varmap):
        tree = Tree.empty_tree(list(varmap.values()), [])

        graph = pydot.graph_from_dot_file(filepath)[0]

        nodes = []
        for node in graph.get_nodes():
            try:
                int(node.get_name())
            except ValueError:
                continue

            label = node.get_attributes()['label'].strip('"').split("<=")
            label = list(map(lambda x: x.strip(' '), label))
            if len(label) == 1:
                nodes.append(Leaf(0, action=int(float(label[0]))))
            else:
                var = varmap[label[0]]
                var_id = tree.var2id[var]
                bound = float(label[1])
                nodes.append(Node(var, var_id, bound))

        for edge in graph.get_edges():
            src = nodes[int(edge.get_source())]
            dst = nodes[int(edge.get_destination())]
            low = True if edge.get_label().strip('"') == 'True' else False

            if low:
                src.low = dst
            else:
                src.high = dst

        tree.root = nodes[0]
        tree.size = tree.root.size
        return tree


class QTree:
    def __init__(self, path: str):
        roots, actions, variables, meta = UppaalLoader.load(path)
        self.act2id = { a: i for i, a in enumerate(actions) }
        self.actions = actions
        self.roots = roots
        self.variables = variables

    def predict(self, state: np.ndarray):
        return self.actions[np.argmin(self.predict_qs(state))]

    def predict_qs(self, state: np.ndarray) -> np.ndarray:
        return np.array([r.get(state).cost for r in self.roots])
