import json
import pydot
import numpy as np

from copy import deepcopy
from random import shuffle
from numpy.typing import ArrayLike

from trees.loaders import UppaalLoader
from trees.nodes import Node, Leaf, State


class DecisionTree:
    def __init__(self, root, variables, actions, size=None, meta={}):
        self.root = root
        self.variables = variables
        self.var2id = { v: i for i, v in enumerate(variables) }
        self.actions = actions
        self.act2id = { a: i for i, a in enumerate(actions) }
        self.size = size
        self.meta = meta

    def predict(self, state: ArrayLike, maximize=False) -> int:
        """
        Predict the best action based on a `state`.

        Parameters
        ----------
        state : array_like
            Input state
        maximize : bool, optional
            If set to True, return the action with the largest Q value

        Returns
        ------
        action_id : int
            The index of the preferred action
        """
        l = self.root.get(state)
        return self.act2id[l.action]

    def count_visits(self, data: list[ArrayLike]) -> None:
        """
        Count the number of visits to each leaf node when evaluating the states
        in `data` on the tree. Typically used as a preprocessing step before
        calling `Tree.emp_prune()`.

        Parameters
        ----------
        data : list of lists
            The sample states to evaluate.
        """
        leaves = self.leaves()
        for l in leaves:
            l.visits = 0

        for state in data:
            leaf = self.root.get(state)
            leaf.visits += 1

        for l in leaves:
            l.ratio = l.visits / len(data)

    def get_for_region(self, min_state, max_state, actions=True):
        return set(self.act2id[a] for a in self.root.get_for_region(
            min_state, max_state, actions=actions, collect=set()
        ))

    def get_bounds(self):
        bounds = self.root.get_bounds([set() for _ in self.variables])
        return [sorted(list(vbounds)) for vbounds in bounds]

    def leaves(self) -> list[Leaf]:
        """
        Get the list of all leaves in the tree.

        Returns
        -------
        leaves : list
            The list of leaves in the tree
        """
        return self.root.get_leaves()

    def put_leaf(self, leaf):
        self.root.put_leaf(leaf, State(self.variables))

    def emp_prune(self, sub_action=None):
        self.root = Node.emp_prune(self.root, sub_action=sub_action)
        self.size = self.root.size

    def set_depth(self):
        self.root.set_depth()
        depths = np.array([l.depth for l in self.leaves()])
        self.max_depth = depths.max()
        self.avg_depth = depths.mean()

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
        return DecisionTree(None, variables, actions)

    @classmethod
    def build_from_leaves(cls, leaves, actions, variables, meta=None):
        tree = DecisionTree.empty_tree(variables, actions)
        leaves.sort(key=lambda x: x.cost)
        root = DecisionTree.make_root_from_leaf(tree, leaves[0])
        for i in range(1, len(leaves)):
            root = root.put_leaf(leaves[i], State(variables))

        tree.root = root.prune()
        tree.size = root.size
        tree.meta = meta
        return tree

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
        return DecisionTree(
            root,
            variables,
            actions,
            size=root.size,
            meta=data.get('meta', '')
        )

    @classmethod
    def parse_from_dot(cls, filepath, varmap):
        tree = DecisionTree.empty_tree(list(varmap.values()), [])

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
        self.meta = meta

        self._size = sum([r.size for r in self.roots])

    @property
    def size(self):
        """Get size of the tree in the number of leaves"""
        if not hasattr(self, '_size') or self._size is None:
            self._size = sum([r.size for r in self.roots])
        return self._size

    def predict(self, state: ArrayLike, maximize=False) -> int:
        """
        Predict the best action based on a `state`.

        Parameters
        ----------
        state : array_like
            Input state
        maximize : bool, optional
            If set to True, return the action with the largest Q value

        Returns
        ------
        action_id : int
            The index of the preferred action
        """
        qs = self.predict_qs(state)
        return np.argmax(qs) if maximize else np.argmin(qs)

    def predict_qs(self, state: ArrayLike) -> np.ndarray:
        """
        Predict the Q values for each action given `state`.

        Parameters
        ----------
        state : array_like
            Input state

        Returns
        -------
        q_values : numpy.ndarry
            An array of Q values for each action
        """
        return np.array([r.get(state).cost for r in self.roots])

    def to_decision_tree(self) -> DecisionTree:
        """
        Create a decision tree representing a strategy equivalent to greedily
        selecting the smallest Q value for any state on this Q tree.

        Returns
        -------
        dt : DecisionTree
            A decision tree instance.
        """
        return DecisionTree.build_from_leaves(
            self.leaves(), self.actions, self.variables, meta=self.meta
        )

    def leaves(self, sort='min') -> list[Leaf]:
        """
        Get all the leaves of the roots of this Q tree.

        Parameters
        ----------
        sort : str
            If set to `'min'` (default), sort the leaves ascendingly according
            to cost. If set to `'max'` sort descendingly. Otherwise, no sorting
            is applied.

        Returns
        -------
        leaves : list
            The list of all leaves in the Q tree.
        """
        leaves = [l for ls in [r.get_leaves() for r in self.roots] for l in ls]
        if sort == 'min':
            leaves.sort(key=lambda x: x.cost)
        elif sort == 'max':
            leaves.sort(key=lambda x: -x.cost)
        return leaves
