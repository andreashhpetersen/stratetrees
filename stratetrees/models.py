import json
import pydot
import pathlib
import operator
import numpy as np

from copy import deepcopy
from numpy.typing import ArrayLike

from stratetrees.loaders import UppaalLoader, SklearnLoader
from stratetrees.nodes import Node, Leaf, State


class MpTree:
    def __init__(self, tree, FMIN=None, FMAX=None):
        self.variables = tree.variables
        self.actions = tree.actions

        self.children_low = []
        self.children_high = []
        self.features = []
        self.values = []
        self.thresholds = []

        self.n_leaves = 0
        self.n_features = len(tree.variables)

        ones = np.ones((self.n_features,))
        if FMIN is not None:
            try:
                self.FMIN = ones * np.array(FMIN)
            except ValueError:
                raise ValueError(
                    f'FMIN must either be scalar or list of size `n_features`'
                )
        else:
            self.FMIN = -ones * np.inf

        if FMAX is not None:
            try:
                self.FMAX = ones * np.array(FMAX)
            except ValueError:
                raise ValueError(
                    f'FMAX must either be scalar or list of size `n_features`'
                )
        else:
            self.FMAX = ones * np.inf

        self._build_tree(tree.root, 0)

    @property
    def size(self):
        return (self.n_leaves * 2) - 1

    def is_leaf(self, node_id):
        return self.children_low[node_id] == self.children_high[node_id]

    def leaves(self):
        return np.argwhere(np.array(self.children_low) < 0).flatten()

    def partitions(self):
        state = np.vstack((
            np.ones((self.n_features,)) * self.FMIN,
            np.ones((self.n_features,)) * self.FMAX
        )).T
        collect = []
        self._get_partitions(0, state, collect)
        return collect

    def _get_partitions(self, node_id, state, collect):
        if self.is_leaf(node_id):
            collect.append((node_id, state))
            return

        f = self.features[node_id]
        t = self.thresholds[node_id]

        lstate = state.copy()
        hstate = state.copy()

        lstate[f,1] = t
        hstate[f,0] = t

        self._get_partitions(self.children_low[node_id], lstate, collect)
        self._get_partitions(self.children_high[node_id], hstate, collect)

    def predict(self, state):
        return self.values[self._predict(state, 0)]

    def predict_node_id(self, state, cheat=False):
        return self._predict(state, 0, cheat)

    def _predict(self, state, node_id, cheat=False):
        if self.is_leaf(node_id):
            return node_id

        op = operator.lt if cheat else operator.le
        if op(state[self.features[node_id]], self.thresholds[node_id]):
            return self._predict(state, self.children_low[node_id], cheat)
        else:
            return self._predict(state, self.children_high[node_id], cheat)

    def predict_for_region(self, min_state, max_state, node_ids=False):
        """
        Predict actions in the area defined by `]min_state, max_state]`. If
        `node_ids` is True, return the node ids in the region instead.

        Parameters
        ----------
        min_state : array_like
            The k-dimensional minimum state
        max_state : array_like
            The k-dimensional maximum state
        node_ids : bool
            Flag to determine wether to return actions or node ids

        Returns
        -------
        predictions : set
            Either a set of actions in the given region or a list of node ids
        """
        res = self._predict_for_region(min_state, max_state, 0, [])
        if node_ids:
            return res
        else:
            return set(self.values[node_id] for node_id in res)

    def _predict_for_region(self, min_state, max_state, node_id, collect):
        if self.is_leaf(node_id):
            collect.append(node_id)
            return collect

        if min_state[self.features[node_id]] < self.thresholds[node_id]:
            collect = self._predict_for_region(
                min_state,
                max_state,
                self.children_low[node_id],
                collect
            )

        if max_state[self.features[node_id]] > self.thresholds[node_id]:
            collect = self._predict_for_region(
                min_state,
                max_state,
                self.children_high[node_id],
                collect
            )

        return collect

    def _build_tree(self, node, node_id):
        self.children_low.append(-1)
        self.children_high.append(-1)
        self.features.append(-1)
        self.values.append(-1)
        self.thresholds.append(-1)

        if node.is_leaf:
            self.values[node_id] = int(node.action)
            self.n_leaves += 1
            return node_id, node_id
        else:
            self.features[node_id] = node.var_id
            self.thresholds[node_id] = node.bound

            low_id, max_id = self._build_tree(node.low, node_id + 1)
            self.children_low[node_id] = low_id

            high_id, max_id = self._build_tree(node.high, max_id + 1)
            self.children_high[node_id] = high_id

            return node_id, max_id


class DecisionTree:
    def __init__(self, root, variables, actions, meta={}):
        self.root = root
        self.variables = variables
        self.var2id = { v: i for i, v in enumerate(variables) }
        self.actions = [ int(a) for a in actions ]
        self.act2id = { a: i for i, a in enumerate(actions) }
        self.meta = meta

        if not self.root is None:
            self.set_state()

    @property
    def size(self):
        if self.root is None:
            return 0

        return self.root.size

    @property
    def n_leaves(self):
        if self.root is None:
            return 0
        elif self.root.is_leaf:
            return 1

        return self.root.n_leaves

    @property
    def max_depth(self):
        return self.root._max_depth

    @property
    def min_depth(self):
        return self.root._min_depth

    def predict(self, state: ArrayLike) -> int:
        """
        Predict the best action based on a `state`.

        Parameters
        ----------
        state : array_like
            Input state

        Returns
        ------
        action_id : int
            The index of the preferred action
        """
        l = self.root.get(state)
        return l.action

    def path_to(self, state):
        path = self.root.path_to(state, [])
        return path

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
        return set(self.root.get_for_region(
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

    def put_leaf(self, leaf, prune=False):
        self.root.put_leaf(leaf, State(self.variables), prune=prune)

    def emp_prune(self, sub_action=None):
        self.root = Node.emp_prune(self.root, sub_action=sub_action)

    def set_state(self):
        self.root.set_state(State(self.variables))

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
        roots = self.root.to_q_trees([self.act2id[a] for a in self.actions])
        meta = self.meta.copy()
        for action, root in roots:
            meta['regressors'][loc]['regressor'][action] = root.to_uppaal(
                self.var2id
            )

        return meta

    def export_to_uppaal(self, filepath, loc='(1)'):
        with open(filepath, 'w') as f:
            json.dump(self.get_uppaal_meta(loc=loc), f, indent=4)

    def export_to_c_code(self, signature='takeAction()', out_fp='tree.c'):
        s = 'int ' + signature + ' {\n '
        s += self.root.to_c_code(1)
        s += '\n}'

        with open(out_fp, 'w') as f:
            f.write(s)

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
    def load_from_file(cls, filepath):
        with open(filepath, 'r') as f:
            data = json.load(f)

        variables = data['variables']
        actions = data['actions']

        var2id = { v: i for i, v in enumerate(variables) }

        root = Node.build_from_dict(data['root'], var2id)
        root.set_state(State(variables))
        return DecisionTree(
            root,
            variables,
            actions,
            meta=data.get('meta', '')
        )

    @classmethod
    def from_sklearn(cls, src, variables=None, actions=None):
        if isinstance(src, str):
            root, variables, actions = SklearnLoader.load(src, variables, actions)
        else:
            loader = SklearnLoader(src, variables, actions)
            root = loader.root
            variables = loader.variables
            actions = loader.actions
        return DecisionTree(root, variables, actions)

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
                nodes.append(Leaf(int(float(label[0]))))
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

    @classmethod
    def from_grid(cls, grid, variables, actions, intervals, bounds, bvars=[]):
        root = DecisionTree._from_grid(grid, intervals, bounds, State(variables), bvars=bvars)
        return DecisionTree(root, variables, actions)

    @classmethod
    def _from_grid(cls, grid, intervals, bounds, state, bvars=[]):
        cut_dim = np.argmax(grid.shape)

        cut_idx = grid.shape[cut_dim] // 2

        if cut_idx == 0:
            return Leaf(grid.item(), state=state)

        if cut_dim in bvars:
            cut_val = 0.5
        else:
            blow, bhigh = bounds[cut_dim]
            smin, smax = state.min_max(cut_dim, min_limit=blow, max_limit=bhigh)
            cut_val = smin + cut_idx * intervals[cut_dim]

        low_state, high_state = state.split(cut_dim, cut_val)

        low_grid = np.take(grid, np.arange(cut_idx), axis=cut_dim)
        high_grid = np.take(grid, np.arange(cut_idx, grid.shape[cut_dim]), axis=cut_dim)

        low_node = DecisionTree._from_grid(low_grid, intervals, bounds, low_state, bvars=bvars)
        high_node = DecisionTree._from_grid(high_grid, intervals, bounds, high_state, bvars=bvars)

        node = Node(cut_dim, cut_dim, cut_val, low_node, high_node, state=state)
        return node


class QTree:
    def __init__(self, path: str):
        roots, actions, variables, meta = UppaalLoader.load(path)
        self.variables = variables
        self.meta = meta

        self.actions = [ int(a) for a in actions ]
        self.act2id = { a: i for i, a in enumerate(actions) }
        self.roots = roots

        for r, a in zip(roots, actions):
            for leaf in r.get_leaves():
                leaf.action = self.act2id[a]

        self._size = sum([r.size for r in self.roots])

    @property
    def size(self):
        """Get size of the tree in the number of leaves"""
        if not hasattr(self, '_size') or self._size is None:
            self._size = sum([r.size for r in self.roots])
        return self._size

    @property
    def n_leaves(self):
        """Get the combined number of leaves of all roots in the QTree"""
        return sum([r.n_leaves for r in self.roots])

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
        state = np.array(state)
        a = len(state.shape) - 1

        qs = self.predict_qs(state)
        return np.argmax(qs, axis=a) if maximize else np.argmin(qs, axis=a)

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
        squeeze = False
        state = np.array(state)
        if len(state.shape) == 1:
            squeeze = True
            state = np.array([state])

        qs = []
        for obs in state:
            qs.append([r.get(obs).cost for r in self.roots])

        qs = np.array(qs)
        if squeeze:
            qs = qs.squeeze()

        return qs

    def to_decision_tree(self) -> DecisionTree:
        """
        Create a decision tree representing a strategy equivalent to greedily
        selecting the smallest Q value for any state on this Q tree.

        Returns
        -------
        dt : DecisionTree
            A decision tree instance.
        """
        leaves = self.leaves()
        leaves.sort(key=lambda x: x.cost)
        root = Node.make_node_from_leaf(leaves[0], self.variables)
        for leaf in leaves[1:]:
            root = root.put_leaf(leaf, State(self.variables), prune=True)

        return DecisionTree(root, self.variables, self.actions, meta=self.meta)

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
