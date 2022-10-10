import json
import math
import pydot
from copy import deepcopy
from random import shuffle


class State:
    def __init__(self, variables):
        self.variables = variables
        self.min = { v: -math.inf for v in variables }
        self.max = { v: math.inf  for v in variables }

    def center(self, point=True):
        """
        Returns the center point of the state box. None if one of the bounds
        are infinite. If `point=False`, returns a dict with variables as
        keys and coordinates as values
        """
        center = []
        vcenters = {}
        for v in self.variables:
            vmin, vmax = self.min_max(v)
            if vmin == -math.inf or vmax == math.inf:
                return None

            vcenter = vmin + ((vmax - vmin) / 2)
            vcenters[v] = vcenter
            center.append(vcenter)

        return center if point else vcenters

    def greater_than(self, var, bound):
        self.min[var] = bound

    def less_than(self, var, bound):
        self.max[var] = bound

    def min_max(self, var, min_limit=-math.inf, max_limit=math.inf):
        """
        Return a tuple of the min and max values of `var` in this state

        params:
            var:        the variable to check
            limit_val:  (optional) the value to return if `var` has no limit
                        (defaults to `math.inf` or `-math.inf`)
        """
        var_min = self.min[var] if self.min[var] != -math.inf else min_limit
        var_max = self.max[var] if self.max[var] != math.inf else max_limit
        return var_min, var_max

    def copy(self):
        state = State(self.variables)
        state.min = self.min.copy()
        state.max = self.max.copy()
        return state

    def contains(self, other):
        for var in self.variables:
            if not (
                self.min[var] <= other.min[var] and
                    self.max[var] >= other.max[var]
            ):
                return False
        return True

    def __str__(self):
        ranges = [
            '{}: [{},{}]'.format(var, self.min[var], self.max[var])
            for var in self.variables
        ]
        return f"State({', '.join(ranges)})"

    def __repr__(self):
        return self.__str__()


class Tree:
    def __init__(self, variables, actions, root=None):
        self.variables = variables
        self.actions = actions
        self.root = root

    def get_paths(self):
        if self.root is None:
            raise ValueError('Tree has no root node')

        return self._get_paths(self.root, ())

    def _get_paths(self, node, path):
        if node.is_leaf:
            return path + (node,)

        path = path + (node,)
        lps = self._get_paths(node.low, path)
        hps = self._get_paths(node.high, path)
        return lps + hps

    def advanced_prune(self):
        assert self.root is not None


        res = self._advanced_prune(self.root, [])
        return res

    def _advanced_prune(self, node, path):
        if node.is_leaf:
            state = { v: [None, None] for v in self.variables }
            for i in range(len(path) - 1, -1, -1):
                n, sub = path[i]
                if state[n.variable][sub] is None:
                    state[n.variable][sub] = n
            return ((state, node.action),)

        lres = self._advanced_prune(node.low, path + [(node, 1)])
        hres = self._advanced_prune(node.high, path + [(node, 0)])

        if len(lres) == len(hres) == 1:
            if lres[0][1] == hres[0][1]:
                pass

        return lres + hres


class Node:
    def __init__(self, variable, bound, low=None, high=None, state=None):
        self.variable = variable
        self.bound = bound
        self.state = state
        self.low = low
        self.high = high
        self.is_leaf = False

    @property
    def size(self):
        return self.count_leaves()

    def count_leaves(self):
        low_count = 1 if self.low.is_leaf else self.low.count_leaves()
        high_count = 1 if self.high.is_leaf else self.high.count_leaves()
        return low_count + high_count

    def put_leaf(self, leaf, state):
        var_min, var_max = leaf.state.min_max(self.variable)
        if var_min < self.bound:
            low_state = state.copy()
            low_state.less_than(self.variable, self.bound)

            self.low = self.low.put_leaf(leaf, low_state)

        if var_max > self.bound:
            high_state = state.copy()
            high_state.greater_than(self.variable, self.bound)

            self.high = self.high.put_leaf(leaf, high_state)

        return self

    def get_leaf(self, state):
        """
        Get a particular leaf corresponding to the given `state`
        """
        if state[self.variable] > self.bound:
            return self.high.get_leaf(state)
        else:
            return self.low.get_leaf(state)

    def get_leaves(self):
        """
        return a list of all leaves of this tree
        """
        leaves = []
        self.low._get_leaves(leaves)
        self.high._get_leaves(leaves)
        return leaves

    def _get_leaves(self, leaves=[]):
        self.low._get_leaves(leaves=leaves)
        self.high._get_leaves(leaves=leaves)

    def get_leaves_at_symbolic_state2(self, min_state, max_state, pairs=[]):
        if min_state[self.variable] < self.bound:
            pairs = self.low.get_leaves_at_symbolic_state2(
                min_state, max_state, pairs
            )

        if max_state[self.variable] > self.bound:
            pairs = self.high.get_leaves_at_symbolic_state2(
                min_state, max_state, pairs
            )

        return pairs

    def get_leaves_at_symbolic_state(self, state, pairs=[]):
        """
        Takes a symbolic state (of type `State`) and returns a list of tuples
        indicating each action/state combination found
        """
        var_min, var_max = state.min_max(self.variable)
        var_min = float(var_min)
        var_max = float(var_max)
        if var_min <= self.bound:
            pairs = self.low.get_leaves_at_symbolic_state(state, pairs)

        if var_max > self.bound:
            pairs = self.high.get_leaves_at_symbolic_state(state, pairs)

        return pairs

    def get_bounds(self):
        bounds = self._get_bounds(bounds={})
        for var, bs in bounds.items():
            bounds[var] = sorted(list(set(bs)))
        return bounds

    def _get_bounds(self, bounds={}):
        if self.variable not in bounds:
            bounds[self.variable] = []
        bounds[self.variable].append(self.bound)

        if not self.low.is_leaf:
            bounds = self.low._get_bounds(bounds)

        if not self.high.is_leaf:
            bounds = self.high._get_bounds(bounds)

        return bounds

    def set_state(self, state):
        """
        set the state of this Node to `state` and call the function on
        child nodes aswell
        """
        self.state = state
        low_state = state.copy()
        low_state.less_than(self.variable, self.bound)

        high_state = state.copy()
        high_state.greater_than(self.variable, self.bound)

        self.low.set_state(low_state)
        self.high.set_state(high_state)

    def prune(self):
        if not self.low.is_leaf:
            self.low = self.low.prune()

        if not self.high.is_leaf:
            self.high = self.high.prune()

        if self.low.is_leaf and self.high.is_leaf:
            if self.low.action == self.high.action:
                return Leaf(
                    max(self.low.cost, self.high.cost),
                    action=self.low.action
                )

        elif not (self.low.is_leaf or self.high.is_leaf):
            if Node.equivalent(self.low, self.high):
                # arbitrary if we here choose high or low
                return self.low

        return self

    def save_as(self, filepath, filetype='json'):
        """
        Save tree as json to a file at `filepath`.

        TODO: support multiple filetypes
        """
        with open(filepath, 'w') as f:
            json.dump(self.as_dict(), f, indent=4)

    def as_dict(self, var_func=lambda x: x):
        """
        Output the tree as a dictionary.
        """
        return {
            'var': var_func(self.variable),
            'bound': self.bound,
            'low': self.low.as_dict(var_func=var_func),
            'high': self.high.as_dict(var_func=var_func)
        }

    def to_uppaal(self, var_map):
        """
        Export to dict with in a way that's ready for UPPAAL json format
        """
        if isinstance(self.low, Leaf):
            low = self.low.cost
        else:
            low = self.low.to_uppaal(var_map)

        if isinstance(self.high, Leaf):
            high = self.high.cost
        else:
            high = self.high.to_uppaal(var_map)

        return {
            'var': var_map[self.variable],
            'bound': self.bound,
            'low': low,
            'high': high
        }

    def to_q_trees(self, actions, MAX_COST=9999, MIN_COST=0):
        """
        Export this decision tree to a forest of Q-trees
        """
        # here we store the resulting (action, root) pairs
        out = []

        # create a root per action
        roots = [deepcopy(self) for _ in actions]

        # generate each tree
        for root, action in zip(roots, actions):
            leaves = root.get_leaves()
            for leaf in leaves:

                # if another action is here, make it very expensive
                if leaf.action != action:
                    leaf.cost = MAX_COST

                # otherwise, make it cheap
                else:
                    leaf.cost = MIN_COST
            out.append((action, root))
        return out

    def export_to_uppaal(
            self, actions, variables, meta, path='./out.json', loc='(1)'
    ):
        """
        Make sure the variables in `variables` is ordered the same way as
        in `meta[pointvars]`
        """
        var_map = { v: i for i, v in enumerate(variables) }
        roots = self.to_q_trees(actions)
        for action, root in roots:
            meta['regressors'][loc]['regressor'][action] = root.to_uppaal(
                var_map
            )

        with open(path, 'w') as f:
            json.dump(meta, f, indent=4)

    def __str__(self):
        return f'Node(var: {self.variable}, bound: {self.bound})'

    def __repr__(self):
        return self.__str__()

    def __copy__(self):
        return type(self)(
            self.variable, self.bound, self.low, self.high, self.state
        )

    def __deepcopy__(self, memo):
        id_self = id(self)
        _copy = memo.get(id_self)
        if _copy is None:
            _copy = type(self)(
                deepcopy(self.variable, memo),
                deepcopy(self.bound, memo),
                deepcopy(self.low, memo),
                deepcopy(self.high, memo),
                deepcopy(self.state, memo)
            )
            memo[id_self] = _copy
        return _copy

    @classmethod
    def load_from_file(cls, filepath):
        with open(filepath, 'r') as f:
            json_root = json.load(f)

        variables, actions = set(), set()
        root = cls.build_from_dict(json_root, variables, actions)
        root.set_state(State(variables))
        return root, list(variables), list(actions)

    @classmethod
    def parse_from_dot(cls, filepath, variables=None):
        graph = pydot.graph_from_dot_file(filepath)[0]

        nodes = []
        for node in graph.get_nodes():
            try:
                int(node.get_name())
            except ValueError:
                continue

            label = node.get_attributes()['label'].strip('"').split(" ")
            if len(label) == 1:
                nodes.append(Leaf(0, action=int(label[0])))
            else:
                var = variables[label[0]] if variables else label[0]
                bound = float(label[2])
                nodes.append(Node(var, bound))

        for edge in graph.get_edges():
            src = nodes[int(edge.get_source())]
            dst = nodes[int(edge.get_destination())]
            low = True if edge.get_label().strip('"') == 'True' else False

            if low:
                src.low = dst
            else:
                src.high = dst

        return nodes[0]

    @classmethod
    def build_from_dict(cls, node_dict, variables, actions):
        """
        Recursively build a tree using the top level in `node_dict` as root and
        keep track of variables and actions
        """

        # is this a leaf?
        if not 'low' in node_dict:
            actions.add(node_dict['action'])
            return Leaf(node_dict['cost'], action=node_dict['action'])

        variables.add(node_dict['var'])
        return Node(
            node_dict['var'],
            node_dict['bound'],
            low=cls.build_from_dict(node_dict['low'], variables, actions),
            high=cls.build_from_dict(node_dict['high'], variables, actions)
        )

    @classmethod
    def make_root_from_leaf(cls, leaf):
        root = None
        last = None
        for var in leaf.state.variables:
            var_min, var_max = leaf.state.min_max(
                var, min_limit=None, max_limit=None
            )
            if var_min is not None:
                new_node = Node(var, var_min)
                new_node.low = Leaf(math.inf, action=None)

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
                new_node = Node(var, var_max)
                new_node.high = Leaf(math.inf, action=None)

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

        root.set_state(State(leaf.state.variables))
        return root

    @classmethod
    def equivalent(cls, node1, node2):
        """
        Test if `node1` and `node2` are equivalent. Requires both nodes to
        have leaves at both their `low` and `high` directions.
        """
        if not (node1.low.is_leaf and node1.high.is_leaf):
            return False

        if not (node2.low.is_leaf and node2.high.is_leaf):
            return False

        if not node1.variable == node2.variable:
            return False

        if not node1.bound == node2.bound:
            return False

        if not node1.low.action == node2.low.action:
            return False

        if not node1.high.action == node2.high.action:
            return False

        return True

    @classmethod
    def get_all_leaves(cls, roots, sort=True):
        leaves = [l for ls in [root.get_leaves() for root in roots] for l in ls]
        if sort:
            leaves.sort(key=lambda x: x.cost)
        return leaves

    @classmethod
    def make_decision_tree_from_leaves(cls, leaves):
        variables = leaves[0].state.variables
        leaves.sort(key=lambda x: x.cost)
        root = Node.make_root_from_leaf(leaves[0])
        for i in range(1, len(leaves)):
            root = root.put_leaf(leaves[i], State(variables))

        return root.prune()

    @classmethod
    def make_decision_tree_from_roots(cls, roots):
        return cls.make_decision_tree_from_leaves(cls.get_all_leaves(roots))

    @classmethod
    def get_var_data(cls, root, variables=None):
        """
        class function to fetch data (`bound` values as of now) for each
        variable in `variables`
        """
        if variables is None:
            try:
                variables = root.variables
            except AttributeError:
                raise ValueError(
                    "argument `variables` cannot be None if argument \
                    `root` has no attribute `variables`"
                )

        data = { v: { 'bounds': [] } for v in variables }
        return cls._get_var_data(root, data)

    @classmethod
    def _get_var_data(cls, tree, data):
        if cls.is_leaf(tree):
            return data

        data[tree.variable]['bounds'].append(tree.bound)
        data = cls._get_var_data(tree.high, data)
        data = cls._get_var_data(tree.low, data)
        return data

    @classmethod
    def is_leaf(cls, tree):
        return isinstance(tree, Leaf)

    @classmethod
    def emp_prune(cls, node, thresh=0.0):
        if isinstance(node, Leaf):
            if node.ratio <= thresh:
                return None
            else:
                return node

        low = cls.emp_prune(node.low, thresh)
        high = cls.emp_prune(node.high, thresh)

        if low is None and high is None:
            return None

        if low is None:
            return high

        if high is None:
            return low

        node.low = low
        node.high = high
        return node.prune()


class Leaf:
    def __init__(self, cost, action=None, state=None):
        self.cost = cost
        self.action = action
        self.state = state
        self.visits = 0
        self.is_leaf = True

    def split(self, variable, bound, state):
        new_node = Node(variable, bound)

        low_state = state.copy()
        low_state.less_than(variable, bound)
        new_node.low = Leaf(self.cost, action=self.action, state=low_state)

        high_state = state.copy()
        high_state.greater_than(variable, bound)
        new_node.high = Leaf(self.cost, action=self.action, state=high_state)

        return new_node

    def put_leaf(self, leaf, state):
        """
        If all variables in `leaf` has been checked, compare cost value to
        determine action. Otherwise, insert Node checking for unchecked
        variables
        """
        if self.cost < leaf.cost:
            return self

        variables = state.variables.copy()
        shuffle(variables)
        for var in variables:
            self_var_min, self_var_max = state.min_max(var)
            leaf_var_min, leaf_var_max = leaf.state.min_max(var)

            if self_var_min < leaf_var_min:
                new_node = self.split(var, leaf_var_min, state)
                return new_node.put_leaf(leaf, state)

            if self_var_max > leaf_var_max:
                new_node = self.split(var, leaf_var_max, state)
                return new_node.put_leaf(leaf, state)

        # all variables are checked
        return Leaf.copy(leaf)

    @classmethod
    def copy(cls, leaf):
        """
        Returns a new Leaf that is a copy of `leaf`
        """
        return Leaf(leaf.cost, action=leaf.action, state=leaf.state.copy())

    def __copy__(self):
        return type(self)(self.cost, self.action, self.state.copy)

    def __deepcopy__(self, memo):
        id_self = id(self)
        _copy = memo.get(id_self)
        if _copy is None:
            _copy = type(self)(
                deepcopy(self.cost, memo),
                deepcopy(self.action, memo),
                deepcopy(self.state, memo)
            )

            if hasattr(self, 'visits'):
                _copy.visits = self.visits
            if hasattr(self, 'ratio'):
                _copy.ratio = self.ratio

            memo[id_self] = _copy
        return _copy

    def _get_leaves(self, leaves=[]):
        leaves.append(self)

    def get_leaf(self, state):
        return self

    def get_leaves_at_symbolic_state2(self, min_state, max_state, pairs=[]):
        """
        Append the local state of the leaf to `pairs` and return the list
        """
        pairs.append(self)
        return pairs

    def get_leaves_at_symbolic_state(self, state, pairs=[]):
        """
        Append the local state of the leaf to `pairs` and return the list
        """
        pairs.append(self)
        return pairs

    def get_leaves_at_symbolic_state(self, state, pairs=[]):
        """
        Append the local state of the leaf to `pairs` and return the list
        """
        pairs.append(self)
        return pairs

    def set_state(self, state):
        self.state = state

    def as_dict(self, var_func=None):
        return {
            'action': self.action,
            'cost': self.cost
        }

    def __str__(self):
        return f'Leaf(action: {self.action}, cost: {self.cost}, {self.state})'

    def __repr__(self):
        return self.__str__()
