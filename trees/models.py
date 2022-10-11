import json
import math
import pydot
import numpy as np
from copy import deepcopy
from random import shuffle


class State:
    def __init__(self, variables, constraints=None):
        self.variables = variables
        self.var2id = { v: i for i, v in enumerate(variables) }

        if constraints is None:
            constraints = np.vstack(
                (
                    np.ones((len(variables),)) * -np.inf,
                    np.ones((len(variables),)) * np.inf,
                )
            ).T
        self.constraints = constraints

    def min(self, var):
        if isinstance(var, str):
            var = self.var2id[var]
        return self.constraints[var,0]

    def max(self, var):
        if isinstance(var, str):
            var = self.var2id[var]
        return self.constraints[var,1]

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
        if isinstance(var, str):
            var = self.var2id[var]
        self.constraints[var,0] = bound

    def less_than(self, var, bound):
        if isinstance(var, str):
            var = self.var2id[var]
        self.constraints[var,1] = bound

    def min_max(self, var, min_limit=-math.inf, max_limit=math.inf):
        """
        Return a np.arry of the min and max values of `var` in this state

        params:
            var:        the variable to check
            limit_val:  (optional) the value to return if `var` has no limit
                        (defaults to `math.inf` or `-math.inf`)
        """
        if isinstance(var, str):
            var = self.var2id[var]
        vbounds = self.constraints[var]
        vmin = vbounds[0] if vbounds[0] > -np.inf else min_limit
        vmax = vbounds[1] if vbounds[0] < np.inf else max_limit
        return vmin, vmax

    def copy(self):
        state = State(self.variables, constraints=self.constraints.copy())
        return state

    def __str__(self):
        ranges = [
            '{}: [{},{}]'.format(var, self.min(var), self.max(var))
            for var in self.variables
        ]
        return f"State({', '.join(ranges)})"

    def __repr__(self):
        return self.__str__()


class Tree:
    def __init__(self, root, variables, actions, size=None):
        self.root = root
        self.variables = variables
        self.var2id = { v: i for i, v in enumerate(variables) }
        self.actions = actions
        self.act2id = { a: i for i, a in enumerate(actions) }
        self.size = size

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

    def get_leaves(self):
        return self.root.get_leaves()

    def put_leaf(self, leaf):
        self.root.put_leaf(leaf, State(self.variables))

    def save_as(self, filepath):
        """
        Save tree as json to a file at `filepath`.
        """
        data = {
            'variables': self.variables,
            'actions': self.actions,
            'root': self.root.as_dict()
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)

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

        tree.root = root.prune()
        tree.size = root.size
        return tree

    @classmethod
    def build_from_roots(cls, roots, variables, actions):
        return cls.build_from_leaves(cls.get_all_leaves(roots))

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
                new_node = Node(var, tree.var2id[var], var_max)
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
        return Tree(root, variables, actions, size=root.size)


class Node:
    def __init__(self, variable, var_id, bound, low=None, high=None, state=None):
        self.variable = variable
        self.bound = bound
        self.state = state
        self.low = low
        self.high = high
        self.is_leaf = False

        self.var_name = variable
        self.var_id = var_id

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

    def get(self, state):
        if state[self.var_id] <= self.bound:
            return self.low.get(state)
        else:
            return self.high.get(state)

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

    def get_for_region(self, min_state, max_state, actions, collect):
        if min_state[self.var_id] < self.bound:
            collect = self.low.get_for_region(
                min_state, max_state, actions, collect
            )

        if max_state[self.var_id] > self.bound:
            collect = self.high.get_for_region(
                min_state, max_state, actions, collect
            )

        return collect

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

    def get_bounds(self, collect):
        collect[self.var_id].add(self.bound)

        if not self.low.is_leaf:
            collect = self.low.get_bounds(collect)

        if not self.high.is_leaf:
            collect = self.high.get_bounds(collect)

        return collect

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
        return f'Node(var: {self.var_name}, bound: {self.bound})'

    def __repr__(self):
        return self.__str__()

    def __copy__(self):
        return type(self)(
            self.variable, self.var_id, self.var_name, self.bound,
            self.low, self.high, self.state
        )

    def __deepcopy__(self, memo):
        id_self = id(self)
        _copy = memo.get(id_self)
        if _copy is None:
            _copy = type(self)(
                deepcopy(self.variable, memo),
                deepcopy(self.var_id, memo),
                deepcopy(self.var_name, memo),
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
    def build_from_dict(cls, node_dict, var2id, act2id):
        """
        Recursively build a tree using the top level in `node_dict` as root and
        keep track of variables and actions
        """

        # is this a leaf?
        if not 'low' in node_dict:
            action = node_dict['action']
            return Leaf(node_dict['cost'], action=action, act_id=act2id[action])

        var = node_dict['var']
        return Node(
            var,
            var2id[var],
            node_dict['bound'],
            low=cls.build_from_dict(node_dict['low'], var2id, act2id),
            high=cls.build_from_dict(node_dict['high'], var2id, act2id)
        )

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
    def __init__(self, cost, action=None, act_id=None, state=None):
        self.cost = cost
        self.action = action
        self.act_id = act_id
        self.state = state
        self.visits = 0
        self.is_leaf = True

    def split(self, variable, bound, state):
        new_node = Node(variable, state.var2id[variable], bound)

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

    def get(self, *args):
        return self

    def get_leaf(self, state):
        return self

    def get_for_region(self, s1, s2, actions, collect):
        """
        Add this `Leaf' (or its associated action if `actions=True') to the set
        `collect'.
        """
        if self.action is None:
            return collect

        if actions:
            collect.add(self.action)
        else:
            collect.add(self)
        return collect

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

    def _get_leaves(self, leaves=[]):
        leaves.append(self)

    def set_state(self, state):
        self.state = state

    def as_dict(self, var_func=None):
        return {
            'action': self.action,
            'cost': self.cost
        }

    @classmethod
    def copy(cls, leaf):
        """
        Returns a new Leaf that is a copy of `leaf`
        """
        return Leaf(leaf.cost, action=leaf.action, state=leaf.state.copy())

    def __copy__(self):
        return type(self)(self.cost, self.action, self.act_id, self.state.copy)

    def __deepcopy__(self, memo):
        id_self = id(self)
        _copy = memo.get(id_self)
        if _copy is None:
            _copy = type(self)(
                deepcopy(self.cost, memo),
                deepcopy(self.action, memo),
                deepcopy(self.act_id, memo),
                deepcopy(self.state, memo)
            )

            if hasattr(self, 'visits'):
                _copy.visits = self.visits
            if hasattr(self, 'ratio'):
                _copy.ratio = self.ratio

            memo[id_self] = _copy
        return _copy

    def __str__(self):
        return f'Leaf(action: {self.action}, cost: {self.cost}, {self.state})'

    def __repr__(self):
        return self.__str__()
