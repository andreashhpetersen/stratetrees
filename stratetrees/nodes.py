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
            if vmin == -np.inf or vmax == np.inf:
                return None

            vcenter = vmin + ((vmax - vmin) / 2)
            vcenters[v] = vcenter
            center.append(vcenter)

        return center if point else vcenters

    def split(self, var, bound):
        if isinstance(var, str):
            var = self.var2id[var]

        return (self.less_than(var, bound, inline=False), \
            self.greater_than(var, bound, inline=False))

    def greater_than(self, var, bound, inline=True):
        if isinstance(var, str):
            var = self.var2id[var]

        if not inline:
            state = self.copy()
            state.constraints[var, 0] = bound
            return state
        else:
            self.constraints[var,0] = bound

    def less_than(self, var, bound, inline=True):
        if isinstance(var, str):
            var = self.var2id[var]

        if not inline:
            state = self.copy()
            state.constraints[var, 1] = bound
            return state
        else:
            self.constraints[var,1] = bound

    def min_max(self, var, min_limit=-np.inf, max_limit=np.inf):
        """
        Return a np.arry of the min and max values of `var` in this state

        params:
            var:        the variable to check
            limit_val:  (optional) the value to return if `var` has no limit
                        (defaults to `np.inf` or `-np.inf`)
        """
        if isinstance(var, str):
            var = self.var2id[var]
        vbounds = self.constraints[var]
        vmin = vbounds[0] if vbounds[0] > -np.inf else min_limit
        vmax = vbounds[1] if vbounds[1] < np.inf else max_limit
        return vmin, vmax

    def copy(self):
        return State(self.variables, constraints=self.constraints.copy())

    def __eq__(self, other):
        return self.variables == other.variables and \
            (self.constraints == other.constraints).all()

    def __getitem__(self, indicies):
        return self.constraints[indicies]

    def __str__(self):
        return f'Variables: {self.variables},\n{self.constraints}'

    def __repr__(self):
        return str(self.constraints)


class Node:
    def __init__(self, variable, var_id, bound, low, high, state=None):
        self.variable = variable
        self.bound = bound
        self.state = state
        self.low = low
        self.high = high
        self.is_leaf = False

        self.var_name = variable
        self.var_id = var_id

        if low is not None and high is not None:
            self._update_size_and_depth()

    def set_depth(self, depth=0):
        self.depth = depth

        if self.low.is_leaf:
            self.low.depth = depth + 1
        else:
            self.low.set_depth(depth + 1)

        if self.high.is_leaf:
            self.high.depth = depth + 1
        else:
            self.high.set_depth(depth + 1)

    @property
    def size(self):
        return self._size

    @property
    def n_leaves(self):
        return (self.size + 1) // 2

    def visitor(self, func, *args):
        func(self, *args)

        if not self.low.is_leaf:
            self.low.visitor(func, *args)

        if not self.high.is_leaf:
            self.high.visitor(func, *args)

    def count_leaves(self):
        low_count = 1 if self.low.is_leaf else self.low.count_leaves()
        high_count = 1 if self.high.is_leaf else self.high.count_leaves()
        return low_count + high_count

    def put_leaf(self, leaf, state, prune=False):
        var_min, var_max = leaf.state.min_max(self.variable)
        if var_min < self.bound:
            low_state = state.copy()
            low_state.less_than(self.variable, self.bound)

            self.low = self.low.put_leaf(leaf, low_state, prune=prune)

        if var_max > self.bound:
            high_state = state.copy()
            high_state.greater_than(self.variable, self.bound)

            self.high = self.high.put_leaf(leaf, high_state, prune=prune)

        if prune:
            if self.low.is_leaf and self.high.is_leaf and \
                    self.low.action == self.high.action:

                return Leaf(
                    self.low.action,
                    cost=max(self.low.cost, self.high.cost),
                )

        self._size = 1 + self.low._size + self.high._size
        self._max_depth = 1 + max(self.low._max_depth, self.high._max_depth)
        self._min_depth = 1 + min(self.low._min_depth, self.high._min_depth)
        return self

    def get(self, state):
        if state[self.var_id] <= self.bound:
            return self.low.get(state)
        else:
            return self.high.get(state)

    def path_to(self, state, path):
        if state[self.var_id] <= self.bound:
            child = self.low
            n = 'l'
        else:
            child = self.high
            n = 'h'

        path.append((self.var_id, self.bound, n))
        if child.is_leaf:
            path.append(('end', child.action))
            return path
        else:
            return child.path_to(state, path)

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

    def prune(self, cost_prune=False):
        if not self.low.is_leaf:
            self.low = self.low.prune(cost_prune=cost_prune)

        if not self.high.is_leaf:
            self.high = self.high.prune(cost_prune=cost_prune)

        if self.low.is_leaf and self.high.is_leaf and self.low.action == self.high.action:
            if not cost_prune or self.low.cost == self.high.cost:
                return Leaf(
                    self.low.action,
                    cost=max(self.low.cost, self.high.cost)
                )

        elif not (self.low.is_leaf or self.high.is_leaf):
            if Node.equivalent(self.low, self.high):
                # arbitrary if we here choose high or low
                return self.low

        self._update_size_and_depth()
        return self

    def as_dict(self, var_func=lambda x: x):
        """
        Output the tree as a dictionary.
        """
        return {
            'var': var_func(self.variable),
            'bound': float(self.bound),
            'low': self.low.as_dict(var_func=var_func),
            'high': self.high.as_dict(var_func=var_func)
        }

    def to_uppaal(self, var_map):
        """
        Export to dict with in a way that's ready for UPPAAL json format
        """
        if self.low.is_leaf:
            low = self.low.cost
        else:
            low = self.low.to_uppaal(var_map)

        if self.high.is_leaf:
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
                    leaf.action = action
                    leaf.cost = MAX_COST

                # otherwise, make it cheap
                else:
                    leaf.cost = MIN_COST
            root.prune(cost_prune=True)
            out.append((action, root))
        return out

    def to_c_code(self, lvl):
        tab = '  ' * lvl
        s = '{}if ({} <= {}) {}\n'.format(
            tab, self.var_name, self.bound, '{'
        )
        if self.low.is_leaf:
            s += '{}return {};\n'.format(tab + '  ', self.low.action)
        else:
            s += self.low.to_c_code(lvl + 1)

        s += '{}{} else {}\n'.format(tab, '}', '{')
        if self.high.is_leaf:
            s += '{}return {};\n'.format(tab + '  ', self.high.action)
        else:
            s += self.high.to_c_code(lvl + 1)

        s += '{}{}\n'.format(tab, '}')
        return s

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

    def copy(self):
        return self.__deepcopy__()

    def _update_size_and_depth(self):
        self._size = 1 + self.low._size + self.high._size
        self._max_depth = 1 + max(self.low._max_depth, self.high._max_depth)
        self._min_depth = 1 + min(self.low._min_depth, self.high._min_depth)

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
    def build_from_dict(cls, node_dict, var2id):
        """
        Recursively build a tree using the top level in `node_dict` as root and
        keep track of variables and actions
        """

        # is this a leaf?
        if not 'low' in node_dict:
            action = node_dict['action']
            return Leaf(action, cost=node_dict.get('cost', -1))

        var = node_dict['var']
        return Node(
            var,
            var2id[var],
            node_dict['bound'],
            cls.build_from_dict(node_dict['low'], var2id),
            cls.build_from_dict(node_dict['high'], var2id)
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
    def emp_prune(cls, node, sub_action=None, thresh=0.0):
        if node.is_leaf:
            if node.ratio <= thresh:
                if sub_action is None:
                    return None
                else:
                    node.action = sub_action
            return node

        low = cls.emp_prune(node.low, sub_action=sub_action, thresh=thresh)
        high = cls.emp_prune(node.high, sub_action=sub_action, thresh=thresh)

        if low is None and high is None:
            return None

        if low is None:
            return high

        if high is None:
            return low

        node.low = low
        node.high = high
        return node.prune()

    @classmethod
    def make_node_from_leaf(cls, leaf, variables):
        """
        Create the node(s) required to represent `leaf` and return the root
        """
        leaf = leaf.copy()
        vmap = { v: i for i, v in enumerate(variables) }
        branches = []

        for var in variables[::-1]:
            var_min, var_max = leaf.state.min_max(
                var, min_limit=None, max_limit=None
            )
            if var_min is not None:
                branches.append((var, var_min, True))

            if var_max is not None:
                branches.append((var, var_max, False))

        var, bound, is_lower = branches.pop(0)
        nl = Leaf(None, cost=np.inf)
        low, high = (nl, leaf) if is_lower else (leaf, nl)

        new_node = Node(var, vmap[var], bound, low, high)

        for var, bound, is_lower in branches:
            nl = Leaf(None, cost=np.inf)
            low, high = (nl, new_node) if is_lower else (new_node, nl)
            new_node = Node(var, vmap[var], bound, low, high)

        new_node.set_state(State(variables))
        return new_node


class Leaf:
    def __init__(self, action, cost=0, act_id=None, state=None):
        self.cost = cost
        self.action = action
        self.act_id = act_id
        self.state = state
        self.visits = 0
        self.is_leaf = True

        self._size = 1
        self._max_depth = 0
        self._min_depth = 0

    @property
    def size(self):
        return self._size

    def split(self, v, bound, state):
        """
        Split leaf in two and return the parent branching node that splits on
        `v <= bound`
        """
        low_state, high_state = state.split(v, bound)
        low = Leaf(self.action, cost=self.cost, state=low_state)
        high = Leaf(self.action, cost=self.cost, state=high_state)

        v_id = state.var2id[v]
        return Node(v, v_id, bound, low, high, state=state)

    def put_leaf(self, leaf, state, prune=False):
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
                return new_node.put_leaf(leaf, state, prune=prune)

            if self_var_max > leaf_var_max:
                new_node = self.split(var, leaf_var_max, state)
                return new_node.put_leaf(leaf, state, prune=prune)

        # all variables are checked
        return leaf.copy()

    def get(self, *args):
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

    def _get_leaves(self, leaves=[]):
        leaves.append(self)

    def set_state(self, state):
        self.state = state

    def as_dict(self, var_func=None):
        return {
            'action': self.action,
            'cost': self.cost
        }

    def copy(self):
        return Leaf(self.action, cost=self.cost, state=self.state.copy())

    def __copy__(self):
        return type(self)(
            self.action,
            cost=self.cost,
            act_id=self.act_id,
            state=self.state.copy
        )

    def __deepcopy__(self, memo):
        id_self = id(self)
        _copy = memo.get(id_self)
        if _copy is None:
            _copy = type(self)(
                deepcopy(self.action, memo),
                deepcopy(self.cost, memo),
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
        return f'Leaf(action: {self.action}, {self.state})'

    def __repr__(self):
        return self.__str__()
