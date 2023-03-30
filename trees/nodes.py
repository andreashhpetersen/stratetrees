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

    def greater_than(self, var, bound):
        if isinstance(var, str):
            var = self.var2id[var]
        self.constraints[var,0] = bound

    def less_than(self, var, bound):
        if isinstance(var, str):
            var = self.var2id[var]
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
        state = State(self.variables, constraints=self.constraints.copy())
        return state

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
    def __init__(self, variable, var_id, bound, low=None, high=None, state=None):
        self.variable = variable
        self.bound = bound
        self.state = state
        self.low = low
        self.high = high
        self.is_leaf = False

        self.var_name = variable
        self.var_id = var_id

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
        # self.low.is_leaf, self.high.is_leaf = self.low.is_leaf, self.high.is_leaf

        if not self.low.is_leaf:
            self.low = self.low.prune(cost_prune=cost_prune)

        if not self.high.is_leaf:
            self.high = self.high.prune(cost_prune=cost_prune)

        if self.low.is_leaf and self.high.is_leaf and self.low.action == self.high.action:
            if not cost_prune or self.low.cost == self.high.cost:
                return Leaf(
                    max(self.low.cost, self.high.cost),
                    action=self.low.action
                )

        elif not (self.low.is_leaf or self.high.is_leaf):
            if Node.equivalent(self.low, self.high):
                # arbitrary if we here choose high or low
                return self.low

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
                    leaf.action = action
                    leaf.cost = MAX_COST

                # otherwise, make it cheap
                else:
                    leaf.cost = MIN_COST
            root.prune(cost_prune=True)
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

    def copy(self):
        return self.__deepcopy__()

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
    def build_from_dict(cls, node_dict, var2id):
        """
        Recursively build a tree using the top level in `node_dict` as root and
        keep track of variables and actions
        """

        # is this a leaf?
        if not 'low' in node_dict:
            action = node_dict['action']
            return Leaf(cost=node_dict.get('cost', -1), action=action)

        var = node_dict['var']
        return Node(
            var,
            var2id[var],
            node_dict['bound'],
            low=cls.build_from_dict(node_dict['low'], var2id),
            high=cls.build_from_dict(node_dict['high'], var2id)
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
