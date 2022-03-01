import math


class State:
    def __init__(self, variables):
        self.variables = variables
        self.min = { v: -math.inf for v in variables }
        self.max = { v: math.inf  for v in variables }

    def greater_than(self, var, bound):
        self.min[var] = bound

    def less_than(self, var, bound):
        self.max[var] = bound

    def min_max(self, var, limit_val='default'):
        """
        Return a tuple of the min and max values of `var` in this state

        params:
            var:        the variable to check
            limit_val:  (optional) the value to return if `var` has no limit
                        (defaults to `math.inf` or `-math.inf`)
        """
        if limit_val != 'default':
            var_min = self.min[var] if self.min[var] != -math.inf else limit_val
            var_max = self.max[var] if self.max[var] != math.inf else limit_val
            return var_min, var_max
        else:
            return self.min[var], self.max[var]

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


class Node:
    def __init__(self, variable, bound, low=None, high=None, state=None):
        self.variable = variable
        self.bound = bound
        self.state = state
        self.low = low
        self.high = high

    @classmethod
    def make_root_from_leaf(cls, leaf):
        root = None
        last = None
        for var in leaf.state.variables:
            var_min, var_max = leaf.state.min_max(var, limit_val=None)
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

        return root


    def _make_root_from_leaf(cls, node, leaf):
        pass

    def put_leaf(self, leaf, state):
        var_min, var_max = leaf.state.min_max(self.variable)
        if var_min < self.bound:
            state = state.copy()
            state.less_than(self.variable, self.bound)
            self.low = self.low.put_leaf(leaf, state)

        if var_max > self.bound:
            state = state.copy()
            state.greater_than(self.variable, self.bound)
            self.high = self.high.put_leaf(leaf, state)

        return self

    def get_leafs(self):
        """
        return a list of all leafs of this tree
        """
        ls = []
        self._get_leafs(ls)
        return ls

    def _get_leafs(self, ls):
        self.low._get_leafs(ls)
        self.high._get_leafs(ls)

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

    def get_leaf(self, state):
        """
        Get a particular leaf corresponding to the given `state`
        """
        if state[self.variable] > self.bound:
            return self.high.get_leaf(state)
        else:
            return self.low.get_leaf(state)

    def __str__(self):
        return f'Node(var: {self.variable}, bound: {self.bound})'

    def __repr__(self):
        return self.__str__()


class Leaf:
    def __init__(self, cost, action=None, state=None):
        self.cost = cost
        self.action = action
        self.state = state
        self.visits = 0

    def put_leaf(self, leaf, state):
        """
        If all variables in `leaf` has been checked, compare cost value to
        determine action. Otherwise, insert Node checking for unchecked
        variables
        """
        if self.cost < leaf.cost:
            return self

        for var in state.variables:
            self_var_min, self_var_max = state.min_max(var)
            leaf_var_min, leaf_var_max = leaf.state.min_max(var)

            if self_var_min < leaf_var_min:
                new_node = Node(var, leaf_var_min)

                low_state = state.copy()
                low_state.less_than(var, leaf_var_min)
                new_node.low = Leaf(self.cost, action=self.action, state=low_state)

                high_state = state.copy()
                high_state.greater_than(var, leaf_var_min)
                new_node.high = Leaf(self.cost, action=self.action, state=high_state)

                return new_node.put_leaf(leaf, state)

            if self_var_max > leaf_var_max:
                new_node = Node(var, leaf_var_max)

                low_state = state.copy()
                low_state.less_than(var, leaf_var_max)
                new_node.low = Leaf(self.cost, action=self.action, state=low_state)

                high_state = state.copy()
                high_state.greater_than(var, leaf_var_max)
                new_node.high = Leaf(self.cost, action=self.action, state=high_state)

                return new_node.put_leaf(leaf, state)

        # all variables are checked
        return leaf

    def _get_leafs(self, ls):
        ls.append(self)

    def get_leaf(self, state):
        return self

    def set_state(self, state):
        self.state = state

    def merge(self, other):
        """
        not working
        """
        s1, s2 = self.state, other.state
        new_state = State(s1.variables)
        for var in self.state.variables:
            new_state.less_than(var, min(s1.max[var], s2.max[var]))
            new_state.greater_than(var, max(s1.min[var], s2.min[var]))

        return Leaf(
            self.cost + other.cost,
            self.action + other.action,
            state=new_state
        )

    def __str__(self):
        return f'Leaf(action: {self.action}, cost: {self.cost}, {self.state})'

    def __repr__(self):
        return self.__str__()
