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


class Leaf:
    def __init__(self, cost, action=None, state=None):
        self.cost = cost
        self.action = action
        self.state = state
        self.visits = 0

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
