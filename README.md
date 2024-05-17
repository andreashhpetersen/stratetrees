# stratetrees

Tool for working with UPPAAL Stratego strategies in Python.

## Installation

Clone this repository and install the requirements.


## Usage

Assuming `myStrategy.json` has been generated with UPPAALs `saveStrategy` command, import it using

```python
from stratetrees.models import QTree

qtree = QTree("path/to/myStrategy.json")
```

A `QTree` is a set of decision trees that in combination represents an entire Q-function for a policy in a Reinforcement Learning setting. Each of the trees that makes up the `QTree` represents the Q-function with respect to a single action, ie. a mapping from a state to a Q-value. The `QTree` is an abstraction that allows the mapping from a state to the optimal action by querying each tree and returning the action with the best Q-value.

However, often we would rather want a single decision tree that maps directly from a state to an optimal action. This can be done easily with the function `to_decision_tree()`:

```python

dtree = qtree.to_decision_tree()

```

These trees can, however, become quite large, both as a result of UPPAALs learning algorithm and because of the inefficiency of the merging operation in `QTree`. A lossless minimization algorithm is provided to accommodate for this issue and can achieve quite substantial reductions:

```python
from stratetrees.advanced import max_parts

small_tree = max_parts(dtree)
```

A repeated application of `max_parts` can probably improve the reduction even further, which can be done easily with `minimize_tree`:

```python
from stratetrees.advanced import minimize_tree

min_tree, _ = minimize_tree(dtree, max_iter=10, verbose=True)
```

In all cases, a `DecisionTree` is returned. It exposes several methods and attributes, but most notably `tree.predict(state)` which returns the optimal action (according to the strategy) in `state`. Other attributes are `tree.size`, `tree.n_leaves`, `tree.variables` and `tree.actions`. A tree can be saved as a `json`-file via the call `tree.save_as("my_dt_strategy.json")` or exported back to a format readable by UPPAAL with `tree.export_to_uppaal("my_minimized_uppaal_strategy.json")` (given it was originally created from a UPPAAL file.
