# Working with RL strategies as decision trees (or Q-trees)

Library to work with Reinforcement Learning strategies represented as Q-value
trees. That is, the Q-values of each state/action pair is stored in the leaf
nodes of a decision tree representing one possible action. A complete strategy
is thus given by having one tree for each possible action.

The representation is based on the json output of strategies trained with
 [UPPAAL Stratego](https://people.cs.aau.dk/~marius/stratego/).

---

Currently messy. But you can 

- draw trees
- convert Q-trees to decision trees (with actions in the leafs)

## Draw trees

Make a virtual environment, activate it and run `pip install -r
requirements.txt` (which currently just installs `pydot` that is necessary for
actually drawing the graph).

Then, you can run the following command

```
python draw_tree.py <path/to/strategy.json> -o output.png
```

which will load the strategy in `strategy.json` and draw it as a graph which is
written to the file `output.png`. If `strategy.json` has more than one
`regressor`, then the program will ask you to choose just one of them to plot.

Here is an example of a tree drawn from `exampleStrategy.json` where regressor
`(1)` has been chosen:

![Example tree]( ./examples/exampleQTree.png )

## Convert Q-trees to decision trees

Currently, this can't be done from the command line. But to - in a completely
naive way - take a set of trees representing a complete strategy and turn it
into a single tree with the best actions in the leaf nodes, you can do the
following:

```python
from trees.models import State
from trees.utils import load_tree

# load tree and extract leafs
roots, variables, actions = load_tree("path/to/strategy.json")
leafs = [l for ls in [root.get_leafs() for root in roots] for l in ls]

# make a root node and build from that
root = Node.make_root_from_leaf(leafs[0])
for i in range(1, len(leafs)):
    root = root.put_leaf(leafs[i], State(variables))
```

The result of this conversion based on the example above (the test strategy in
`examples/exampleStrategy.json`), yields the following decision tree:

![Example decision tree](./examples/exampleDecisionTree.png)

However, we can get a better (smaller) representation of this tree by sorting
the leafs according to cost value first. This is done automatically if we call
the handy class method `make_decision_tree_from_leafs(leafs)` on `Node`. The
resulting tree is shown below.

![Example decision tree made from a list of sorted leafs](./examples/exampleTreeSorted.png)

Examples involving a larger strategy can be found in `examples/`.
