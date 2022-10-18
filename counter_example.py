import numpy as np

from trees.advanced import max_parts, boxes_to_tree
from trees.models import Node, Leaf, State, Tree
from trees.utils import draw_partitioning, draw_graph


leaves = []

actions = [
    ['a', 'b', 'a', 'c'],
    ['b', 'b', 'a', 'c'],
    ['a', 'b', 'c', 'c'],
    ['a', 'b', 'a', 'c']
]

variables = ['x', 'y']
for x in range(4):
    for y in range(4):
        bs = np.array([
            [x * 2, (x + 1) * 2],
            [y * 2, (y + 1) * 2]
        ])
        s = State(variables, bs)
        l = Leaf(0, action=actions[y][x], state=s)
        leaves.append(l)

tree = boxes_to_tree(leaves, variables, ['a', 'b', 'c'])

min_leaves = max_parts(tree, [0,0], [8,8])
min_tree = boxes_to_tree(min_leaves, variables, ['a', 'b', 'c'])

draw_graph([min_tree.root], out_fp='tree_counter_example.png')

draw_partitioning(
    min_leaves,
    'x', 'y',
    [0, 8],
    [0, 8],
    { 'a': 'r', 'b': 'g', 'c': 'b' },
    dpi=500,
    lw=0.3,
    out_fp='./partitioning_counter_example.jpeg'
)
