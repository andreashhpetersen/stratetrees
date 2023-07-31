"""
This script will test a complete pipeline of the algorithm - from minimizing an
input partitioning to rebuilding it as a decision tree.
"""
import numpy as np

from trees.advanced import max_parts3, boxes_to_tree
from trees.models import DecisionTree
from trees.utils import calc_volume

SUCCESSES = 0

strategy_path = './automated/cartpole/generated/constructed_0/trees/dt_original.json'
tree = DecisionTree.load_from_file(strategy_path)

print(f'running max_parts on the original tree of size {tree.size}')
boxes = max_parts3(tree, seed=42)
print(f'found {len(boxes)} regions\n')

print('asserting that none of the found regions violate the singular mapping ' \
      'property')
try:
    for b in boxes:
        state = b.state.constraints
        assert len(tree.get_for_region(state[:,0], state[:,1])) == 1

    SUCCESSES += 1
    print('SUCCESS!')

except AssertionError:
    print('ERROR! One of the new regions maps to more than one action in the '\
          'original tree!')
    print('The leaf as state: ', state)

print()

print('asserting that the volume of the found regions are equal to the volume '\
      'of the leaves in the original tree')
try:
    org_vol = calc_volume(tree.leaves(), leaves=True)
    new_vol = calc_volume(boxes, leaves=True)
    assert org_vol == new_vol

    SUCCESSES += 1
    print('SUCCESS!')

except AssertionError:
    print('ERROR! The volume of the new set of regions did not match the ' \
          'volume of the original tree!')
    print(f'Original volume: {org_vol}')
    print(f'New volume: {new_vol}')

print()

print('asserting that the tree constructed from the found regions are ' \
      'equivalent to the original tree')
try:
    ntree = boxes_to_tree(boxes, tree.variables, actions=tree.actions)
    for leaf in ntree.leaves():
        state = leaf.state.constraints
        assert len(tree.get_for_region(state[:,0], state[:,1])) == 1

    SUCCESSES += 1
    print('SUCCES!')
except AssertionError:
    print('ERROR! The new tree was not consistent with the original tree.')
    print('The problematic leaf has state: ', state)

print()

if SUCCESSES == 3:
    print('It seems you have succesfully fixed the algorithm! But do you trust ' \
          'it to work for all examples..? Anyway, have yourself a cookie for now.')
else:
    print('It seems there are still work to do...')
