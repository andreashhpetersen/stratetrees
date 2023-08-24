import pathlib
import unittest

import trees.tests as test_module
from trees.advanced import minimize_tree
from trees.models import QTree
from trees.tests.test_max_parts import TestMaxParts
from trees.utils import parse_from_sampling_log, visualize_strategy


def run_tests(draw=False):
    if draw:
        TestMaxParts.draw_all()

    suite = unittest.TestLoader().loadTestsFromModule(test_module)
    unittest.TextTestRunner(verbosity=2).run(suite)


def minimize(strategy_fp, output_dir, samples=None, visualize=False):
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    qtree = QTree(strategy_fp)

    print(f'Imported QTree-strategy of size {qtree.size} leaves\n')

    print('Converting to decision tree...')

    tree = qtree.to_decision_tree()
    print(f'Constructed decision tree with {tree.n_leaves} leaves\n')

    print('Minimizing tree with repeated application of maxparts...')
    ntree, (data, best_i) = minimize_tree(tree)
    ntree.meta = tree.meta

    print(f'Constructed minimized tree with {ntree.n_leaves} leaves\n')

    ctree = None
    if samples:
        samples = parse_from_sampling_log(args.samples)
        print('performing empirical pruning...')
        ctree = ntree.copy()
        ctree.count_visits(samples)
        ctree.emp_prune()
        ctree, _ = minimize_tree(ctree)
        print(f'Constructed new tree with {ctree.n_leaves} leaves\n')

    ntree.export_to_uppaal(f'{output_dir}/maxparts_uppaal.json')
    ntree.save_as(f'{output_dir}/maxparts_dt.json')

    if len(ntree.variables) == 2:
        visualize_strategy(
            tree,
            labels={ a: a for a in tree.actions },
            lw=0.2,
            out_fp=f'{output_dir}/original_visual.png'
        )

        visualize_strategy(
            ntree,
            labels={ a: a for a in tree.actions },
            lw=0.2,
            out_fp=f'{output_dir}/maxparts_visual.png'
        )

        if ctree is not None:
            visualize_strategy(
                ctree,
                labels={ a: a for a in tree.actions },
                lw=0.2,
                out_fp=f'{output_dir}/empprune_visual.png'
            )

    if ctree is not None:
        ctree.export_to_uppaal(f'{output_dir}/empprune_uppaal.json')
        ctree.save_as(f'{output_dir}/empprune_dt.json')

    print(f'Output stored in {output_dir}/\n')
