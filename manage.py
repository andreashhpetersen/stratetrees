import os
import re
import csv
import sys
import json
import argparse
import numpy as np

from tqdm import tqdm
from glob import glob
from copy import deepcopy
from time import perf_counter

from trees.advanced import minimize_tree
from trees.models import QTree, DecisionTree
from trees.utils import parse_from_sampling_log, performance, visualize_strategy

from experiments.experiments import dump_json, write_results, \
    write_trans_results, run_experiment, run_single_experiment


def parse_args():
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(dest='command')

    parser_exp = subparsers.add_parser(
        'run_experiments',
        help='Run the experiments suite'
    )

    parser_exp.add_argument(
        'MODEL_DIR',
        help='The directory containing the model to run experiments on.'
    )
    parser_exp.add_argument(
        '-k', '--repeats',
        nargs='?', type=int, const=1, default=1,
        help='Specify how many times the experiment should run (default 1)'
    )
    parser_exp.add_argument(
        '-u', '--store-uppaal',
        action='store_true',
        help='Export the constructed strategies to a UPPAAL Stratego format'
    )
    parser_exp.add_argument(
        '-p', '--print-results',
        action='store_true',
        help='Print results to screen'
    )

    parser_min = subparsers.add_parser(
        'minimize',
        help='Minimize a strategy'
    )

    parser_min.add_argument(
        'STRATEGY_FILE',
        help='Path to the strategy file'
    )
    parser_min.add_argument(
        '-s', '--samples',
        nargs='?', type=argparse.FileType('r'), default=sys.stdin,
        help='Samples file. If this is provided, empirical pruning is performed'
    )
    parser_min.add_argument(
        '-v', '--visualize',
        action='store_true',
        help='Make visualization of the minimized state space (2D only)'
    )


    return parser.parse_args()

# EXPORT_UPPAAL = args.store_uppaal
S_ID, T_ID = 0, 1   # size and time



if __name__ == '__main__':
    args = parse_args()

    if args.command == 'minimize':
        qtree = QTree(args.STRATEGY_FILE)

        print(f'Imported QTree-strategy of size {qtree.size} leaves\n')

        print('Converting to decision tree...')

        tree = qtree.to_decision_tree()
        print(f'Constructed decision tree with {tree.n_leaves} leaves\n')

        print('Minimizing tree with repeated application of maxparts...')
        ntree, (data, best_i) = minimize_tree(tree)

        print(f'Constructed minimized tree with {ntree.n_leaves} leaves\n')

        ctree = None
        if args.samples:
            try:
                samples = parse_from_sampling_log(args.samples)
                print('performing empirical pruning...')
                ctree = ntree.copy()
                ctree.count_visits(samples)
                ctree.emp_prune()
                ctree, _ = minimize_tree(ctree)
                print(f'Constructed new tree with {ctree.n_leaves} leaves\n')

            except:
                print('processing of samples file failed (did you remember ' \
                      'to run it through log2ctrl.py?)')

        try:
            os.mkdir('./minimized_output')
        except FileExistsError:
            pass

        ntree.export_to_uppaal('./minimized_output/maxparts_uppaal.json')
        ntree.save_as('./minimized_output/maxparts_dt.json')

        if len(ntree.variables) == 2:
            visualize_strategy(
                tree,
                labels={ a: a for a in tree.actions },
                lw=0.2,
                out_fp='./minimized_output/original_visual.png'
            )

            visualize_strategy(
                ntree,
                labels={ a: a for a in tree.actions },
                lw=0.2,
                out_fp='./minimized_output/maxparts_visual.png'
            )

            if ctree is not None:
                visualize_strategy(
                    ctree,
                    labels={ a: a for a in tree.actions },
                    lw=0.2,
                    out_fp='./minimized_output/empprune_visual.png'
                )

        if ctree is not None:
            ctree.export_to_uppaal('./minimized_output/empprune_uppaal.json')
            ctree.save_as('./minimized_output/empprune_dt.json')

    else:

        model_dir, k = args.MODEL_DIR, args.repeats
        model_names, data = run_experiment(model_dir, k=k)
        if args.print_results:
            for i in range(len(model_names)):
                model_d = data[:,i].T
                print(model_names[i])
                print('\tTime:\t\t(avg)\t\t(std)\t\t(best)')
                print('\t     \t\t{:0.2f}\t\t{:0.2f}\t\t{:0.2f}'.format(
                    model_d[T_ID].mean(),
                    model_d[T_ID].std(),
                    model_d[T_ID].min()
                ))
                print('\tSize:\t\t(avg)\t\t(std)\t\t(best)')
                print('\t     \t\t{:0.2f}\t\t{:0.2f}\t\t{}'.format(
                    model_d[S_ID].mean(),
                    model_d[S_ID].std(),
                    int(model_d[S_ID].min())
                ))
