import os
import re
import csv
import json
import argparse
import numpy as np

from tqdm import tqdm
from glob import glob
from copy import deepcopy
from time import perf_counter

from trees.advanced import max_parts, max_parts3, boxes_to_tree
from trees.models import QTree, DecisionTree
from trees.utils import parse_from_sampling_log, performance

from experiments.run_experiments import dump_json, write_results, \
    write_trans_results, transitive_max_parts, run_experiment, \
    run_single_experiment

parser = argparse.ArgumentParser()
parser.add_argument(
    'MODEL_DIR',
    help='The directory containing the model to run experiments on.'
)
parser.add_argument(
    '-k', '--repeats',
    nargs='?', type=int, const=1, default=1,
    help='Specify how many times the experiment should run (default 1)'
)
parser.add_argument(
    '-u', '--store-uppaal',
    action='store_true',
    help='Export the constructed strategies to a UPPAAL Stratego format'
)
parser.add_argument(
    '-p', '--print-results',
    action='store_true',
    help='Print results to screen'
)
args = parser.parse_args()

EXPORT_UPPAAL = args.store_uppaal
S_ID, T_ID = 0, 1   # size and time



if __name__ == '__main__':
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
