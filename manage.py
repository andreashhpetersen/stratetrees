#!/usr/bin/env python3

import argparse
from glob import glob

from trees.commands import minimize, run_tests
from experiments.commands import run_experiments, make_samples


def get_parser():
    parser = argparse.ArgumentParser(prog='stratetrees')

    subparsers = parser.add_subparsers(title='actions', dest='command')

    parser_exp = subparsers.add_parser(
        'run_experiments',
        help='Run the experiments suite'
    )

    parser_exp.add_argument(
        'MODEL_DIR',
        help='The directory containing the model to run experiments on.'
    )
    parser_exp.add_argument(
        '--all', '-a', action='store_true',
        help='If this flag is passed, MODEL_DIR is considered a parent ' \
        'directory containing the actual model directories to run' \
        'experiments on'
    )
    parser_exp.add_argument(
        '-k', '--repeats',
        nargs='?', type=int, const=1, default=1,
        help='Specify how many times the experiment should run (default 1)'
    )
    parser_exp.add_argument(
        '--early_stopping', '-e', action='store_true',
        help='Use early stopping for repeated maxparts application'
    )
    parser_exp.add_argument(
        '--skip-sampling', '-s', action='store_true',
        help='Set if you do not want to make samples before running experiments'
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
        nargs='?', type=str,
        help='Samples file. If this is provided, empirical pruning is performed'
    )
    parser_min.add_argument(
        '-v', '--visualize',
        action='store_true',
        help='Make visualization of the minimized state space (2D only)'
    )
    parser_min.add_argument(
        '-o', '--output-dir',
        type=str, nargs='?', default='',
        help='(optional) The directory to store the output in (the path will ' \
             'be created if it does not already exist)'
    )

    parser_make_samples = subparsers.add_parser(
        'make_samples', help='Generate samples by running UPPAAL Stratego'
    )
    parser_make_samples.add_argument(
        'MODEL_DIR',
        help='The directory containing the model to run experiments on.'
    )


    parser_test = subparsers.add_parser(
        'test', help='Run the test suite'
    )
    parser_test.add_argument(
        '--draw', '-d', action='store_true',
        help='Draw test cases'
    )

    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    if args.command == 'minimize':

        OUT_DIR = args.output_dir
        if len(OUT_DIR) > 0 and not OUT_DIR.endswith('/'):
            OUT_DIR += '/'

        OUT_PATH = OUT_DIR + 'minimized_output'
        minimize(
            args.STRATEGY_FILE,
            OUT_PATH,
            samples=args.samples,
            visualize=args.visualize
        )


    elif args.command == 'run_experiments':

        early_stopping = args.early_stopping
        model_dir, k = args.MODEL_DIR, args.repeats
        if args.all:
            model_dirs = glob(f'{model_dir}/*/')
        else:
            model_dirs = [model_dir]

        for model_dir in model_dirs:
            msg = f"RUNNING EXPERIMENTS FOR '{model_dir}'"
            print('\n{}\n{}\n{}\n'.format('#' * len(msg), msg, '#' * len(msg)))

            if not args.skip_sampling:
                make_samples(model_dir)
            run_experiments(model_dir, k=k, early_stopping=early_stopping)

    elif args.command == 'make_samples':

        model_dir = args.MODEL_DIR
        make_samples(model_dir)

    elif args.command == 'test':

        run_tests(draw=args.draw)
