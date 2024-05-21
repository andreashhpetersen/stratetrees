#!/usr/bin/env python3

import argparse
from glob import glob

from trees.commands import minimize, run_tests, convert, draw
from experiments.commands import run_experiments, make_samples, controller_experiment


def get_parser():
    parser = argparse.ArgumentParser(prog='stratetrees')

    subparsers = parser.add_subparsers(title='actions', dest='command')

    parser_new = subparsers.add_parser(
        'experiment',
        help='New experiment parser'
    )
    parser_new.add_argument(
        'MODEL_DIR',
        help='The directory containing the model to run experiments on.'
    )

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

    parser_convert = subparsers.add_parser(
        'convert',
        help='Convert a QTree-strategy (from UPPAAL Stratego) to a Decision Tree'
    )
    parser_convert.add_argument(
        'STRATEGY_FILE',
        help='Path to the strategy file'
    )
    parser_convert.add_argument(
        '--output-name', '-o',
        nargs='?', type=str,
        help='Name for output file'
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

    parser_draw = subparsers.add_parser(
        'draw', help='Draw a tree as a graph'
    )
    parser_draw.add_argument(
        'STRATEGY_FILE',
        help='Path to the strategy file'
    )
    parser_draw.add_argument(
        '--outfile', '-o', nargs='?',
        help='Optional file to store the drawing in'
    )
    parser_draw.add_argument(
        '--qtree', action='store_true',
        help='Set this flag if the strategy is a QTree'
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

    elif args.command == 'convert':
        if args.output_name:
            convert(args.STRATEGY_FILE, output_fp=args.output_name)
        else:
            convert(args.STRATEGY_FILE)

    elif args.command == 'draw':
        draw(args.STRATEGY_FILE, args.outfile, qtree=args.qtree)

    elif args.command == 'experiment':
        meta, results = controller_experiment(args.MODEL_DIR)
        import ipdb; ipdb.set_trace()

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
