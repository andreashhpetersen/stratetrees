import unittest
import argparse

import trees.tests as test_module
from trees.tests.test_max_parts import TestMaxParts3

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--test', '-t', action='store_true',
        help='Run tests'
    )
    parser.add_argument(
        '--draw', '-d', action='store_true',
        help='Draw test cases'
    )
    return parser.parse_args()

def run_tests():
    suite = unittest.TestLoader().loadTestsFromModule(test_module)
    unittest.TextTestRunner(verbosity=2).run(suite)

if __name__ == '__main__':
    args = parse_args()

    if args.draw:
        TestMaxParts3.draw_all()

    if args.test:
        run_tests()
