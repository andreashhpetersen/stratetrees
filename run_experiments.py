from glob import glob
from time import perf_counter

from trees.advanced import max_parts, boxes_to_tree
from trees.models import Tree
from trees.utils import load_trees, parse_from_sampling_log, count_visits


class performance:
    def __enter__(self):
        self.start = perf_counter()
        return self

    def __exit__(self, *args):
        self.stop = perf_counter()
        self.time = self.stop - self.start


def run_single_experiment(model_dir):
    qt_strat_file = f'{model_dir}/qt_strategy.json'
    sample_logs = glob(f'{model_dir}/sample_*.log')

    data = {}

    qtrees, variables, actions, meta = load_trees(qt_strat_file, verbosity=1)

    data['qtrees'] = (sum([r.size for r in qtrees]), 0, None)

    with performance() as p:
        tree = Tree.build_from_roots(qtrees, variables, actions)

    data['dt_original'] = (tree.size, p.time, None)

    with performance() as p:
        leaves = max_parts(tree)
        mp_tree = boxes_to_tree(leaves, variables, actions)

    data['dt_max_parts'] = (mp_tree.size, p.time, None)

    for sample_log in sample_logs:
        prune_tree = mp_tree.copy()
        samples = parse_from_sampling_log(sample_log)
        sample_size = len(samples)

        with performance() as p:
            count_visits(prune_tree, samples)
            prune_tree.emp_prune()
            leaves = max_parts(prune_tree)
            prune_tree = boxes_to_tree(leaves, variables, actions)

        data[f'dt_prune_{sample_size}'] = (prune_tree.size, p.time)

    return data


if __name__=='__main__':
    data = run_single_experiment('./automated')
    for name, ms in data.items():
        print(f'- {name} -  \n\tSize: {ms[0]}\n\tTime: {ms[1]:0.2f}s')
