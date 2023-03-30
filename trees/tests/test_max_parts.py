import unittest
import numpy as np

from glob import glob

from trees.models import DecisionTree
from trees.nodes import Leaf, State
from trees.advanced import max_parts, max_parts3
from trees.utils import draw_partitioning, calc_volume, set_edges, get_edge_vals


class TestMaxParts3(unittest.TestCase):
    DATA_PATH = 'trees/tests/data'
    RENDER_PATH = 'trees/tests/data/renders'

    @classmethod
    def read_boxes(cls, fp):
        if fp.endswith('.json'):
            return DecisionTree.load_from_file(fp).leaves()
        elif fp.endswith('.boxes'):
            with open(fp, 'r') as f:
                actions = f.readline().strip().split(' ')
                n_actions = len(actions)

                variables = f.readline().strip().split(' ')
                K = len(variables)

                lines = [l.strip().split(' ') for l in f.readlines()]
                data = [(int(l[0]), list(map(float, l[1:]))) for l in lines]

                leaves = [
                    Leaf(0, action=a, state=State(
                        variables,
                        constraints=np.array(cs).reshape(-1,K).T
                    ))
                    for a, cs in data
                ]
                return leaves

    def get_sorted_matrix(self, ls):
        """
        return a sorted matrix representation of a list of leaves
        so they can easily be compared to another list
        """

        return np.array(sorted(tuple([
            np.hstack((l.action, l.state.constraints.T.flatten())).tolist()
            for l in ls
        ])))

    @classmethod
    def draw_all(cls):
        files = sorted(glob(f'{cls.DATA_PATH}/dt_*'))
        pairs = [(files[i], files[i+1]) for i in range(0, len(files), 2)]

        for inp_f, exp_f in pairs:
            inp_tree = DecisionTree.load_from_file(inp_f)
            exp_leaves = cls.read_boxes(exp_f)

            if len(inp_tree.variables) != 2:
                continue

            name = inp_f.split('/')[-1].replace('.json','').replace('.boxes','')
            for (typ, bs) in [('org', inp_tree.leaves()), ('exp', exp_leaves)]:
                cls.draw_boxes(inp_tree, f'{name}_{typ}.jpeg', boxes=bs)

    @classmethod
    def draw_boxes(cls, tree, fname, boxes=None):
        bs = tree.leaves() if boxes is None else boxes
        bounds = tree.get_bounds()
        draw_partitioning(
            bs, tree.variables[0], tree.variables[1],
            [min(0, bounds[0][0] - 1), bounds[0][-1] + 1],
            [min(0, bounds[1][0] - 1), bounds[1][-1] + 1],
            { 0: 'g', 1: 'r', 2: 'b', 3: 'y' },
            lw=0.1,
            dpi=500,
            out_fp=f'{cls.RENDER_PATH}/{fname}'
        )

    def test_examples(self):
        files = sorted(glob(f'{self.DATA_PATH}/dt_*'))
        pairs = [(files[i], files[i+1]) for i in range(0, len(files), 2)]

        for inp_f, exp_f in pairs:
            inp_tree = DecisionTree.load_from_file(inp_f)
            exp_leaves = self.__class__.read_boxes(exp_f)

            boxes, track = max_parts3(inp_tree, seed=42, return_track_tree=True)

            if len(inp_tree.variables) == 2:
                # draw it
                name = inp_f.split('/')[-1].replace('.json','').replace('.boxes','')
                self.__class__.draw_boxes(inp_tree, f'{name}_output.jpeg', boxes=boxes)

            exp = self.get_sorted_matrix(exp_leaves)
            res = self.get_sorted_matrix(boxes)

            msg = f'{inp_f} did not match its expected partition'
            self.assertEqual(exp.shape, res.shape, msg=msg)
            self.assertTrue((exp == res).all(), msg=msg)

            # compare volumes, but use the same edge values for both
            bs_inp = np.array([b.state.constraints.copy() for b in inp_tree.leaves()])
            bs_res = np.array([b.state.constraints.copy() for b in boxes])

            edges = get_edge_vals(bs_inp, broadcast=False)
            set_edges(bs_inp, edges=edges, inline=True)
            set_edges(bs_res, edges=edges, inline=True)

            msg = f'{inp_f} did not match its expected volume'
            self.assertEqual(calc_volume(bs_inp), calc_volume(bs_res), msg=msg)

            # assert everything is explored
            msg = f'{inp_f} did not explore all parts of the state space'
            unexplored = [l for l in track.leaves() if l.action is None]
            self.assertEqual(len(unexplored), 0, msg=msg)
