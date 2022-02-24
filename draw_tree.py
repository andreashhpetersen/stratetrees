import json, pydot, argparse
from trees.utils import load_tree, draw_graph

parser = argparse.ArgumentParser()
parser.add_argument(
    'file',
    help='file path to the json file containing the tree'
)
parser.add_argument(
    '-o', '--output-file',
    help='path of where to store output .png'
)
args = parser.parse_args()

fp = args.file
with open(fp, 'r') as f:
    data = json.load(f)

# set out_name if not given
if args.output_file is None:
    out_name = fp.replace('_output.json', '_drawing.png')
else:
    out_name = args.output_file

regs = list(data['regressors'].keys())
if len(regs) > 1:
    print('Multiple regressors found in the strategy.')
    print('Choose one (type an integer corresponding the chosen regressor):')
    for i in range(0, len(regs)):
        print(f'{i+1}: {regs[i]}')

    loc = regs[int(input())-1]

roots, variables, actions = load_tree(fp, loc=loc)
draw_graph(roots, labels=actions, out_fp=out_name)
