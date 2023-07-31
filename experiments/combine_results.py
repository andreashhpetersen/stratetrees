import re
import csv
import argparse
import numpy as np
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument(
    'MODEL_DIR',
    help='The directory containing the model to run experiments on.'
)
args = parser.parse_args()
S_ID, T_ID = 0, 1   # size and time


def parse_uppaal_results(model_dir):
    with open(f'{model_dir}/generated/eval_results.txt', 'r') as f:
        plain_data = f.readlines()

    results = {}
    model_ptr = r'([\w]+).json'
    eval_ptr = r'^\([\d]+ runs\) E\([\w]+\) = ([\d]+.[\d]+) ± ([\d]+.[\d]+)'

    check_next = False
    for line in plain_data:
        line = line.strip()
        if line.startswith('EVALUATE'):
            model_name = re.findall(model_ptr, line)[0]

        if check_next:
            assert line == '-- Formula is satisfied.'
            check_next = False

        if line.startswith('Verifying'):
            check_next = True

        match = re.match(eval_ptr, line)
        if match:
            exp = float(match.groups()[0])
            std = float(match.groups()[1])
            results[model_name] = [exp, std]

    return results


if __name__=='__main__':
    model_dir = args.MODEL_DIR
    eval_data = parse_uppaal_results(model_dir)
    dt_data = pd.read_csv(f'{model_dir}/generated/dt_results.csv', index_col=0)

    dt_data['perf (exp)'] = 0
    dt_data['perf (std)'] = 0

    for model_name, (exp, std) in eval_data.items():
        dt_data.at[model_name, 'perf (exp)'] = exp
        dt_data.at[model_name, 'perf (std)'] = std

    dt_data.to_csv(f'{model_dir}/generated/combined_results.csv')
