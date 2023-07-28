import json
import inspect
import numpy as np

from trees.models import DecisionTree
from trees.utils import performance, is_consistent
from trees.advanced import SearchHeuristics, max_parts3


if __name__ == '__main__':
    N = 5
    res = {}
    tree = DecisionTree.load_from_file(
        './automated/cartpole/generated/constructed_0/trees/dt_original.json'
    )

    heuristics = inspect.getmembers(
        SearchHeuristics,
        predicate=inspect.ismethod
    )
    for name, func in heuristics:
        print(f"running with '{name}' as heuristic:")
        times = []
        track_sizes = []
        n_regions = []
        for i in range(N):
            with performance() as perf:
                regions, info = max_parts3(
                    tree,
                    return_info=True,
                    heuristic_func=func
                )

            ts = info['track_sizes']
            times.append(perf.time)
            track_sizes.append(ts)
            n_regions.append(len(regions))

            print(f'run {i+1} finished in {perf.time:0.2f} seconds')
            print(f'found {len(regions)} regions')
            print(f'avg track size: {ts.mean():0.2f} (+/- {ts.std():0.2f})')
            print()

        times = np.array(times)
        sizes = np.concatenate(track_sizes)

        # avg, std, max, min
        res[name] = {
            'time': (times.mean(), times.std(), int(times.max()), int(times.min())),
            'track': (sizes.mean(), sizes.std(), int(sizes.max()), int(sizes.min())),
            'regions': (np.mean(n_regions)),
        }

        print(f'MEAN TIME: {times.mean():0.2f}')
        print(f'MEAN REGIONS: {np.mean(n_regions)}')
        print(f'MEAN TRACK: {sizes.mean():0.2f} (+/- {sizes.std():0.2f})')
        print(f'MAX TRACK: {sizes.max()}')
        print()

    with open('heuristic_search.json', 'w') as f:
        json.dump(res, f)
