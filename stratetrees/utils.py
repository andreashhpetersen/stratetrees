import csv
import json
import time
import pydot
import operator
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.animation as animation

from copy import copy, deepcopy
from itertools import product
from collections import defaultdict
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection

from stratetrees.nodes import Node, Leaf, State
from stratetrees.smc2py import parse_engine_output


####### Functions to add color gradients #######


def hex_to_RGB(hex):
    """
    #FFFFFF -> [255,255,255]
    """

    # Pass 16 to the integer function for change of base
    return [int(hex[i:i+2], 16) for i in range(1,6,2)]


def RGB_to_hex(RGB):
    """
    [255,255,255] -> #FFFFFF
    """
    # Components need to be integers for hex to make sense
    RGB = [int(x) for x in RGB]
    return "#"+"".join(["0{0:x}".format(v) if v < 16 else
              "{0:x}".format(v) for v in RGB])


def color_gradient(weight, start="#FFFFFF", end="#000000"):
    s = hex_to_RGB(start)
    f = hex_to_RGB(end)
    return RGB_to_hex([
        int(s[j] + (weight * (f[j] - s[j]))) for j in range(3)
    ])


####### Functions to draw a tree and save it to png #######


def draw_leaf(
    graph, leaf, n,
    print_action=False, print_cost=False, print_ratio=False
):
    label = ''
    font_color = 'black'
    color = 'white'

    if print_action:
        if hasattr(leaf, 'verbose_action'):
            label += f'{leaf.verbose_action}\n'
        else:
            label += f'{leaf.action}\n'
    if print_cost:
        label += f'Cost: {round(leaf.cost, 2)}'
    if print_ratio and hasattr(leaf, 'ratio'):
        label += f'\nFreq: {round(leaf.ratio * 100, 2)}'
        color = color_gradient(leaf.ratio, start="#FFFFFF", end="#FF0000")
        if leaf.ratio == 0:
            color = 'black'
    if hasattr(leaf, 'best_ratio'):
        label += f'\nbest: {round(leaf.best_ratio * 100, 2)}'

    node = pydot.Node(
        str(n), label=label,
        shape='circle', fillcolor=color, fontcolor=font_color, style='filled'
    )
    graph.add_node(node)
    return node, n


def draw_node(graph, root, n, print_action=False, print_cost=False):
    if root.is_leaf:
        return draw_leaf(
            graph, root, n, print_action=print_action, print_cost=print_cost
        )

    low_node, low_n = draw_node(
        graph, root.low, n, print_action=print_action, print_cost=print_cost
    )
    high_node, high_n = draw_node(
        graph, root.high, low_n + 1, print_action=print_action,
        print_cost=print_cost
    )

    new_n = high_n + 1
    label = f'{root.variable} <= {root.bound:0.2f}'
    node = graph.add_node(
        pydot.Node(str(new_n), label=label, shape='square')
    )


    graph.add_edge(pydot.Edge(str(new_n), str(low_n), label='true'))
    graph.add_edge(pydot.Edge(str(new_n), str(high_n), label='false'))
    return node, new_n


def draw_graph(
    trees, labels=None, out_fp='graph_drawing.png', print_action=True,
    print_cost=False
):
    graph = pydot.Dot(graph_type='digraph')
    root = pydot.Node('root')
    iterator = zip(labels, trees) if labels is not None else enumerate(trees)

    n = 0
    for label, tree in iterator:
        node, n = draw_node(
            graph, tree, n, print_action=print_action, print_cost=print_cost
        )
        if len(trees) > 1:
            graph.add_edge(pydot.Edge('action', str(n), label=label))
            n += 1

    if out_fp.endswith('.dot'):
        graph.write_dot(out_fp)
    elif out_fp.endswith('.png'):
        graph.write_png(out_fp)
    elif out_fp.endswith('.svg'):
        graph.write_svg(out_fp)
    else:
        suffix = out_fp.split('.')[-1]
        raise ValueError(f'{suffix} format not supported')


def draw_partitioning(
    leaves, x_var, y_var, xlim, ylim, cmap,
    xticks=None, yticks=None, extra_patches=None,
    labels={}, dpi=600, lw=0, bounds=False, show=False,
    grid=False, out_fp='./tmp.svg'
):
    min_x, max_x = xlim
    min_y, max_y = ylim

    matplotlib.rc('xtick', labelsize=14)
    matplotlib.rc('ytick', labelsize=14)
    fig, ax = plt.subplots(figsize=(5,4), dpi=dpi)

    bounds_data = []

    actions = []
    patches = []
    for l in leaves:
        s = l.state
        actions.append(l.action)
        x_start, x_end = s.min_max(x_var, min_limit=min_x, max_limit=max_x)
        y_start, y_end = s.min_max(y_var, min_limit=min_y, max_limit=max_y)

        if bounds:
            bounds_data.append((
                ([x_end, x_end], [min_y, max_y]),
                { 'linestyle': (0,(5,20)), 'c': 'black', 'lw': 0.2 }
            ))
            bounds_data.append((
                ([y_end, y_end], [min_x, max_x]),
                { 'linestyle': (0,(5,20)), 'c': 'black', 'lw': 0.2 }
            ))

        width = x_end - x_start
        height = y_end - y_start
        c = cmap[l.action]
        patches.append(
            Rectangle(
                (x_start, y_start), width, height, color=c, ec='black', lw=lw,
                zorder=0
            )
        )

    # patches.append(Rectangle((0,0), 2, 2, fill=None, hatch='//'))

    # rule 1
    # patches.extend([
    #     Rectangle((0,2), 2, 2, color='white', alpha=.5, lw=0),
    #     Rectangle((0,2), 2, 2, fill=None, edgecolor='black', lw=2, linestyle='dashed'),
    # ])

    # rule 2
    # patches.extend([
    #     Rectangle((0,2), 3, 1, color=cmap[0], lw=0),
    #     Rectangle((0,2), 3, 1, fill=None, hatch='\\\\'),

    #     Rectangle((0,3), 2, 1, color=cmap[1], lw=0),
    #     Rectangle((0,3), 2, 1, fill=None, hatch='//'),

    #     Rectangle((2,0), 1, 3, color='white', alpha=.5, lw=0),
    #     Rectangle((2,0), 1, 3, fill=None, edgecolor='black', lw=2, linestyle='dashed'),
    # ])

    # rule 3
    # patches.extend([
    #     # Rectangle((0,2), 4, 1, color=cmap[0], lw=0),
    #     Rectangle((0,2), 4, 1, color='white', alpha=.5, lw=0),
    #     Rectangle((0,2), 4, 1, fill=None, edgecolor='black', lw=2, linestyle='dashed'),
    # ])

    if extra_patches is not None:
        patches.extend(extra_patches)
        for p in patches:
            cp = copy(p)
            ax.add_patch(cp)

    else:
        ax.add_collection(PatchCollection(patches, match_original=True))
        for args, kwargs in bounds_data:
            ax.plot(*args, **kwargs)

    proxies = []
    if isinstance(labels, dict) and len(labels) > 0:
        for k, v in labels.items():
            proxies.append(mpatches.Patch(label=v, color=cmap[k]))
        plt.legend(handles=proxies, fontsize=12)

    plt.xticks(xticks)
    plt.yticks(yticks)

    plt.xlim(xlim)
    plt.ylim(ylim)

    plt.xlabel(x_var, fontsize=16)
    plt.ylabel(y_var, fontsize=16)

    if grid:
        plt.grid(None)

    plt.tight_layout()

    if show:
        plt.show()

    if out_fp is not None:
        plt.savefig(out_fp, dpi=600)

    plt.close('all')


def plot_voxels(bs, actions, points=[], max_v=10, animate=False):
    plt.style.use('_mpl-gallery')

    colormap = ['#B8000050',  '#26DC2650', '#4A90E260']

    # Prepare the coordinates
    shape = (max_v, max_v, max_v)
    x, y, z = np.indices(shape)

    cubes = []
    colors = np.empty(shape, dtype=object)
    for i in range(bs.shape[0]):
        cube = (bs[i,0,0] <= x) & (x < bs[i,0,1]) & \
            (bs[i,1,0] <= y) & (y < bs[i,1,1]) & \
            (bs[i,2,0] <= z) & (z < bs[i,2,1])
        cubes.append(cube)
        colors[cube] = colormap[actions[i]]
        # colors[cube] = '#B8000050' if actions[i] == 0 else '#26DC2650'

    # Plot
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(projection='3d')
    ax.set(xlabel='cart_pos', ylabel='pole_ang', zlabel='pole_vel')

    def update_vox(num):
        voxels = np.full(shape, False, dtype=bool)
        for cube in cubes[:num+1]:
            voxels = voxels | cube
        ax.voxels(voxels, facecolors=colors)

    if len(points) > 0:
        for (point, c) in points:
            x,y,z = point
            ax.scatter(x,y,z, marker='o', s=120, c=c)

    if animate:
        ani = animation.FuncAnimation(fig, update_vox, len(cubes), interval=1000)
        ani.save('animation.gif', dpi=500, writer='pillow')
    else:
        update_vox(len(cubes) + 1)
        plt.show()


def visualize_strategy(tree, *args, **kwargs):
    K = len(tree.variables)
    colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k', 'w']
    cmap = { a: c for a, c in zip(tree.actions, colors) }
    leaves = tree.leaves()

    if K == 2:
        bounds = tree.get_bounds()
        x, y = tree.variables

        xlim = kwargs.pop('xlim', (bounds[0][0] - 1, bounds[0][-1] + 1))
        ylim = kwargs.pop('ylim', (bounds[1][0] - 1, bounds[1][-1] + 1))

        draw_partitioning(leaves, x, y, xlim, ylim, cmap, *args, **kwargs)
    elif K == 3:
        raise NotImplementedError(f'visualization not implemented for 3 dims')
    else:
        raise ValueError(f'can only visualize 2 or 3 dimensions (tree has {K})')


####### Functions to load and add statistics to a tree #######


def convert_uppaal_samples(infile):
    """
    Convert the UPPAAL Stratego sample output from `infile` to a file where each
    line contains a timestamp and the value of each state variable at that time.

    Parameters
    ----------
    infile : str
        The filepath to the sample log

    Returns
    -------
    outfile : str
        The path to the file with the converted samples. If `infile` is
        'samples.log' then `outfile` will be 'samples_converted.log'

    Author: Peter Gjøl Jensen
    """
    filepath = infile.split('.')
    if len(filepath) > 2:
        filepath = ['.'.join(filepath[:-1])] + [filepath[-1]]

    outfile = '.'.join([filepath[0] + '_converted', filepath[-1]])

    from_engine = parse_engine_output(infile)
    trajectories = from_engine[-1]
    with open(outfile, "w") as out:
        for i in range(len(trajectories)):
            hints = [0 for _ in trajectories.fields]
            for (t, v) in trajectories.data[0][i].points:
                if v > 0:
                    # the controller is in power
                    values = []
                    for d in range(1, len(trajectories.fields)):
                        (value, hint) = trajectories.data[d][i].predict(t,hints[d])
                        if isinstance(value, list):
                            value = value[0]
                        values.append(str(value))
                        hints[d] = hint

                    out.write(",".join(values))
                    out.write("\n")

    return outfile


def parse_from_sampling_log(filepath, from_uppaal=True):
    """
    Parse the sample data in `filepath` and return a numpy array. If the data
    comes directly from the UPPAAL Stratego sampling call, set `from_uppaal` to
    `True`
    """
    if from_uppaal:
        filepath = convert_uppaal_samples(filepath)

    with open(filepath, 'r') as f:
        data = f.readlines()

    return np.array([list(map(float, s.strip().split(','))) for s in data])


###### Misc

class performance:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.stop = time.perf_counter()
        self.time = self.stop - self.start


def time_it(alg, *args, **kwargs):
    tic = time.perf_counter()
    out = alg(*args, *kwargs)
    toc = time.perf_counter()
    # print(f'finished in {toc - tic:0.4f} seconds')
    return out, toc - tic


def in_box(b, s, inclusive=True):
    if not isinstance(b, np.ndarray):
        b = np.array(b).T

    if not b.shape[-1] == 2:
        b = b.T

    op = operator.le if inclusive else operator.lt
    for i in range(len(s)):
        if not (b[i][0] < s[i] and op(s[i], b[i][1])):
            return False

    return True


def in_list(ls, ms):
    for b in ls:
        if in_box(b.state.constraints, ms):
            return True
    return False


def make_state(p_state, bounds):
    return bounds[np.arange(len(bounds)), p_state].T


def make_leaf(action, variables, state, cost=0):
    return Leaf(action, cost=cost, state=State(variables, constraints=state))


def has_overlap(ps1, ps2):
    """
    Returns `True` if the K-dimensional cubes given by `ps1` and `ps2` are
    overlapping in any way (has a non-empty intersection).
    """
    pmin1, pmax1 = ps1
    pmin2, pmax2 = ps2
    K = len(pmin1)

    for i in range(K):
        if min(pmax1[i], pmax2[i]) - max(pmin1[i], pmin2[i]) <= 0:
            return False

    return True


def breaks_box(box, breaker):
    p_min, p_max = breaker
    n_pmin, n_pmax = box

    remainder = 0
    for i in range(len(p_max)):
        if n_pmin[i] < p_max[i] and p_max[i] < n_pmax[i]:
            remainder += 1

        if n_pmin[i] < p_min[i] and p_min[i] < n_pmax[i]:
            remainder += 1

        if remainder > 1:
            return True
    return False


def cut_overlaps(ps_main, ps_to_cut, allow_multiple=False):
    """
    If `ps_main` and `ps_to_cut` has any overlap, return a tuple with the min
    and max points of `ps_to_cut` after the overlap has been cut away. Throws
    an `AssertionError` if more than one cut is needed and `allow_multiple` is
    `False`.
    """
    pmin, pmax = ps_main
    n_pmin, n_pmax = ps_to_cut
    K = len(pmin)
    assert K == len(pmax) == len(n_pmin) == len(n_pmax)

    if not has_overlap(ps_main, ps_to_cut):
        return ps_to_cut

    cuts = 0
    out_pmin, out_pmax = [], []
    for i in range(K):
        if pmax[i] > n_pmin[i] >= pmin[i] and n_pmax[i] > pmax[i]:
            cuts += 1
            out_pmin.append(pmax[i])
            out_pmax.append(n_pmax[i])

        elif pmin[i] < n_pmax[i] <= pmax[i] and n_pmin[i] < pmin[i]:
            cuts += 1
            out_pmin.append(n_pmin[i])
            out_pmax.append(pmin[i])

        else:
            out_pmin.append(n_pmin[i])
            out_pmax.append(n_pmax[i])

    if not allow_multiple:
        assert cuts < 2

    return out_pmin, out_pmax


def get_bbox(bs, K):
    """
    Get bounding box of `bs`
    """
    bbox = np.ones((K,2)) * [np.inf, -np.inf]
    for b in bs:
        for i in range(K):
            bbox[i,0] = min(bbox[i,0], b[i,0])
            bbox[i,1] = max(bbox[i,1], b[i,1])

    return bbox


def leaves_to_state_constraints(leaves):
    return np.array([l.state.constraints.copy() for l in leaves])


def get_edge_vals(bs, pad=1, broadcast=True, leaves=False):
    """
    Calculate the edge values of the in each dimension from the boxes in `bs`.
    The edges are the lowest/highest finite value of any box in `bs`, possibly
    padded with `pad` if the actual limit is infinite.

    If `broadcast=True`, the returned np.ndarray will have the same shape as
    `bs`, otherwise it will have the shape of a single box in `bs`.
    """
    if leaves:
        bs = leaves_to_state_constraints(bs)

    K = bs.shape[1]
    edges = np.zeros((K,2))

    for i in range(K):
        finite = bs[:,i,:][np.isfinite(bs[:,i,:])]

        if len(finite) == 0:
            minv = 0.0
            maxv = 1.0
        else:
            minv = finite.min() - pad
            maxv = finite.max() + pad

        edges[i] = [minv, maxv]

    assert (edges[:,0] < edges[:,1]).all()
    if broadcast:
        edges = np.repeat(np.expand_dims(edges, axis=0), bs.shape[0], axis=0)
    return edges


def set_edges(bs, pad=1, edges=None, inline=False, leaves=False):
    """
    Replace infinite limits in `bs` with finite edge values.
    """
    if leaves:
        bs = leaves_to_state_constraints(bs)

    if not inline and not leaves:
        bs = bs.copy()

    if edges is None:
        edges = get_edge_vals(bs, pad=pad)

    if not bs.shape == edges.shape:
        edges = np.repeat(np.expand_dims(edges, axis=0), bs.shape[0], axis=0)

    infs = np.isinf(bs)
    bs[infs] = edges[infs]
    return bs


def calc_volume(bs, leaves=False):
    """
    Calculate the volume of the boxes in `bs`. If `leaves=True`, then `bs` is
    expected to be a list of `trees.nodes.Leaf` objects.
    """
    if leaves:
        bs = leaves_to_state_constraints(bs)
        bs = set_edges(bs)

    if len(bs.shape) == 2:
        bs = np.expand_dims(bs, axis=0)
    return np.product(bs[:,:,1] - bs[:,:,0], axis=1).sum()


def compare_volumes(ls1, ls2):
    edges = get_edge_vals(ls1, leaves=True, broadcast=False)
    bs = set_edges(ls2, edges=edges, leaves=True)

    vol1 = calc_volume(ls1, leaves=True)
    vol2 = calc_volume(bs)

    return np.isclose(vol1, vol2), (vol1, vol2)


def is_consistent(tree, ntree, verbose=False):
    bad_leaves = []
    for leaf in ntree.leaves():
        action = leaf.action
        state = leaf.state.constraints
        if not tree.get_for_region(state[:,0], state[:,1]) == set([action]):
            if verbose:
                bad_leaves.append(leaf)
            else:
                return False
    return len(bad_leaves) == 0, bad_leaves if verbose else True


def transform_bounds(tree, pad=1, inline=False):
    """
    Transform the bounds of `tree` to simple integer values in the range
    [0, N*`pad`], where N is the number of bounds in the dimension that has the
    most bounds. If `inline=True`, the transformation is performed on the
    provided tree, else a new tree is returned.
    """
    bounds = tree.get_bounds()
    bmap = [{} for _ in range(len(tree.variables))]
    upper = (max([len(bs) for bs in bounds])) * pad

    for v in range(len(bounds)):
        for i, b in enumerate(bounds[v]):
            bmap[v][b] = upper if i == len(bounds[v]) - 1 else i * pad

    def rec(n, bmap):
        n.bound = bmap[n.var_id][n.bound]

    if not inline:
        tree_copy = tree.copy()
        tree_copy.root.visitor(rec, bmap)
        tree.set_state()
        return tree_copy
    else:
        tree.root.visitor(rec, bmap)
        tree.set_state()
