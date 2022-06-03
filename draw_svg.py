import json
import math
import drawSvg as draw
from trees.models import Node, Leaf, State

in_file = './ball_strategy_zeroPruned.json'
out_file = './testOutputPruned.svg'
# in_file = './complexExample.json'
# out_file = './complexExample.svg'
# in_file = './bigBoxTreeFromShuffledBoxes.json'
# out_file = './bigBoxTreeFromShuffledBoxes.svg'
root, variables, actions = Node.load_from_file(in_file)
leafs = root.get_leafs()

varX, varY = 'Ball[0].v', 'Ball[0].p'
# varX, varY = 'x', 'y'
print(f'x: {varX}\ny: {varY}')

boxes_pairs = []
min_x, min_y, max_x, max_y = math.inf, math.inf, -math.inf, -math.inf
for leaf in leafs:
    s = leaf.state
    x1, x2 = s.min[varX], s.max[varX]
    y1, y2 = s.min[varY], s.max[varY]
    boxes_pairs.append(((x1,y1), (x2,y2)))

    min_x = x1 if x1 > -math.inf and x1 < min_x else min_x
    min_y = y1 if y1 > -math.inf and y1 < min_y else min_y
    max_x = x2 if x2 < math.inf and x2 > max_x else max_x
    max_y = y2 if y2 < math.inf and y2 > max_y else max_y


min_x, min_y = min(0, math.floor(min_x)), min(0, math.floor(min_y))
max_x, max_y = math.ceil(max_x) + 2, math.ceil(max_y)

# max_y = 5
boxes = []
for i, ((x1, y1), (x2, y2)) in enumerate(boxes_pairs):
    x1 = min_x if x1 == -math.inf else x1
    y1 = min_y if y1 == -math.inf else y1

    x2 = max_x if x2 == math.inf else x2
    y2 = max_y if y2 == math.inf else y2

    width = x2 - x1
    height = y2 - y1
    boxes.append((x1, y1, width, height, leafs[i].action))


can_width = 2400
can_height = 1900

print(min_x, min_y)
print(max_x, max_y)

state_width = max_x - min_x
state_height = max_y - min_y

conv_x = lambda x: (x / state_width) * can_width
conv_y = lambda y: (y / state_height) * can_height

ox = conv_x(state_width / 2)
d = draw.Drawing(can_width, can_height, origin=(-ox, 0), displayInline=False)

a_colors = { a: c for a,c in zip(actions, ['red', 'green', 'blue']) }
for x, y, w, h, action in boxes:
    cx, cw = map(conv_x, (x, w))
    cy, ch = map(conv_y, (y, h))

    fill = 'green' if action == '1' else 'white'
    fill = a_colors[action]
    r = draw.Rectangle(
        cx, cy, cw, ch, fill=fill, stroke='black', stroke_width=0.0
    )
    d.append(r)


# draw the grid

def draw_grid(d, xk, yk, min_x, min_y, max_x, max_y, lines=True):
    """
    d: a Drawing object
    """

    x_ticks = [
        x for x in range(min_x + (xk - min_x % xk), max_x - (xk - max_x % xk), xk)
    ]
    for x in x_ticks:
        l = draw.Line(
            conv_x(x), conv_y(min_y), conv_x(x), conv_y(max_y) if lines else conv_y(0.1),
            stroke='black', stroke_width=2
        )
        d.append(l)

    y_ticks = [
        y for y in range(min_y + (yk - min_y % yk), max_y - (yk - max_y % yk), yk)
    ]
    for y in y_ticks:
        l = draw.Line(
            conv_x(min_x), conv_y(y), conv_x(max_x) if lines else conv_x(min_x + 0.1), conv_y(y),
            stroke='black', stroke_width=2
        )
        d.append(l)

    xax = draw.Line(
        conv_x(min_x), conv_y(min_y), conv_x(max_x), conv_y(min_y),
        stroke='black', stroke_width=2
    )
    yax = draw.Line(
        conv_x(0), conv_y(min_y), conv_x(0), conv_y(max_y),
        stroke='black', stroke_width=2
    )
    d.append(xax)
    d.append(yax)

draw_grid(d, 1, 1, min_x, min_y, max_x, max_y, lines=False)
d.saveSvg(out_file)
