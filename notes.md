# Notes to remember stuff by

### 2022-07-07 | dtControl

Running `dtControl --input <some_uppaal_strategy>.dump` causes an error. Seems
like the file format doesn't fit what `dtControl` expects. Maybe `dtControl`
thinks that UPPAAL Stratego create look-up tables? It kind of seems that way
from the literature.

_Impurity_ is the degree to which the states agree on their action.


### 2022-04-28

Attempted to make conversion stuff on bouncing ball example from Peters folder.
The strategy in `animations/ball_strategy` has 67046 lines. Importing it with
`roots, variables, actions = load_tree("/path/to/ball_strategy", loc="(1)")`
gave a total of 13.405 leaves. Converting it to decision tree gave a root with
between 80.000 and 100.000 leaves.

I then created a sampling log with the command

```sh
~/uppaal-4.1.20-stratego-9-linux64/bin/verifyta \
    /path/to/BouncingBallStratego01.xml \
    /path/to/query_file.q \
    --sampling-time 0.01
```

and then loaded the output from `./sampling1.log`. The `query_file.q1` just loads
the strategy and makes a simulation for 300 time steps, so the log contains 3000
entries.

When calling `count_visits(root, data, variables)`, I see that the decision
tree chooses to hit the ball many more times, than what the data suggests.
However, it acts equivalently to the Q-trees (as they were loaded into
`roots`), which I checked with the new `trees.utils.test_equivalence` function.
Also, inspecting a case where a "wrong" action (ie.\ one not reflected by the
sample run) was chosen showed that the strategy in `ball_strategy` would also
suggest this action.

For state `Ball[0].p: 4.01469, Ball[0].v: 8.47169` the tree chose action `"1"`
(which is to hit). This is consistent with the values in `ball_strategy`: my
manual evaluation found a Q value of 9.07... for action `"1"` (line 5151) and
a Q value of 9.45... for action `"0"` (line 40593). However, in the
`sampling1.log` line 172, no new action is seen (ie.\ action `"0"` is taken).

Now, when I try to train a new strategy using the queries given in
`BouncingBallStratego01.xml`, I get a strategy `bouncingball_HitWell.json` that
only takes up 9026 lines. The number of leaves in the forest is 1801, and when
turning it into a tree I get 1940. However, the same problem arises when
testing against a sample output, `sampling.log`. This time, the issue can be
seen in state `Ball[0].p: 5.24345, Ball[0].v: 7.52287`, where action `"1"` has
cost 7.77 (line 1516) and action `"0"` has cost 7.88 (line 8466).
