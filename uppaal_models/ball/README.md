# Bouncing Ball examples

Model and strategy from [Peter Gjøl Jensen, et. al.](https://zenodo.org/record/3268381)


## Testing effect of empirically pruning

- Load strategy from `./strategies/large_strategy.json`
- Q-trees have a total of 13,405 leaves
- Original decision tree strategy has 81,324 leaves
    - Exported as `./strategies/large_converted_noPrune.json` (but too large
        for GitHub)
- Samples gathered from `./sampling.log` generated using `./BouncingBall01Stratego.xml`
    and `simulate 1 [<=300] {Ball(0).p, Ball(0).v, LearnerPlayer.fired} under s`
    with `--sampling-time 0.01`
- Pruning all zero visits gives a new strategy with just 3,223 leaves
    - Exported as `./strategies/large_converted_zeroPruned.json`
- Pruning all single visits gives a strategy with 994 leaves
    - Exported as `./strategies/large_converted_onePruned.json`
- Pruning all leaves visited twice brings the number down to 401
    - Exported as `./strategies/large_converted_twoPruned.json`

Initial results from running `E[<=120;1000] (max:LearnerPlayer.fired + (sum (id
: ball_id) (Ball(id).number_deaths) * 1000 ))` under the different strategies
in UPPAAL:

Strategy | Leaves | Expectation | Deviation
--- | --- | --- | --- |
Original | 13,405 | 38,490 | 0,178024
Converted (no prune) | 81,324 | 38,337 | 0,172967
ZeroPruned | 3,223 | 50,804 | 5,16206
OnePruned | 994 | 897,497 | 48,0594
TwoPruned | 401 | 2251,27 | 53,5604

Lets try again...

This time, we make 50 simulations  but we sample with `--sampling-time 0.3`.
Pruning zero, one and two visits and exporting to
`./strategies/large_converted_xxxPruned2.json`. This gives:

Strategy | Leaves | Expectation | Deviation
--- | --- | --- | --- |
Original | 13,405 | 38,490 | 0,178024
Converted (no prune) | 81,324 | 38,337 | 0,172967
ZeroPruned2 | 6,256 | 39,900 | 0,205447
OnePruned2 | 3,387 | 70,383 | 9,66556
TwoPruned2 | 1,962 | 926,173 | 49,134


## Minify strategies

In the following, we use the tools in the `trees.advanced` that let us minify a
strategy by merging adjacent partitions with the same optimal action. This is
done without considering the tree structure and is therefor rather a geometric
minimization. However, the strategy is turned back into a tree after the fact,
so that it can be inspected and exported to UPPAAL.

We start by considering the complete strategy, derived from `./strategies/large_strategy.json`, which for this alteration gave a tree with 86,807 leaves (there is stochasticity in the tree generating algorithm, which results in slightly different sized trees for each run). A visual representation of the strategy can be seen below:

![Full strategy partitioning]( ./svgs/largeStrategyNoPrunePartitioning.svg )

Then we use the function `get_boxes()` as follows:

```python
boxes = get_boxes(
    root,                        # this is our strategy
    ['Ball[0].p', 'Ball[0].v'],  # these are the variables in the strategy
    min_vals=[0, -14],           # these are minimum values of our two variables
    eps=0.000001                 # this is how much we 'move' into a new box at each step
)
```

This gives us a list of just 703 leaves, that still represent an equivalent
partitioning of the state space. The visual representation is shown below in which it can be seen, that there are way fewer partitions.

![Minimal partitioning](./svgs/largeStrategyNoPruneMinimalPartitioning.svg )

We now convert this list of leaves to a tree using `tree = boxes_to_tree(boxes, variables)`. Since we cannot perfectly capture the partitioning with a tree (we would need a diagram), we get a version that has slightly more leaves, namely 1,077. But this is still a considerable reduction from the original 86,807! The UPPAAL version is saved as `./strategies/large_converted_noPrune_minified.json`.

We then do the same with the zero pruned version, which goes from 6,204 leaves, to 576 after `get_boxes` and to 1,025 when converted back into a tree. In the table below their performance in UPPAAL is shown.


Strategy | Leaves | Expectation | Deviation
--- | --- | --- | --- |
No pruning | 1,077 | 38,213 | 0,174478
ZeroPruned | 1,025 | 39,871 | 0,204642
