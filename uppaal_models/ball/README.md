# Bouncing Ball examples

Model and strategy from [Peter Gjøl Jensen, et. al.](https://zenodo.org/record/3268381)


## Testing effect of empirically pruning

- Load strategy from `./strategies/large_strategy.json`
- Q-trees have a total of 13.405 leaves
- Original decision tree strategy has 104.978 leaves
    - Exported as `./strategies/large_converted_noPrune.json`
- Samples gathered from `./sampling.log` (using `./BouncingBall01Stratego.xml`)
    with `--sampling-time 0.01`
- Pruning all zero visits gives a new strategy with just 3.938 leaves
    - Exported as `./strategies/large_converted_zeroPruned.json`
- Pruning all single visits gives a strategy with 1065 leaves
    - Exported as `./strategies/large_converted_onePruned.json`
- Pruning all leaves visited twice brings the number down to 365
    - Exported as `./strategies/large_converted_twoPruned.json`

Initial results from running `E[<=120;1000] (max:LearnerPlayer.fired + (sum (id
: ball_id) (Ball(id).number_deaths) * 1000 ))` under the different strategies
in UPPAAL:

Strategy | Expectation | Deviation
--- | --- | --- |
Original | 38.352 | 0.183107
Converted (no prune) | 38.468 | 0.170944
ZeroPruned | 55.125 | 6.45953
OnePruned | 1812.65 | 56.8024
TwoPruned | 3570.96 | 42.4793

