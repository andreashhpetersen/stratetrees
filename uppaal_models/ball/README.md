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
