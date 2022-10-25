# Decision tree minimization for Reinforcement Learning strategies

Run `./run_all.sh` to run experiments on all models in `automated/`. This will
first construct and minimize decision trees from
`automated/MODEL_DIR/qt_strategy.json`, then evaluate the constructed
strategies with queries from `automated/MODEL_DIR/make_eval_query.sh`, and
finally combine the DT results (size and time) with the evaluation results
(expectation), storing the output in `automated/MODEL_DIR/combined_results.csv`.

The script can also be run for only a subset of models, by given their
directories as arguments to the script. So for example, `./run_all.sh
automated/traffic_light automated/isola` only experiments on `traffic_light` and
`isola` (note that `bouncing_ball` is the time consumer here).

There is also a `make_samples.sh` script to generate samples. However, right now
the experiment script is not able to do empirical pruning based on samples as
there needs to be found a way to map locations from a sample script to a loaded
strategy.
