# Decision tree minimization for Reinforcement Learning strategies

Run `python run_experiments.py [MODEL_NAME]` with `MODEL_NAME` being the name
of one of the model directories in `automated/`. You can add an argument `-k
10` to run the experiment 10 times and a flag `-u` to store the constructed
strategies in a UPPAAL Stratego format in
`automated/MODEL_NAME/constructed_x/`.

So for example, `python run_experiments.py isola -k 2 -u` will convert
`automated/isola/qt_strategy.json` into a decision tree, find a reduced
partition with the `max_parts` algorithm and create a new tree. This will be
done twice, the results printed to the terminal and the strategies stored in
`automated/isola/constructed_0` and `automated/isola/constructed_1`.

Further, in the `bouncing_ball` directory, there is a script to evaluate the
constructed strategies. Go to the folder and run `evaluate.sh constructed_x` to
evaluate the strategies in `automated/isola/constructed_x/`.

There is also a `make_samples.sh` script to generate samples. However, right now
the experiment script is not able to do empirical pruning based on samples as
there needs to be found a way to map locations from a sample script to a loaded
strategy.
