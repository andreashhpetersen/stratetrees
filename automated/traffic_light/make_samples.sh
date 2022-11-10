#! /bin/bash

MODEL_DIR=$(dirname $(realpath $0))
# S='10 100 1000'
S='2'
M=$MODEL_DIR/model.xml

for s in $S ; do
    Q=$(mktemp)
    echo "strategy s = loadStrategy {Light.location} -> {x, y, E, S} (\"qt_strategy.json\")" > $Q
    echo "simulate [<=1000;$s] {Light.location, Light.location, x, y, E, S}" >> $Q
    $VERIFYTA_PATH $M $Q -sqyW &> $MODEL_DIR/sample_${s}.log
done
