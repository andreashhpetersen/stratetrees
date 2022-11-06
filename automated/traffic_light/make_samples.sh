#! /bin/bash

S='10 100 1000'
M=./model.xml

for s in $S ; do
    Q=$(mktemp)
    echo "strategy s = loadStrategy {Light.location} -> {x, y, E, S} (\"qt_strategy.json\")" > $Q
    echo "simulate [<=1000;$s] {Light.location, x, y, E, S}" >> $Q
    $VERIFYTA_PATH $M $Q -sqyW &> sample_${s}.log
done
