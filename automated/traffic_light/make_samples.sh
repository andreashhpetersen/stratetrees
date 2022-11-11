#! /bin/bash

MODEL_DIR=$(dirname $(realpath $0))
SAMPLE_DIR=$MODEL_DIR/samples

if [[ -d $SAMPLE_DIR ]] ; then
    rm -rf $SAMPLE_DIR
fi
mkdir $SAMPLE_DIR

S='10 100 1000'
M=$MODEL_DIR/model.xml
for s in $S ; do
    Q=$(mktemp)
    echo "strategy s = loadStrategy {Light.location} -> {x, y, E, S} (\"$MODEL_DIR/qt_strategy.json\")" > $Q
    echo "simulate [<=1000;$s] {Light.location, Light.location, x, y, E, S}" >> $Q
    $VERIFYTA_PATH $M $Q &> $SAMPLE_DIR/sample_${s}.log
done
