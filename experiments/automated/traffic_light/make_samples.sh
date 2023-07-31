#! /bin/bash

MODEL_DIR=$(dirname $(realpath $0))
SAMPLE_DIR=$MODEL_DIR/samples

if [[ -d $SAMPLE_DIR ]] ; then
    rm -rf $SAMPLE_DIR
fi
mkdir $SAMPLE_DIR

S='10'
M=$MODEL_DIR/model.xml
for s in $S ; do
    Q=$(mktemp)
    echo "strategy s = loadStrategy {Light.cl} -> {x, y, E, S} (\"$MODEL_DIR/qt_strategy.json\")" > $Q
    echo "simulate [<=1000;$s] {Light.location == 2 || Light.location == 3, Light.cl, x, y, E, S} under s" >> $Q
    $VERIFYTA_PATH $M $Q &> $SAMPLE_DIR/sample_${s}.log
done
