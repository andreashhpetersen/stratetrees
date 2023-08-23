#! /bin/bash

MODEL_DIR=$(dirname $(realpath $0))
SAMPLE_DIR=$MODEL_DIR/samples

if [[ -d $SAMPLE_DIR ]] ; then
    rm -rf $SAMPLE_DIR
fi
mkdir $SAMPLE_DIR

S='10 50 100'
M=$MODEL_DIR/model.xml
for s in $S ; do
    Q=$(mktemp)
    echo "strategy s = loadStrategy {} -> {rVelocityEgo - rVelocityFront, rDistance} (\"$MODEL_DIR/qt_strategy.json\")" > $Q
    echo "simulate [<=1000;$s] {Ego.Choose, rVelocityEgo - rVelocityFront, rDistance} under s" >> $Q
    $VERIFYTA_PATH $M $Q &> $SAMPLE_DIR/sample_${s}_uppaal.log
done
