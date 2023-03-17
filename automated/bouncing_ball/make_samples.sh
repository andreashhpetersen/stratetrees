#! /bin/bash

MODEL_DIR=$(dirname $(realpath $0))
SAMPLE_DIR=$MODEL_DIR/samples

if [[ -d $SAMPLE_DIR ]] ; then
    rm -rf $SAMPLE_DIR
fi
mkdir $SAMPLE_DIR

S='5 10 100 1000'
M=$MODEL_DIR/model.xml
for s in $S ; do
    Q=$(mktemp)
    echo "strategy s = loadStrategy {} -> {Ball(0).p, Ball(0).v} (\"$MODEL_DIR/qt_strategy.json\")" > $Q
    echo "simulate [<=300;$s] {LearnerPlayer.C, Ball(0).p, Ball(0).v} under s" >> $Q
    $VERIFYTA_PATH $M $Q -sqyW > $SAMPLE_DIR/sample_${s}.log
done
