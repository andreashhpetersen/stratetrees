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
    echo "strategy s = loadStrategy {} -> {t*1.0, d*1.0} (\"$MODEL_DIR/qt_strategy.json\")" > $Q
    echo "simulate [#<=100;$s] {P.P, t*1.0, d*1.0} under s" >> $Q
    $VERIFYTA_PATH $M $Q &> $SAMPLE_DIR/sample_${s}.log
done
