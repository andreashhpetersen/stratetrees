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
    echo "strategy s = loadStrategy {} -> {Converter.location, x1, x2} (\"$MODEL_DIR/qt_strategy.json\")" > $Q
    echo "simulate [<=500;$s] {Converter.location, Converter.location, x1, x2} under s" >> $Q
    $VERIFYTA_PATH $M $Q &> $SAMPLE_DIR/sample_${s}.log
done
