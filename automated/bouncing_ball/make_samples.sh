#! /bin/bash

S='10 100 1000'
M=./model.xml

for s in $S ; do
    Q=$(mktemp)
    echo "strategy s = loadStrategy {} -> {Ball(0).p, Ball(0).v} (\"qt_strategy.json\")" > $Q
    echo "simulate [<=300;$s] {Ball(0).p, Ball(0).v, LearnerPlayer.C}" >> $Q
    $VERIFYTA_PATH $M $Q -sqyW > sample_${s}.log
done
