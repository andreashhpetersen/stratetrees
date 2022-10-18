#! /bin/bash

S='10 100 1000'
V=~/uppaal-4.1.20-stratego-10-linux64/bin/verifyta
M=./model.xml

for s in $S ; do
    Q=$(mktemp)
    echo "strategy s = loadStrategy {} -> {Ball(0).p, Ball(0).v} (\"qt_strategy.json\")" > $Q
    echo "simulate [<=300;$s] {Ball(0).p, Ball(0).v}" >> $Q
    $V $M $Q -sqyW --sampling-time 0.5 > /dev/null
    size=$(cat ./sampling.log | wc -l)
    mv ./sampling.log ./sample_$size.log
done
