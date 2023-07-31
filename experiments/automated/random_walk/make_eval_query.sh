#!/bin/bash

S=$1

Q=$(mktemp)
if [[ "$S" == *"qt_strategy.json" ]] ; then
    echo "strategy s = loadStrategy {P.location} -> {d*10.0, -t*10.0} (\"$S\")" > $Q
else
    echo "strategy s = loadStrategy {} -> {d*10.0, -t*10.0} (\"$S\")" > $Q
fi
echo "E[<=100;10000] (max:cost) under s" >> $Q
echo $Q
