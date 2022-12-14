#!/bin/bash

S=$1

Q=$(mktemp)
if [[ "$S" == *"qt_strategy.json" ]] ; then
    echo "strategy s = loadStrategy {P.location} -> {t*1.0, d*1.0} (\"$S\")" > $Q
else
    echo "strategy s = loadStrategy {} -> {P.location, t*1.0, d*1.0} (\"$S\")" > $Q
fi
echo "E[<=100;10000] (max:cost) under s" >> $Q
echo $Q
