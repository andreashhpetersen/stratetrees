#!/bin/bash

S=$1

Q=$(mktemp)
if [[ "$S" == *"qt_strategy.json" ]] ; then
    echo "strategy s = loadStrategy {Light.cl} -> {x, y, E, S} (\"$S\")" > $Q
else
    echo "strategy s = loadStrategy {} -> {Light.cl, x, y, E, S} (\"$S\")" > $Q
fi
echo "E[<=100;1000] (max:Q) under s" >> $Q
echo $Q
