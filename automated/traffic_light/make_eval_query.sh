#!/bin/bash

S=$1

Q=$(mktemp)
echo "strategy s = loadStrategy {Light.location} -> {x, y, E, S} (\"$S\")" > $Q
echo "E[<=100;1000] (max:Q) under s" >> $Q
echo $Q
