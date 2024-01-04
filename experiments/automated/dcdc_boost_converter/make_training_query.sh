#!/bin/bash

S=$1

Q=$(mktemp)
echo "strategy S = minE(Monitor.dist) [<=1000] {} -> {x1, x2, x_R}: <> time >= 1000" >> $Q
echo "saveStrategy(\"$S\", S)" >> $Q
echo $Q

