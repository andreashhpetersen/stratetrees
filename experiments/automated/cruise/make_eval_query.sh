#!/bin/bash

S=$1

Q=$(mktemp)
echo "strategy s = loadStrategy {} -> {rVelocityEgo - rVelocityFront, rDistance} (\"$S\")" > $Q
echo "E[<=100;100] (min:rDistance) under s" >> $Q
echo $Q
