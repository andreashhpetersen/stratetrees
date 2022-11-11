#!/bin/bash

S=$1

Q=$(mktemp)
echo "strategy s = loadStrategy {} -> {P.location, t*1.0, d*1.0} (\"$S\")" > $Q
echo "E[<=100;10000] (max:cost) under s" >> $Q
echo $Q
