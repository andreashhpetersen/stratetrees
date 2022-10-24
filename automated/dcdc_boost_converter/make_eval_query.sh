#!/bin/bash

S=$1

Q=$(mktemp)
echo "strategy s = loadStrategy {Converter.location} -> {x1, x2} (\"$S\")" > $Q
echo "E[<=500;100] (max:Monitor.dist*(x1+0.0 >= 0 && x1+0.0 <= 0.7 && x2+0.0 >= 14.8 && x2+0.0<=15.2))  under s" >> $Q
echo $Q
