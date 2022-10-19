#!/bin/bash

if [ -z "$1" ]
then
    echo "No argument supplied"
    exit 1
fi

V="~/uppaal-4.1.20-stratego-10-linux64/bin/verifyta"
D=$1


if [ -f ./results.txt ]
then
    rm ./results.txt
fi
touch ./results.txt
for f in $D/* ; do
    Q=$(mktemp)
    echo "strategy s = loadStrategy {Light.location} -> {x, y, E, S} (\"$f\")" > $Q
    echo "E[<=100;100] (max:Q) under s" >> $Q
    CMD="$V model.xml $Q -s -W -q -y >> results.txt"
    echo "Evaluate $f"
    echo "EVALUATE $f:" >> ./results.txt
    eval "$CMD"
done
