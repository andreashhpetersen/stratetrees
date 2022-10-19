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
    echo "strategy s = loadStrategy {} -> {Ball(0).p, Ball(0).v} (\"$f\")" > $Q
    echo "E[<=120;1000] (max:LearnerPlayer.fired + (sum (id : ball_id) (Ball(id).number_deaths) * 1000 )) under s" >> $Q
    CMD="$V model.xml $Q -s -W -q -y >> results.txt"
    echo "Evaluate $f"
    eval "$CMD"
done
