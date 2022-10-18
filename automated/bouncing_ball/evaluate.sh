#!/bin/bash

file="./queries.q"
V="~/uppaal-4.1.20-stratego-10-linux64/bin/verifyta"

rm ./results.txt
touch ./results.txt
for f in ./constructed_0/* ; do
    Q=$(mktemp)
    echo "strategy s = loadStrategy {} -> {Ball(0).p, Ball(0).v} (\"$f\")" > $Q
    cat "./eval_queries.q" >> $Q
    CMD="$V model.xml $Q -s -W -q -y >> results.txt"
    echo "$CMD"
    eval "$CMD"
done
