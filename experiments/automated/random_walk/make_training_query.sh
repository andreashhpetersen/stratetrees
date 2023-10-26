#!/bin/bash

S=$1

Q=$(mktemp)
echo "strategy S = minE(cost) [#<=100] {} -> {d, t} : <> P.Done" >> $Q
echo "saveStrategy(\"$S\", S)" >> $Q
echo $Q
