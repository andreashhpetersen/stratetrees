#!/bin/bash

S=$1

Q=$(mktemp)
echo "strategy S = minE(LearnerPlayer.fired) [<=120] {} -> {p, v}: <> time >= 120" >> $Q
echo "saveStrategy(\"$S\", S)" >> $Q
echo $Q

