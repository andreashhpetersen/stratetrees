#!/bin/bash

S=$1

Q=$(mktemp)
echo "strategy S = minE(D) [<=120] {} -> {rVelocityEgo, rVelocityFront, rDistance}: <> time >= 120" >> $Q
echo "saveStrategy(\"$S\", S)" >> $Q
echo $Q

