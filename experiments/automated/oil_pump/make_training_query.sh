#!/bin/bash

S=$1

Q=$(mktemp)
echo "strategy S = minE(aov)[<=120] {} -> {t, v, p, l}: <> elapsed  >= 120" >> $Q
echo "saveStrategy(\"$S\", S)" >> $Q
echo $Q

