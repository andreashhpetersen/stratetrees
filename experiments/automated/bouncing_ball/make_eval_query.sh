#!/bin/bash

S=$1

Q=$(mktemp)
echo "strategy s = loadStrategy {} -> {Ball(0).p, Ball(0).v} (\"$S\")" > $Q
echo "E[<=120;1000] (max:LearnerPlayer.fired + (sum (id : ball_id) (Ball(id).number_deaths) * 1000 )) under s" >> $Q
echo $Q
