#! /bin/bash

MODEL_DIR=$(dirname $(realpath $0))
SAMPLE_DIR=$MODEL_DIR/samples

if [[ -d $SAMPLE_DIR ]] ; then
    rm -rf $SAMPLE_DIR
fi
mkdir $SAMPLE_DIR

S='1 10 100'
M=$MODEL_DIR/model.xml
for s in $S ; do
    Q=$(mktemp)
    echo "strategy s = loadStrategy {} -> {CartPole.cart_pos, CartPole.cart_vel, CartPole.pole_ang, CartPole.pole_vel} (\"$MODEL_DIR/qt_strategy.json\")" > $Q
    echo "simulate [#<=1000;$s] {Agent.location, CartPole.cart_pos, CartPole.cart_vel, CartPole.pole_ang, CartPole.pole_vel} under s" >> $Q
    $VERIFYTA_PATH $M $Q &> $SAMPLE_DIR/sample_${s}.log
done
