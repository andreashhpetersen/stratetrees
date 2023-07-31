#!/bin/bash

S=$1

Q=$(mktemp)
echo "strategy s = loadStrategy {} -> {CartPole.cart_pos, CartPole.cart_vel, CartPole.pole_ang, CartPole.pole_vel} (\"$S\")" > $Q
echo "E[<=10;1000] (max: CartPole.num_deaths) under s" >> $Q
echo $Q
