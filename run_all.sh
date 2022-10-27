#!/bin/bash

V="~/uppaal-4.1.20-stratego-10-linux64/bin/verifyta"

if [ $# == 0 ] ; then
    DIRS=./automated/*
else
    DIRS=$@
fi

for MODEL_DIR in $DIRS ; do
    rm -rf $MODEL_DIR/constructed_*

    model=${MODEL_DIR##*/}
    echo "BUILDING trees for model '$model'"
    python run_experiments.py $MODEL_DIR -k 10 -u

    echo "EVALUATE '$model' strategies"
    R=$MODEL_DIR/eval_results.txt
    M=$MODEL_DIR/model.xml

    if [ -f $R ] ; then
        rm $R
    fi
    touch $R

    D=$(cat $MODEL_DIR/smallest.txt)
    strategies=($MODEL_DIR/qt_strategy.json $MODEL_DIR/$D/*)
    for S in "${strategies[@]}" ; do
        strat=${S##*/}

        echo -e "\t $strat"
        echo "EVALUATE $strat" >> $R

        Q=$($MODEL_DIR/make_eval_query.sh $S)
        eval $V $M $Q --seed $RANDOM -sWqy >> $R
        rm $Q
    done
    echo "COMBINE results for '$model'"
    python combine_results.py $MODEL_DIR
done
