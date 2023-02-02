#!/bin/bash

if [ -z "$UPPAAL_PATH" ] ; then
    UPPAAL_PATH="$(pwd)/uppaal-4.1.20-stratego-11-beta1-linux64"
    export UPPAAL_PATH
fi

VERIFYTA_PATH="$UPPAAL_PATH/bin/verifyta"
export VERIFYTA_PATH=$VERIFYTA_PATH

if [ $# == 0 ] ; then
    DIRS=./automated/*
else
    DIRS=$@
fi

for MODEL_DIR in $DIRS ; do
    if [ -d $MODEL_DIR/generated/ ] ; then
        rm -rf $MODEL_DIR/generated/
    fi
    mkdir $MODEL_DIR/generated
    mkdir $MODEL_DIR/generated/dtcontrol

    model=$(basename "$MODEL_DIR")

    echo "SAMPLING for '$model'"
    $MODEL_DIR/make_samples.sh

    for SAMPLE_FILE in $MODEL_DIR/samples/* ; do
        python log2ctrl.py $SAMPLE_FILE $SAMPLE_FILE
    done

    echo "BUILDING trees for '$model'"
    python run_experiments.py $MODEL_DIR -k 1 -u
    D="generated/$(cat $MODEL_DIR/generated/smallest.txt)"

    echo "RUN DTCONTROL for '$model'"
    L=$(ls $MODEL_DIR/samples/ | sort -nr -t _ -k 2 | head -n 1)
    SAMPLES=$MODEL_DIR/samples/$L
    python make_dtcontrol.py $MODEL_DIR/$D/trees/dt_original.json $SAMPLES
    dtcontrol \
        -i $MODEL_DIR/samples/dtcontrol_samples.csv \
        -o $MODEL_DIR/generated/dtcontrol/ \
        -b $MODEL_DIR/generated/dtcontrol/benchmark.json \
        -r &> dt_control_test.txt
    python make_dtcontrol.py \
        $MODEL_DIR/generated/dtcontrol/default/dtcontrol_samples/default.dot
    # rm -rf .benchmark_suite

    echo "EVALUATE '$model' strategies"
    R=$MODEL_DIR/generated/eval_results.txt
    M=$MODEL_DIR/model.xml

    if [ -f $R ] ; then
        rm $R
    fi
    touch $R

    strategies=($MODEL_DIR/qt_strategy.json $MODEL_DIR/$D/uppaal/*)
    for S in "${strategies[@]}" ; do
        strat=${S##*/}

        echo -e "\t $strat"
        echo "EVALUATE $strat" >> $R

        Q=$($MODEL_DIR/make_eval_query.sh $S)
        $VERIFYTA_PATH $M $Q --seed $RANDOM -sWqy >> $R
        rm $Q
    done


    echo "COMBINE results for '$model'"
    python combine_results.py $MODEL_DIR

    echo \n
done
