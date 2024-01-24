#!/bin/bash

if [ -n "$1" ]
then
	if [ $1 == "example" ]
	then
		exm=1
	else
		exm=0
	fi
else
	exm=0
fi

if [ $exm == 1 ]
then
    # a smaller dataset for fast get started
    python train_KNN.py --data_source data/example/single_test --input data/example/fv2_serial --time_selection z3 --time_limit_setting 200 --model_selection increment-knn

    python simulation.py --model_name KNN --load_file data/example/fv2_serial/train --test_directory data/example/single_test/arch --time_selection adjust
else
    # run the incremental KNN, here represent the experiment for GNU coreutils,
    # for more setting of the model, please refer to train_KNN.py help and readme
    python train_KNN.py --data_source data/gnucore/single_test --input data/gnucore/fv2_serial --time_selection z3 --time_limit_setting 200 --model_selection increment-knn

    # you may further run the simulation for a tested program to see the solving as the order of data collection
    python simulation.py --model_name KNN --load_file data/gnucore/fv2_serial/train --test_directory data/gnucore/single_test/arch --time_selection adjust
fi

