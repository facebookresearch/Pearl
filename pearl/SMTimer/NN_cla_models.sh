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
    # a smaller dataset for fast get started, change learning rate because the data size change.
     python train.py --data_source data/example/single_test --input data/example/pad_feature --model lstm --time_selection z3 --threshold 200 --lr 0.1

    python train.py --input data/example/pad_feature --load_file e_pad_feature_l_z_c_200 --model lstm --time_selection z3

    python simulation.py --model_name lstm --load_file checkpoints/e_pad_feature_l_z_c_200.pkl --test_directory data/example/single_test/arch --time_selection adjust
else
    # run the nerual network model, here only represent one model,
    # LSTM to predict the solving time(regression), cross valid
    # to use other models, please refer to train.py --help and readme
    python train.py --data_source data/gnucore/single_test --input data/gnucore/pad_feature --model lstm --time_selection z3 --threshold 200 --cross_index 0

    # evaluate the trained model with program's result
    python train.py --input data/gnucore/pad_feature --load_file g_pad_feature_l_z_c_200_0 --model lstm --time_selection z3

    # you may further run the simulation for a tested program to see the solving as the order of data collection
    python simulation.py --model_name lstm --load_file checkpoints/g_pad_feature_l_z_c_200_0.pkl --test_directory data/gnucore/single_test/arch --time_selection adjust
fi
