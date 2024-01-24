#!/bin/bash

# to show the normal solving result, we provide the solving time log for example dataset in adjustment.log

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
    # a smaller dataset for fast get started, for full experiments you need to download the whole dataset
    python check_time.py --input data/example/single_test

    python merge_check_time.py --script_input data/example/single_test
else
    # get the solving time for single SMT scripts from the input directory, this step is mainly because the solving time is unstable inside symbolic execution, we need the label to be accurate for later prediction
    # noting that: this step is extramely time-consuming, so if you just want to see the result of prediction, you could skip this file and use the data we collected
    python check_time.py --input data/gnucore/single_test

    # add the solving time to the SMT script in json structure for later use
    python merge_check_time.py --script_input data/gnucore/single_test
fi