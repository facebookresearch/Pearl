smt-comp files are not separated with filename, but please use the similar structure of test directory as the other dataset to run the simulation, you can use the following command to test for files start with `mcm`:

`python simulation.py --model_name lstm --load_file checkpoints/s_serial_pad_feature_l_z_r_200_1.pkl --test_directory data/smt-comp/QF_BV/mcm --time_selection adjust --regression`

or 

`python simulation.py --model_name KNN --load_file data/gnucore/fv2_serial/train --test_directory data/smt-comp/QF_BV/mcm --time_selection adjust`

The `data/smt-comp/QF_BV/` is the path of SMT files, and `mcm` is the filename to be tested, the whole list is as follow:
> core, app5, app2, catchconv, gulwani-pldi08, bmc-bv, app10, app8, pspace, RWS, fft, tcas, ecc, ConnectedDominatingSet, 20170501-Heizmann-UltimateAutomizer, brummayerbiere2, GeneralizedSlitherlink, 2019-Mann, mcm, zebra, uclid, samba,stp, cvs, wget,MazeGeneration, float,  20190429-UltimateAutomizerSvcomp2019,  GraphPartitioning, Sage2, app9, tacas07, 2017-BuchwaldFried, KnightTour,  WeightBoundedDominatingSet, app7, Commute,  HamiltonianPath,  2018-Mann, bench, Distrib,  openldap, dwp, inn, galois, rubik, 20170531-Hansen-Check, ChannelRouting, Booth, app6, app1, log-slicing,  2018-Goel-hwbench, bmc-bv-svcomp14, 20190311-bv-term-small-rw-Noetzli, check2, brummayerbiere,  brummayerbiere4, crafted, calypto, challenge, app12, simple, uum, pipe, VS3,  xinetd, lfsr, brummayerbiere3

Sorry for the inconvenience of the usage.