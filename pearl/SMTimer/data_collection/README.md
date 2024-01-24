### angr
We provide our SMT script collection script, but you have to build the environment on your own. To collect data, you need to change `angr` dependency `claripy` in `backends/backend_z3.py`, we modify its SMT solving which in the form of `solver.check()` with a data collection function to record hard solving constraints, you may find the realization in `backend_z3.py` under the current directory. The data collection class structure is `output_query_data_struct.py`, needed to put in the same directory. For specific, our angr version is 8.19.10.30.

To run the symbolic execution, you need to compile the test files first, then you can use the script

`source run_gnu.sh`

There is some path hardcoded, so if you want to use it, you have to change it.

### KLEE
For KLEE, we use the script from the PathConstraintClassifier. You may use `-use-query-log=solver:smt2` to get the constraint model, but the data require some further work since they all save in a single file and all assertions are in one command.