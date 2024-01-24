# Running batch jobs for code2inv
0. Ensure that your machine supports SLURM using the qsub command
1. Make sure the following directories exist- `tmp`, `logs`
    * If they don't, create them using `mkdir -p tmp logs`
2. Make sure the following files exist- `chc_numlist.txt` for the chc benchmarks, `numlist.txt` for the c benchmarks, `learning_numlist.txt` for the transferability benchmarks and `nl_numlist.txt` for the non-linear c benchmarks
3. To run the tests, run the command `python run.py <slurmscript> <numlist> <log-dir>`
    * `slurmscript` is the relevant slurm script for the test, for eg- `cluster_chc.sub` for testing the CHC benchmarks
    * `numlist` is the relevant numlist file for the test, for eg- `chc_numlist.txt` for the CHC benchmarks
    * Result logs generated, if any, would be dumped in `logs/log-dir`