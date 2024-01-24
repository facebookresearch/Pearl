# we split each data into three part and perform experiments on every test_rl programs selection since the result of
# different programs varies dramatically, we fix the separation to ensure result consistency.
def get_dataset_seperation(train_file):
    if "klee" in train_file:
        f1 = ['split', 'cp', 'base64', 'fmt', 'vdir', 'csplit', 'tr', 'join', 'shred', 'chcon']
        f2 = ['tail', 'nice', 'sleep', 'ginstall', 'ls', 'du', 'expr', 'date', 'stat', 'df']
        f3 = ['factor', 'chgrp', 'fold', 'head', 'nl', 'expand', 'setuidgid', 'mv', 'dir', 'tac', 'mkdir']
    elif "busybox" in train_file:
        f1 = ['raidautorun', 'smemcap', 'klogd', 'fstrim', 'cksum', 'killall5', 'mkswap', 'mt', 'mesg',
             'chroot', 'fbsplash', 'insmod', 'nice', 'ionice', 'mkfs.vfat', 'stty', 'volname', 'sulogin']
        f2 = ["adjtimex", "conspy", "fgconsole", "init", "linux", "makemime", "mv", "rmdir", "setconsole",
             "swapon", "uevent", "arp", "devmem", "ipcalc", "loadkmap", "netstat", "route","setpriv", "sync",
             "umount", "bootchartd", "dmesg", "fsync", "ipneigh", "login", "mkdir", "rpm","setserial",
             "sysctl", "usleep", "cat", "dnsdomainname", "getopt", "iprule", "logread", "mkdosfs"]
        f3 = ["nohup", "rpm2cpio", "start-stop-daemon", "timeout", "watchdog", "chattr", "dumpkmap", "gnuzip",
             "kbd_mode", "losetup", "mkfs.ext", "ping", "runlevel", "stat", "touch", "zcat", "chmod",
             "hostname", "kill", "lsattr", "printenv", "run-parts", "udhcpc", "fatattr", "ifconfig", "false",
             "lsmod", "readahead", "setarch", "swapoff", "udhcpd"]
    elif "smt-comp" in train_file:
        f1 = ['core', 'app5', 'app2', 'catchconv', 'gulwani-pldi08', 'bmc-bv', 'app10', 'app8', 'pspace', 'RWS', 'fft',
              'tcas', 'ecc', 'ConnectedDominatingSet', '20170501-Heizmann-UltimateAutomizer', 'brummayerbiere2',
              'GeneralizedSlitherlink', '2019-Mann', 'mcm', 'zebra', 'uclid', 'samba','stp', 'cvs', 'wget']
        f2 = ['MazeGeneration', 'float',  '20190429-UltimateAutomizerSvcomp2019',
             'GraphPartitioning', 'Sage2', 'app9', 'tacas07', '2017-BuchwaldFried', 'KnightTour',
             'WeightBoundedDominatingSet', 'app7', 'Commute',  'HamiltonianPath',
             '2018-Mann', 'bench', 'Distrib',  'openldap', 'dwp', 'inn']
        f3 = ['galois', 'rubik', '20170531-Hansen-Check', 'ChannelRouting', 'Booth', 'app6', 'app1', 'log-slicing',
             '2018-Goel-hwbench', 'bmc-bv-svcomp14', '20190311-bv-term-small-rw-Noetzli', 'check2', 'brummayerbiere',
             'brummayerbiere4', 'crafted', 'calypto', 'challenge', 'app12', 'simple', 'uum', 'pipe', 'VS3',
             'xinetd', 'lfsr', 'brummayerbiere3']
    else:
        f1 = ["arch", "chgrp", "csplit", "dirname", "fmt", "id", "md5sum", "mv", "pinky", "readlink", "seq",
             "sleep", "tac", "tsort", "uptime", "base64", "chmod", "cut", "du", "fold", "join", "mkdir",
             "nice", "pr", "rm"]
        f2 = ["setuidgid", "sort", "tail", "tty", "users", "basename", "chroot", "date", "expand", "ginstall",
             "link", "mkfifo", "nl", "printenv", "rmdir", "sha1sum", "split", "test_rl", "uname", "vdir",
             "cat", "comm", "df", "expr"]
        f3 = ["head", "ln", "mknod", "od", "printf", "runcon", "shred", "stat", "touch", "unexpand", "wc",
             "chcon", "cp", "dir", "factor", "hostname", "ls", "mktemp", "pathchk", "ptx", "shuf", "su",
             "tr", "unlink", "who"]
    return [f1,f2,f3]