import json

import argparse

import configparser
import os
import platform
if "Windows" in platform.platform():
    from symexe_win import conf_para
elif "Linux" in platform.platform():
    from symexe import conf_para
import multiprocessing as mp

basedir = os.path.dirname(os.path.abspath(__file__))
cf = configparser.ConfigParser()
cf.read(basedir + "/config.ini")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dir", help="directory of file")
    parser.add_argument("-c", "--constraints", help="Deprecated: Show generated model", action="store_true")
    parser.add_argument("-C", "--compile", type=int,
                        help="Deprecated: Compile from source, if C > 0, -O option will be used")
    parser.add_argument("-l", "--length", type=int, help="Stdin size")
    parser.add_argument("-r", "--run_program", help="Run program after analysis", action="store_true")
    parser.add_argument("-s", "--summary", type=int, help="Deprecated: Display summary information")
    parser.add_argument("-e", "--expected", type=int, help="Deprecated: Expected amount of results")
    parser.add_argument("-f", "--file_path", type=str, help="file name path")
    parser.add_argument("-t", "--time", help="without time constraint", action="store_false")
    parser.add_argument("-de", "--debug", help="Deprecated: ctrl+c to debug the progress", action="store_true")
    parser.add_argument("-fl", "--file_list", help="list of file name")
    args = parser.parse_args()

    # clear the result of last experiment
    stoplist = []
    test = cf.getboolean("Test", "test_rl")
    if not test:
        try:
            data_dir = "/home/bz302/data/PycharmProjects/angr"
            if os.path.exists(os.path.join(data_dir, "query/timeout_query.log")):
                os.remove(os.path.join(data_dir, "query/timeout_query.log"))
            if os.path.exists(os.path.join(data_dir, "query/timein_query.log")):
                os.remove(os.path.join(data_dir, "query/timein_query.log"))
            if os.path.exists(os.path.join(data_dir, "time/solver_time.log")):
                os.remove(os.path.join(data_dir, "time/solver_time.log"))
            if os.path.exists(os.path.join(data_dir, "query/mid_time_query.log")):
                os.remove(os.path.join(data_dir, "query/mid_time_query.log"))
        except:
            pass
    try:
        pass
        # with open("stop.json", 'r') as f:
        #     jsondata = f.read()
        # stoplist = json.loads(jsondata)["stoplist"]
    except:
        stoplist = []

    # process the file under the directory with multiprocess
    if args.dir is not None:
        dirpath = args.dir
        count = 0
        pro_list = []
        filename_list = args.file_list
        pool = mp.Pool(processes=8, maxtasksperchild=1)
        for root, dirs, files in os.walk(dirpath):
            if filename_list is not None:
                files = filename_list.split(",")
            for file in files:
                # if not os.access(file,os.X_OK):
                #     continue
                # count += 1
                # if count > 7:
                #     break
                if file in stoplist:
                    continue
                if ".c" in file:

                    print('[*] Compiling...')
                    bin_dir, filename = root, file
                    bin_path = filename.split(".")[0] + ".out"
                    if ".cpp" in file:
                        cmd = ' '.join(['g++ -o', os.path.join(bin_dir, bin_path),
                                        '-O1', os.path.join(bin_dir, filename)])
                    else:
                        cmd = ' '.join(['gcc -o', os.path.join(bin_dir, bin_path),
                                        '-O1', os.path.join(bin_dir, filename)])
                    print(cmd)
                    os.system(cmd)
                    print('[*] Compile completed\n')
                    bin_path = os.path.join(bin_dir, bin_path)
                else:
                    bin_path = os.path.join(root, file)
                pool.apply_async(conf_para, (args, bin_path))
            #     pid = mp.Process(target=conf_para, args=(args, bin_path))
            #     pro_list.append(pid)
            #     if len(pro_list) == 8:
            #         for pid in pro_list:
            #             pid.start()
            #         for pid in pro_list:
            #             pid.join()
            #         pro_list = []
            # for pid in pro_list:
            #     pid.start()
            # for pid in pro_list:
            #     pid.join()
            pool.close()
            pool.join()

    if args.file_path is not None:

        src_path = args.file_path
        bin_path = ""
        if src_path in stoplist:
            pass
        elif ".c" in src_path:
            bin_path = src_path.split(".")[0] + '.out'
            print('[*] Compiling...')

            cmd = ' '.join(['gcc -o', bin_path, '-O1', src_path])

            print(cmd)
            os.system(cmd)
            print('[*] Compile completed\n')
        else:
            bin_path = src_path
        conf_para(args, bin_path)

    print("all program ran")
