import argparse
import json
import os
from collections import defaultdict
import re

parser = argparse.ArgumentParser()
parser.add_argument('--script_input', default='./data/gnucore/single_test', help="the path of SMT files")
parser.add_argument('--adjust_log_path', default='adjustment.log', help="solving time log file path")
parser.add_argument('--prefix', default='', help="prefix of SMT file, to filter other files")
parser.add_argument("--output", default=None, help="output file path, None for using input file path")
parser.add_argument("--solver", default="z3", help="SMT solver, including 'msat', 'cvc4', 'yices', 'btor', 'z3'")
args = parser.parse_args()

time_dict = defaultdict(dict)
with open(args.adjust_log_path, "r") as f:
    data = f.read()
time_list = data.split('\n')
solver = args.solver
for time in time_list:
    try:
        if time == "":
            continue
        solver = time.split("solver: ")[-1].split(", result")[0]
        st = float(re.search("\d+\.\d+", time).group())
        if "error" in time:
            st = -1
        if args.prefix != "":
            fn = re.search(args.prefix + "\d+", time).group()
        else:
            fn = re.match('[^,]*(?=,)', time).group()
            if "/" in fn:
                fn_list = fn.split("/")
                if "smt-comp" in fn:
                    fn = "_".join(fn_list[-2:])
                else:
                    fn = fn_list[-1]
        if solver not in time_dict[fn].keys():
            time_dict[fn][solver] = [st]
        else:
            time_dict[fn][solver].append(st)
    except Exception as e:
        print(e)
        print(time)

def modify_time_representation(data):
    if not data.get("solving_time_dic") and data.get("double_check_time"):
        data["solving_time_dic"] = {"z3": data["double_check_time"]}
        data.pop("double_check_time")
    return data

if args.output == None:
    output_dir = args.script_input
else:
    output_dir = args.output
json_list = []
count = 0
for root, dir, files in os.walk(args.script_input):
    for file in files:
        # if os.path.getsize(os.path.join(root, file)) < 512 * 1024:
        #     continue
        if file.endswith("txt"):
            continue
        with open(os.path.join(root, file), "r") as f:
            try:
                data = f.read()
                data = json.loads(data)
                new_script = data["script"]
                new_script = new_script.replace("bvurem_i", "bvurem")
                new_script = new_script.replace("bvsrem_i", "bvsrem")
                new_script = new_script.replace("bvudiv_i", "bvudiv")
                new_script = new_script.replace("bvsdiv_i", "bvsdiv")
                # if len(new_script) == len(data["script"]):
                #     continue
                # else:
                #     data["script"] = new_script
                #     out = data
                # data = modify_time_representation(data)
                if len(time_dict[file]):
                    for key, item in time_dict[file].items():
                        if not data.get("solving_time_dic"):
                            data["solving_time_dic"] = time_dict[file]
                        else:
                            if key in data["solving_time_dic"].keys():
                                data["solving_time_dic"][key].extend(item)
                                check_valid = data["solving_time_dic"][key]
                                while(-1 in check_valid and len(check_valid) > 1):
                                    check_valid.remove(-1)
                            else:
                                data["solving_time_dic"][key] = item
                    # data["solving_time_dic"] = time_dict[file]
                    out = data
                    outs = {"filename": data["filename"], "time": data["time"],
                            "solving_time_dic": data["solving_time_dic"]}
                else:
                    out = data
            except IOError:
                continue
            except (KeyError,IndexError):
                data_list = data.split('\n')
                if "filename" in data_list[0]:
                    out = {"filename": data_list[0].split(' ')[1], "smt_script": '\n'.join(data_list[1:-1]),
                           "time": data_list[-1].split(' ')[1], "solving_time_dic": time_dict[file]}
                else:
                    new_file = "_".join([root.split("/")[-1], file])
                    out = {"filename": new_file, "smt_script": data,
                           "time": -1, "solving_time_dic": time_dict[new_file]}
                    file = new_file
        with open(os.path.join(root, file), "w") as f:
             f.write(json.dumps(out, indent=4))
print(count)