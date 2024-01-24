import json
import os
import traceback

# KLEE SMT scripts are collected in one file, separate them into single files
for root,dir,files in os.walk("data/klee/raw"):
    for file in files:
        if file == "solver-queries.smt2":
            filename = root.split("/")[-1].split("-")[0]
            ind = 0
            with open(os.path.join(root, file)) as f:
                next = False
                start = False
                script = ""
                while (True):
                    try:
                        text_line = f.readline()
                        if text_line == "":
                            break
                    except:
                        continue
                    if "(set-logic QF_AUFBV )" in text_line:
                        start = True
                    if next == True:
                        origin_time = float(text_line.split(": ")[-1][:-2])
                        with open("data/klee/single_test/" + filename + str(ind), "w") as f1:
                            json.dump({"filename": filename, "smt_script": script, "time":origin_time}, f1, indent=4)
                        start = False
                        next = False
                        script = ""
                        ind += 1
                    if start:
                        script += text_line
                    if "(exit)" in text_line:
                        next = True
            if os.path.getsize(os.path.join(root, file)) > 1024 * 1024 * 512:
                print(os.path.join(root, file))
                os.remove(os.path.join(root, file))