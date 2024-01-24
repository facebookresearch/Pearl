import sys
import time
import json
import traceback

from six.moves import cStringIO
from pysmt.shortcuts import Solver
from pysmt.smtlib.parser import SmtLibParser

def process_data(data):
    try:
        data = json.loads(data)
        atime = float(data["time"])
        if 'smt_script' in data.keys():
            data = data["smt_script"]
        else:
            data = data["script"]
    except:
        atime = 0
    data = data.split('\n')
    start = 0
    end = len(data)
    for i in range(len(data)):
        if "declare-fun" in data[i]:
            start = i
            break
    for i in range(len(data)):
        if "check-sat" in data[i]:
            end = i
        if "time:" in data[i]:
            try:
                atime = float(data[i].split(":")[-1])
            except:
                pass
    data = '\n'.join(data[start:end + 1])
    return data, atime

if __name__ == '__main__':

    filename = sys.argv[1]
    solver_name = sys.argv[2]
    with open(filename, "r") as f:
        data = f.read()
    data, _ = process_data(data)
    # you may need to change the solver logic according to your SMT scripts reasoning theory here
    solver = Solver(name=solver_name, logic="BVt")
    parser = SmtLibParser()

    error = False
    s = time.time()
    try:
        data = data.replace("bvurem_i", "bvurem")
        data = data.replace("bvsrem_i", "bvsrem")
        data = data.replace("bvudiv_i", "bvudiv")
        data = data.replace("bvsdiv_i", "bvsdiv")
        script = parser.get_script(cStringIO(data))
        s = time.time()
        log = script.evaluate(solver)
        e = time.time()
    except Exception as a:
        traceback.print_exc()
        e = time.time()
        error = True    
        log = []
        
    """
    print json.dumps({
        'time' : e - s,
        'log' : log,
        'error' : error
    })
    """
    if error:
        res = "error"
    else:
        try:
            res = log[-1][1]
        except IndexError:
            res = "error"
    print(res, str(e - s))
    sys.stdout.flush()
