import itertools
import json

import gc
import os
import signal
import time
import configparser
import traceback

import angr
import claripy
import psutil
import logging
# from rust_procedures import *
import output_query_data_struct
from my_dfs import CFS
from memory_protection import dfs_memory_protection, bfs_memory_protection
import sys

sys.setrecursionlimit(10000)

basedir = os.path.dirname(os.path.abspath(__file__))
cf = configparser.ConfigParser()
cf.read(basedir + "/config.ini")

log_path = cf.get("Path", "log")
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger_angr = logging.getLogger('angr')
logger_angr.setLevel(logging.CRITICAL)
# logger_angr = logging.getLogger('angr.exploration_techniques.loop_seer')
# logger_angr.setLevel(logging.DEBUG)
logger_claripy = logging.getLogger('claripy')
logger_claripy.setLevel(logging.ERROR)
console = logging.StreamHandler()
console.setLevel(logging.ERROR)
# logger.addHandler(console)

class Recorder:
    def __init__(self):
        self.exe_time = None
        self.suc_time = None
        try:
            self.query_record = claripy._backends_module.backend_z3.BackendZ3.query_record
        except:
            self.query_record = output_query_data_struct.query_data()
        self.con_time = None

    def grab_result(self):
        try:
            self.exe_time = angr.engines.vex.light.VEXMixin.exe_time
        except:
            try:
                self.exe_time = angr.engines.vex.engine.SimEngineVEX.exe_time
            except:
                self.exe_time = 0
        try:
            self.suc_time = angr.engines.SimSuccessors.suc_time
        except:
            self.suc_time = 0
        try:
            self.con_time = claripy.frontends.full_frontend.FullFrontend.con_time
        except:
            self.con_time = 0


def handler(signum, frame):
    signal.alarm(1)
    raise TimeoutError


def killmyself():
    os.system('kill %d' % os.getpid())


def sigint_handler(signum, frame):
    killmyself()


def run_symexe(path, argv_size=8, withtime=True):
    log_handler = logging.FileHandler(os.path.join(log_path, os.path.splitext(os.path.basename(path))[0] + ".log"), mode='w')
    log_handler.setLevel(logging.INFO)
    log_handler.setFormatter(formatter)
    logger.addHandler(log_handler)

    logger.info('===========================================')
    logger.info('Analysing ' + path)

    sym_argv = claripy.BVS('sym_argv', argv_size * 8)
    sym_argv2 = claripy.BVS('sym_argv2', 2 * 8)
    sym_argv3 = claripy.BVS('sym_argv3', 2 * 8)
    sym_argc = claripy.BVS('sym_argc', 1 * 8)
    try:
        load_lib_bool = cf.getboolean("Angr", "lib")
        # p2 = angr.Project(path)
        p1 = angr.Project(path, load_options={"auto_load_libs": load_lib_bool})
        p = p1
        # ad = list(map(lambda x: x.name, filter(lambda x: x.is_function, p1.loader.extern_object.symbols)))
        # try:
        #     ad.remove('getopt_long')
        # except:
        #     pass
        # p = angr.Project(path, load_options={"auto_load_libs": True})
        # for i in ad:
        #     if not p.is_hooked(p.loader.find_symbol(i).rebased_addr):
        #         p.hook_symbol(i, angr.SIM_PROCEDURES['stubs']['ReturnUnconstrained'](display_name=i, is_stub=True))
    except:
        print('Invalid path: \"' + path + '\"')
        traceback.print_exc()
        logger.error('invalid path or load lib failed')
        logger.removeHandler(log_handler)
        return
    main_obj = p.loader.main_object.get_symbol('main')
    state = p.factory.full_init_state(argc=sym_argc, args=[p.filename, sym_argv, sym_argv2, sym_argv3])
    # state = p.factory.full_init_state(args=[p.filename, sym_argv])
    # state = p.factory.entry_state(args=[p.filename, sym_argv], add_options={angr.options.LAZY_SOLVES})
    recorder = Recorder()
    pg = p.factory.simgr(state, auto_drop={"unsat", "error","spinning"})
    # add_rust_support(p)
    add_constraint_for_arg(state, sym_argv)
    draw_cfg_bool = cf.getboolean("Angr", "cfg")
    if draw_cfg_bool == True:
        cfg, executed_addr, src_addr, lib_addr = draw_cfg(p1, recorder)
    else:
        cfg, executed_addr, src_addr, lib_addr = None, set(), set(), set()

    if cf.getboolean("Angr", "output"):
        recorder.query_record.set_attr(path.split("/")[-1], cf.get("Path", "time"), cf.get("Path", "output"))
    else:
        recorder.query_record.set_attr(None, None, None)
    # return
    path_count = [0, 0]
    # try:
    #     cfg_function_list = list(map(lambda x: cfg.kb.functions[x], cfg.kb.functions))
    #     cfg_function_list1 = list(filter(lambda x:not x.has_return, cfg_function_list))
    #     with open("/home/lsc/lsc/core6_src_function.json", "r") as f:
    #         data = f.read()
    #         fun_list = json.loads(data)
    #     cfg_function_list2 = list(filter(lambda x: x.name not in fun_list["function_list"], cfg_function_list1))
    #     print(cfg_function_list2)
    #     for i in cfg_function_list2:
    #         if not p.is_hooked(i.addr):
    #             p.hook_symbol(i.name, angr.SIM_PROCEDURES['stubs']['ReturnUnconstrained'](display_name=i.name, is_stub=True))
    # except:
    #     pass

    if cf.getboolean("Technique", "DFS"):
        pg.use_technique(angr.exploration_techniques.LengthLimiter(max_length=max(2 * len(cfg.graph.nodes), 2000),drop=True))
        pg.use_technique(angr.exploration_techniques.LoopSeer(cfg=cfg, bound=3))
        pg.use_technique(dfs_memory_protection())
        pg.use_technique(angr.exploration_techniques.DFS())
    else:
        pg.use_technique(bfs_memory_protection())
    #    pg.use_technique(CFS())
    # pg.use_technique(angr.exploration_techniques.Oppologist())
    # pg.use_technique(angr.exploration_techniques.Veritesting())
    #
    # target_func = cfg.kb.functions.function(name="exit")
    # print(target_func)
    tel = cf.getint("Time", "explore")
    start_time = time.time()
    try:
        if withtime:
            def my_split(state_list):
                jump_list = []
                stay_list = []
                move_list = []
                for i in state_list:
                    if i.addr not in jump_list:
                        jump_list.append(i.addr)
                        stay_list.append(i)
                    else:
                        move_list.append(i)
                return stay_list, move_list

            def my_step_func(lpg):
                # if len(lpg.active) > 0:
                    # next_ip = lpg.active[0].ip
                    # next_addr = lpg.active[0].addr
                    # next_node = cfg.get_any_node(next_addr)
                    # print(next_ip)
                    # print(next_node)
                    # call_stack = str(lpg.active[0].callstack).split("\n")
                    # call_stack = list(map(lambda x: x[9:18], call_stack))
                    # call_stack_list = []
                    # for c in call_stack:
                    #     try:
                    #         a = cfg.get_any_node(int(c, 16))
                    #         call_stack_list.append(a)
                    #     except:
                    #         pass
                for s in lpg.active:
                    s.downsize()
                # if len(lpg.active) > 30:
                #     lpg.split(stash_splitter=my_split)
                #     lpg.drop(stash="stashed")
                # if len(lpg.stashes['active']) > 1:
                #     print(path, lpg)
                    # if len(recorder.query_record.query_list) < query_list_num:
                    #     recorder.query_record.query_list.append("branch point: " + str(lpg.active[0].history.addr))
                # if len(lpg.active):
                #     print(lpg.active[0].ip)
                    # print(lpg.active[0].ip, cfg.get_any_node(lpg.active[0].addr))
                # print(path, lpg)
                # try:
                #     lpg._errored = []
                #     for stash_name in ["spinning", "unsat"]:
                #         if stash_name in lpg._integral_stashes:
                #             lpg.drop(stash=stash_name)
                # except:
                #     pass

                if lpg.deadended:
                    for pg in lpg.deadended:
                        readable_history = list(map(hex, pg.history.bbl_addrs.hardcopy))
                        # readable_block = list(map(cfg.get_any_node, pg.history.bbl_addrs.hardcopy))
                        executed_addr.update(pg.history.bbl_addrs.hardcopy)
                        # const = pg.solver.constraints
                        # print(pg.solver.eval(pg.solver.temporal_tracked_variables[('api', '?', 'getopt_long', 1)], cast_to=bytes))
                        result = pg.solver.eval(sym_argv, cast_to=bytes)
                        # print(result)
                        # print(pg.posix.dumps(1))
                        # logger.info(result)
                    path_count[0] += len(lpg.deadended)
                    lpg.drop(stash='deadended')
                path_count[1] = len(lpg.active)

                return lpg

            # signal.signal(signal.SIGINT, sigint_handler)
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(tel)
            step_count = 0
            for _ in (itertools.count()):
                if not pg.complete() and pg._stashes['active']:
                    # pg.run()
                    if not cf.getboolean("Technique", "DFS"):
                        step_step = 1
                    else:
                        step_step = 10
                    pg.run(n=step_step, step_func=my_step_func)
                    gc.collect()
                    step_count += 1 * step_step
                    end_time = time.time()
                    time_delta = end_time - start_time
                    if time_delta > tel:
                        logger.warning('Analysing time out 2')
                        break
                    m = psutil.Process(os.getpid()).memory_percent()
                    # print(m)
                    if m > 30:
                        logger.error("use too many memory")
                        with open("stop.json", "a") as f:
                            f.write(os.path.splitext(os.path.basename(path))[0])
                        break
                    n = psutil.virtual_memory()[2]
                    if n > 90 and m > 15:
                        logger.error("use too many total memory")
                        break
                    # if step_count % 100 == 0:
                    print(path, pg)
                    # if step_count % 1000 == 0:
                    #     logger.info(str(step_count) + " blocks have been executed.")
                    continue
                break
            signal.alarm(0)
        else:
            pg.run()
    except TimeoutError:
        signal.alarm(0)
        logger.warning('Analysing time out')
    except:
        traceback.print_exc()
        signal.alarm(0)
        logger.error(traceback.format_exc(limit=1))

    test_case_addr = executed_addr
    # state = []
    # p = angr.Project("/home/lsc/gnucore/whoami", load_options={"auto_load_libs": True})
    # for i in ad:
    #     if not p.is_hooked(p.loader.find_symbol(i).rebased_addr):
    #         p.hook_symbol(i, angr.SIM_PROCEDURES['stubs']['ReturnUnconstrained'](display_name=i, is_stub=True))
    # for lpg in pg.deferred:
    #     result = lpg.solver.eval(sym_argv, cast_to=bytes)
    #     argv = claripy.BVV(result, argv_size * 8)
    #     print(argv)
    #     state.append(p.factory.full_init_state(args=[path, argv]))
    # new_pg = p.factory.simgr(state, auto_drop={"unsat"})
    # new_pg.run(step_func=my_step_func)
    # for pg in new_pg.deadended:
    #     readable_history = list(map(hex, pg.history.bbl_addrs.hardcopy))
    #     print(readable_history)
    #     test_case_addr.update(pg.history.bbl_addrs.hardcopy)

    end_time = time.time()
    time_delta = end_time - start_time
    recorder.grab_result()

    try:
        # check_coverage_correctness(cfg, executed_addr, src_addr)
        executed_addr = executed_addr.intersection(src_addr)
        block_cov = len(executed_addr) / len(src_addr)
        test_case_addr = test_case_addr.intersection(src_addr)
        test_case_cov = len(test_case_addr) / len(src_addr)
        executed_addr_lib = executed_addr.intersection(lib_addr)
        block_cov_lib = len(executed_addr_lib) / len(lib_addr)
        print(block_cov_lib)
    except Exception as e:
        print(e)
        block_cov = 0
        test_case_cov = 0
        block_cov_lib = 0

    # output all kinds of data
    logger.info("total_time: " + str(time_delta))
    log_time_info(block_cov, path_count, time_delta, recorder, path, test_case_cov, block_cov_lib)
    # test_rl = cf.getboolean("Test", "test_rl")
    # if not test_rl:
    export_solver_time_list(recorder, path)
    if cf.getboolean("Test", "con"):
        export_selected_query(path, recorder)
        export_random_query(path, recorder)
        # export_pathgroup(path, pg, sym_argv, time_delta)
    logger.removeHandler(log_handler)


def add_constraint_for_arg(state, sym_argv):
    for byte in sym_argv.chop(8):
        state.add_constraints(byte <= '~')


def check_coverage_correctness(cfg, executed_addr, src_addr):
    print(len(executed_addr))
    # print(executed_addr)
    dif_addr = executed_addr - src_addr
    no_list = []
    # print("coverage_miss_list")
    for addr in dif_addr:
        node = cfg.get_any_node(addr)
        if node:
            no_list.append(node)
    no_list = sorted(no_list, key=lambda x: x.addr)
    # for i in no_list:
    #     print(i, hex(i.addr))
    dif_addr = src_addr - executed_addr
    executed_addr = executed_addr.intersection(src_addr)
    print(len(executed_addr))
    print(len(src_addr))
    print(src_addr)
    no_list = []
    print("executed_miss_list")
    for addr in dif_addr:
        node = cfg.get_any_node(addr)
        if node:
            no_list.append(node)
    no_list = sorted(no_list, key=lambda x:x.addr)
    for i in no_list:
        print(i, hex(i.addr))
    # print("executed_list")
    yes_list = []
    for addr in executed_addr:
        node = cfg.get_any_node(addr)
        if node:
            yes_list.append(node)
    yes_list = sorted(yes_list, key=lambda x:x.addr)
    # for i in yes_list:
    #     print(i, hex(i.addr))

    return executed_addr


def draw_cfg(p, recorder):
    cfg = None
    src_addr = set()
    executed_addr = set()
    lib_addr = set()
    try:
        signal.signal(signal.SIGALRM, handler)
        cfg = p.analyses.CFGEmulated()
    except:
        traceback.print_exc()
    if not cfg:
        try:
            cfg = p.analyses.CFGFast()
        except:
            traceback.print_exc()
            pass
    # a = set(map(lambda x:x._name.split("+")[0] if x._name else "", list(cfg.graph.nodes)))
    # f = list(map(lambda x:p.loader.find_symbol(x), a))
    # b = list(filter(lambda x: not x.is_extern if x else False, f))
    try:
        if cfg != None:
            g = cfg.graph
            main_obj = p.loader.main_object.get_symbol('main')
            if main_obj != None:
                own_addr = [main_obj.linked_addr]
            else:
                own_addr = [p.entry]
            own_node = [cfg.get_any_node(main_obj.linked_addr)]
            import networkx
            main_node = cfg.get_any_node(main_obj.linked_addr)
            print(networkx.dfs_tree(cfg.graph, main_node, depth_limit=10))
            try:
                with open("/home/lsc/lsc/core6_src_function.json", "r") as f:
                    data = f.read()
                    fun_list = json.loads(data)
                own_node = list(filter(lambda x: x.name.split("+")[0] in fun_list['function_list'] if x.name else False, cfg.graph.nodes))
                own_addr = list(map(lambda x: x.addr, own_node))
                lib_addr = set(map(lambda x: x.addr, filter(lambda x:x not in own_node, cfg.graph.nodes)))
            except:
                own_node = list(filter(lambda x: x.name.split("+")[0] == "main" if x.name else False, cfg.graph.nodes))
                own_addr = list(map(lambda x: x.addr, own_node))
                i = 0
                while (i < len(own_addr)):
                    # new_node = cfg.get_any_node(own_addr[i])
                    new_node = own_node[i]
                    if new_node is not None:
                        if new_node._name:
                            if new_node._name in ["exit", "printf_parse", "quote"] \
                                    or "quote" in new_node._name or "printf" in new_node._name:
                            # if new_node._name in ["version_etc", "exit", "version_etc_va", "printf_parse", "quote"] \
                            #         or "quote" in new_node._name or "printf" in new_node._name:
                                if new_node._name != "quote_name":
                                    i += 1
                                    continue
                        for succ_block in new_node.successors:
                            # if succ_block.addr not in own_addr and succ_block.addr > main_obj.rebased_addr and succ_block.addr < 0x700000:
                            # if succ_block.addr not in own_addr:
                            if succ_block not in own_node:
                                own_addr.append(succ_block.addr)
                                # print(succ_block)
                                own_node.append(succ_block)
                            else:
                                # print(succ_block.name)
                                pass
                    i += 1
                lib_addr = set(map(lambda x: x.addr, filter(lambda x: x not in own_node, cfg.graph.nodes)))
            # own_addr = set(map(lambda x:x.addr, cfg.graph.nodes))
            src_addr = set(own_addr)
            print(p.filename, len(src_addr))
        else:
            logger.error('cfg recover failed')
            src_addr.add(None)
    except:
        traceback.print_exc()
        pass
    try:
        query_list_num = 100
        recorder.exe_time = 0
        recorder.suc_time = 0
        recorder.query_record.clear()
        recorder.query_record.list_num = query_list_num
    except:
        pass
    try:
        sol_time_dir = cf.get("Path", "time")
        recorder.query_record.time_output_addr = os.path.join(sol_time_dir, "solver_time.log")
    except:
        pass
    return cfg, executed_addr, src_addr, lib_addr


""" export query of single file"""
def export_random_query(path, recorder):
    output_dir = cf.get("Path", "output")
    output_path = os.path.join(output_dir, os.path.splitext(os.path.basename(path))[0] + "_con.json")
    try:
        output_list = []
        print(len(recorder.query_record.query_list))
        with open(output_path, 'w') as f:
            for i in recorder.query_record.query_list:
                i["filename"] = path
                output_list.append(i)
            f.write(json.dumps({"data" : output_list}, indent=4))
            recorder.query_record.query_list = []
    except:
        traceback.print_exc()
        print("output query failed")


def log_time_info(block_cov, path_count, time_delta, recorder, filename, test_case_cov, block_cov_lib):
    try:
        print(filename)
        logger.info("block coverage: " + str(block_cov))
        # logger.info("case: " + str(test_case_cov))
        logger.info("lib: " + str(block_cov_lib))
        logger.info("paths: " + str(path_count[0]))
        logger.info("solver_time: " + str(recorder.query_record.sol_time))
        logger.info("add_con_time: " + str(recorder.con_time))
        logger.info("execute_time: " + str(recorder.exe_time))
        logger.info("add_successor_time: " + str(recorder.suc_time))
        logger.info("time_per: " + str(
            (recorder.exe_time + recorder.suc_time) / time_delta))
        recorder.con_time = 0
        recorder.exe_time = 0
        recorder.suc_time = 0
        recorder.query_record.sol_time = 0
    except:
        logger.info("lib not changed")

def export_solver_time_list(recorder, filename):
    try:
        time_dir = cf.get("Path", "time")
        time_deltas = recorder.query_record.time_list
        fn = filename.split("/")[-1]
        with open(os.path.join(time_dir, fn + "_solver_time.log"), "a") as f:
            if len(time_deltas):
                # try:
                #     f.write(filename.split("/")[-1] + "\n")
                # except:
                #     pass
                for time_delta in time_deltas:
                    f.write(str(time_delta))
                recorder.query_record.time_list = []
    except:
        pass


""" export query with timespan limit"""
def export_selected_query(path, recorder):
    query_dir = cf.get("Path", "query")
    try:
        my_timeout_list = recorder.query_record.timeout_list
        with open(os.path.join(query_dir, "timeout_query.log"), "a") as f:
            for query in my_timeout_list:
                f.write("filename: " + path + "\n")
                f.write(query + "\n")
        my_query_before_timeout = recorder.query_record.query_before_timeout
        with open(os.path.join(query_dir, "timein_query.log"), "a") as f:
            for query in my_query_before_timeout:
                f.write("filename: " + path + "\n")
                f.write(query + "\n")
        mid_time_query = recorder.query_record.mid_time_list
        with open(os.path.join(query_dir, "mid_time_query.log"), "a") as f:
            for query in mid_time_query:
                f.write("filename: " + path + "\n")
                f.write(query + "\n")
    except:
        print("output timeout query failed")


def export_pathgroup(path, pg, sym_argv=None, time_delta=0):
    # output crash info
    for err in pg.errored:
        print('[-] Error: ' + repr(err))
        with open('errors.txt', 'a') as f:
            f.write(path + repr(err) + '\n')
    pg.drop(stash='active')
    try:
        pg.drop(stash='deferred')
    except:
        pass

"""
def add_rust_support(p):
    if "rust" in p.filename:
        # for obj in p.loader.initial_load_objects:
        #     for reloc in obj.imports.values():
        #         if reloc.resolvedby is not None:
        #             print(reloc.resolvedby.name, hex(reloc.resolvedby.rebased_addr))
        #         else:
        #             print(reloc)
        objs = p.loader.main_object
        # lang_start_addr = objs.get_symbol('_ZN2rt10lang_start20h58cfae38546804729kxE').rebased_addr
        p.hook_symbol('_ZN2rt10lang_start20h58cfae38546804729kxE', lang_start())
        # print_addr = objs.get_symbol('_ZN2io5stdio6_print20h47445faa595ef503E6gE').rebased_addr
        p.hook_symbol('_ZN2io5stdio6_print20h47445faa595ef503E6gE', angr.SIM_PROCEDURES['libc']['printf']())
        p.hook_symbol('_ZN6string13_$LT$impl$GT$9to_string9to_string21h12836934065809422381E', to_string())
        # p.hook_symbol('_ZN9panicking9panic_fmt20h4c8d12e3c05f3b8cZEKE', angr.SIM_PROCEDURES['stubs']['ReturnUnconstrained']())
"""

def conf_para(args, bin_path):
    print('[*] Analysing...')
    print(bin_path)
    input_length = args.length

    if input_length is None:
        input_length = cf.getint("Symvar", "length")

    try:
        run_symexe(bin_path, input_length, withtime=args.time)
    except:
        traceback.print_exc()
    print('[*] Analysis completed\n')
