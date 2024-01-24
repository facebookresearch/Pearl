import random
import time
import json

class query_data:
    def __init__(self):
        # total solving time
        self.sol_time = 0
        # selected query list
        self.query_list = []
        # query list length
        self.list_num = 100
        # reduce writing output times
        self.time_list = []
        # query more than limits and before it
        self.timeout_list = []
        self.query_before_timeout = []
        self.mid_time_list = []
        self.last_query = None
        self.query_index = 1
        self.time_limit = 1
        self.time_output_addr = "/home/lsc/data/time/solver_time.log"
        self.filename = ""
        self.query_output_dir = "/home/lsc/data/log/con/"
        self.output = True

    def set_attr(self, filename, time_output_addr, query_output_dir):
        if not time_output_addr:
            self.output = False
            return
        self.filename = filename
        self.time_output_addr = time_output_addr + filename + "_solver_time.log"
        self.query_output_dir = query_output_dir

    def clear(self):
        self.query_list = []
        self.time_list = []

    def update(self, query, time_delta):
        if self.output:
            # record time of query
            try:
                if time_delta > 0:
                    if time_delta < 0.0001:
                        time_delta = 0.0001
                    self.time_list.append(str(time_delta) + "\n")
                if len(self.time_list) == 1000:
                    with open(self.time_output_addr, "a") as f:
                        for time_data in self.time_list:
                            f.write(time_data)
                    self.time_list = [str(time_delta) + "\n"]
            except:
                pass

            output = {"filename": self.filename, "script": query, "time": time_delta, "stamp": str(time.localtime())}
            try:
                # # record query
                if time_delta > self.time_limit:
                    with open(self.query_output_dir + self.filename + str(self.query_index), "w") as f:
                        json.dump(output, f, indent=4)
                    self.query_list.append(output)
                #     self.query_before_timeout.append(self.last_query)
                # else:
                #     if time_delta > 10:
                #         self.mid_time_list.append(output)
                #     if len(self.query_list) < self.list_num:
                #         self.query_list.append(output)
                #     else:
                #         pro = self.list_num / self.query_index
                #         ran = random.random()
                #         if ran < pro:
                #             ran = random.randrange(0,self.list_num)
                #             self.query_list[ran] = output
                # if time_delta > 10:
                #     self.query_list.append(output)
                # else:
                #     if self.list_num == 0:
                #         self.query_list.append(output)
                #         self.list_num = 500
                #     else:
                #         self.list_num -= 1
            except:
                pass
            # if time_delta <= self.time_limit:
            #     self.last_query = output
        self.sol_time += time_delta
        self.query_index += 1
