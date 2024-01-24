from angr import ExplorationTechnique,sim_manager
import random
import traceback

class CFS(ExplorationTechnique):
    """
    Depth-first search.

    Will only keep one path active at a time, any others will be stashed in the 'deferred' stash.
    When we run out of active paths to step, we take the longest one from deferred and continue.
    """

    def __init__(self, deferred_stash='deferred'):
        super(CFS, self).__init__()
        self._random = random.Random()
        self._random.seed(10)
        self.deferred_stash = deferred_stash
        self.executed_block_addr = []
        self.last_unexecuted = 0
        self.all_executed = False
        self.simgr = None

    def setup(self, simgr):
        if self.deferred_stash not in simgr.stashes:
            simgr.stashes[self.deferred_stash] = []

    def easy_state_select(self, simgr, stash):
        self._random.shuffle(simgr.stashes[stash])
        state_list = simgr._fetch_states(stash=stash)
        keep, executed = [], []

        # seperate state with new address
        for i in range(len(state_list)):
            state = state_list[i]
            if state.addr in self.executed_block_addr:
                executed.append(state)
            else:
                keep.append(state)

        if len(keep) == 0:
            simgr.stashes[stash] = [executed.pop()]
        else:
            simgr.stashes[stash] = keep
        executed.extend(simgr.stashes[self.deferred_stash])
        simgr.stashes[self.deferred_stash] = executed

    def state_select(self, simgr, stash):
        self._random.shuffle(simgr.stashes[stash])
        state_list = simgr._fetch_states(stash=stash)
        keep, executed = [], []

        # seperate state with new address
        for i in range(len(state_list)):
            state = state_list[i]
            if state.addr in self.executed_block_addr:
                executed.append(state)
            else:
                keep.append(state)

         # no state with new address
        if len(keep) == 0:
            if self.all_executed == True:
                simgr.stashes[stash].append(executed.pop())
                self.last_unexecuted += len(executed)
            else:
                if len(simgr.stashes[self.deferred_stash]) != 0:
                    state = simgr.stashes[self.deferred_stash].pop()
                    while (state.addr in self.executed_block_addr
                        and len(simgr.stashes[self.deferred_stash]) >= self.last_unexecuted and len(simgr.stashes[self.deferred_stash])):
                        executed.append(state)
                        state = simgr.stashes[self.deferred_stash].pop()
                    if state.addr in self.executed_block_addr:
                        simgr.stashes[stash] = [executed.pop()]
                        self.all_executed = True
                    else:
                        simgr.stashes[stash] = [state]
                    self.last_unexecuted = self.last_unexecuted + len(executed)
                else:
                    simgr.stashes[stash] = [executed.pop()]
                    self.last_unexecuted = 0
            executed.extend(simgr.stashes[self.deferred_stash])
            simgr.stashes[self.deferred_stash] = executed

        # more than one state with new address
        elif len(keep) > 1:
            simgr.stashes[stash] = [keep[0]]
            self.last_unexecuted = self.last_unexecuted + len(executed)
            executed.extend(simgr.stashes[self.deferred_stash])
            executed.extend(keep[1:])
            simgr.stashes[self.deferred_stash] = executed
            self.all_executed = False
        else:
            simgr.stashes[stash] = [keep[0]]
            self.last_unexecuted = self.last_unexecuted + len(executed)
            executed.extend(simgr.stashes[self.deferred_stash])
            simgr.stashes[self.deferred_stash] = executed

    def moved(self, state_list):
        # split, keep, seen = [], [], []
        # for state in state_list:
        #     if state.addr not in seen and state.addr not in self.addr_set:
        #         seen.append(state.addr)
        #         keep.append(state)
        #     else:
        #         split.append(state)
        keep, split = state_list[:150], state_list[150:]
        return keep, split

    def protect_merge(self, *args):
        states = list(args)
        try:
            ret = self.simgr._merge_states(states)
        except:
            return states[-1]
        return ret

    def step(self, simgr, stash='active', **kwargs):
        try:
            simgr = simgr.step(stash=stash, **kwargs)
        except:
            if len(simgr.stashes['active']) == 1:
                simgr.move("active", "stashed")
            else:
                simgr.move("active", "stashed", filter_func=lambda x:x==simgr.active[0])
        try:
            for state in simgr.stashes['active']:
                executed_addr = state.history.bbl_addrs.hardcopy[-1]
                if executed_addr not in self.executed_block_addr:
                    self.executed_block_addr.append(executed_addr)
        except IndexError:
            pass

        if len(simgr.stashes[stash]) > 1:
            self.easy_state_select(simgr, stash)

        if len(simgr.stashes['active']) > 5:
            try:
                self.simgr = simgr
                simgr.merge(stash="active", merge_func=self.protect_merge)
            except:
                traceback.print_exc()

        if len(simgr.stashes['deferred']) > 30:
            try:
                self.simgr = simgr
                simgr.merge(stash="deferred", merge_func=self.protect_merge)
                self._random.shuffle(simgr.stashes["deferred"])
                simgr.drop(stash="stashed")
                simgr.drop(stash="unconstrained")
            except:
                traceback.print_exc()
            # self.addr_set = set(map(lambda x: x.addr, simgr.stashes['deferred']))
            # simgr.split(stash_splitter=self.moved, from_stash="deferred", to_stash='stashed')
            # simgr.drop(stash="stashed")
            # print("after merge", simgr)
            # print(len(simgr.stashes['deferred']), len(self.addr_set))

        if len(simgr.stashes['deferred']) > 200:
            simgr.move("deferred", "stashed", filter_func=self.moved)
            simgr.drop(stash="stashed")

        if len(simgr.stashes[stash]) == 0:
            if len(simgr.stashes[self.deferred_stash]) == 0:
                return simgr
            simgr.stashes[stash].append(simgr.stashes[self.deferred_stash].pop())

        return simgr

