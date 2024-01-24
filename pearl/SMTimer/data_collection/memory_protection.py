import sys
from angr import ExplorationTechnique,sim_manager
import random
import traceback
import collections
import psutil
import os


def handler_error(simgr, e):
    try:
        if len(simgr.stashes['active']) == 1:
            simgr.move("active", "stashed")
            return
        e.__traceback__ = sys.exc_info()[2]
        a = sys.exc_info()[2].tb_next
        while (a and "sim_manager" not in a.tb_frame.__str__()):
            a = a.tb_next
        if a:
            error_state = a.tb_frame.f_locals['state']
            simgr.move(from_stash='active', to_stash='stashed', filter_func=lambda x: x == error_state)
            simgr.drop(stash="stashed")
            del error_state
        del a
        del e
    except:
        pass

def _merge_key(state):
    return (state.addr if not state.regs._ip.symbolic else 'SYMBOLIC',
            set(x.func_addr for x in state.callstack))

class dfs_memory_protection(ExplorationTechnique):
    """
    Depth-first search.

    Will only keep one path active at a time, any others will be stashed in the 'deferred' stash.
    When we run out of active paths to step, we take the longest one from deferred and continue.
    """

    def __init__(self, deferred_stash='deferred'):
        super(dfs_memory_protection, self).__init__()
        self._random = random.Random()
        self._random.seed(10)
        self.deferred_stash = deferred_stash
        self.executed_block_addr = collections.defaultdict(int)
        self.last_unexecuted = 0
        self.all_executed = False
        self.simgr = None
        self.last_deferred_num = 0

    def setup(self, simgr):
        if self.deferred_stash not in simgr.stashes:
            simgr.stashes[self.deferred_stash] = []

    def easy_state_select(self, simgr):
        keep, seen = [], []
        for state in simgr.stashes['active']:
            if state.addr not in seen:
                seen.append(state.addr)
                keep.append(state)
        simgr.stashes['active'] = keep
        # if self.executed_block_addr[keep[0].history.bbl_addrs.hardcopy[-1]] > 5:
        #     if len(keep) > 1:
        #         print("cut path at" + str(keep[0].history.bbl_addrs.hardcopy[-1]))
        #         simgr.stashes['active'] = []
        #     else:
        #         self._random.shuffle(keep)
        #         simgr.stashes['active'] = [keep[0]]

    def moved(self, state_list):
        keep, split = state_list[-50:], state_list[:-50]
        return keep, split

    def protect_merge(self, *args):
        states = list(args)
        # try:
        #     ret = self.simgr._merge_states(states)
        # except:
        #     return states[-1]
        # del self.simgr
        return states[-1]

    def step(self, simgr, stash='active', **kwargs):
        try:
            simgr = simgr.step(stash=stash, **kwargs)
        except TimeoutError:
            raise TimeoutError
        except Exception as e:
            handler_error(simgr, e)

        for s in simgr.active:
            s.downsize()
        #
        # try:
        #     for state in simgr.stashes['active']:
        #         executed_addr = state.addr
        #         self.executed_block_addr[executed_addr] += 1
        # except IndexError:
        #     pass
        #
        # if len(simgr.stashes[stash]) > 2:
        #     self.easy_state_select(simgr)

        if len(simgr.stashes['deferred']) > 1.5 * self.last_deferred_num or self.last_deferred_num > 200 or \
                len(simgr.stashes['active']) == 0:
            try:
                self.simgr = simgr
                simgr.merge(stash="deferred", merge_key=_merge_key, merge_func=self.protect_merge)
            except TimeoutError:
                raise TimeoutError
            except:
                traceback.print_exc()
            # simgr.stashes['stashed'].extend(simgr.stashes['deferred'][150:])
            # simgr.stashes['deferred'] = simgr.stashes['deferred'][:150]
            # simgr.drop(stash='stashed')
            # simgr.stashes['deferred'][150:] = []

        # if len(simgr.stashes['active']) > 50:
        #     try:
        #         simgr.stashes['deferred'].extend(simgr.stashes['active'][:-50])
        #         simgr.stashes['active'] = simgr.stashes['active'][-50:]
        #     except:
        #         traceback.print_exc()

        # if len(simgr.stashes['deferred']) > 30:
        #     try:
        #         self.simgr = simgr
        #         simgr.merge(stash="deferred", merge_func=self.protect_merge)
        #         self._random.shuffle(simgr.stashes["deferred"])
        #         simgr.drop(stash="stashed")
        #         simgr.drop(stash="unconstrained")
        #     except:
        #         traceback.print_exc()

        if len(simgr.stashes[stash]) == 0:
            if len(simgr.stashes[self.deferred_stash]) == 0:
                return simgr
            simgr.stashes[stash].append(simgr.stashes[self.deferred_stash].pop())
        self.last_deferred_num = len(simgr.stashes['deferred'])

        return simgr

class bfs_memory_protection(ExplorationTechnique):
    """
    Depth-first search.

    Will only keep one path active at a time, any others will be stashed in the 'deferred' stash.
    When we run out of active paths to step, we take the longest one from deferred and continue.
    """

    def __init__(self, deferred_stash='deferred'):
        super(bfs_memory_protection, self).__init__()
        self._random = random.Random()
        self._random.seed(10)
        self.deferred_stash = deferred_stash
        self.executed_block_addr = []
        self.last_unexecuted = 0
        self.all_executed = False
        self.simgr = None
        self.last_active_num = 0

    def setup(self, simgr):
        if self.deferred_stash not in simgr.stashes:
            simgr.stashes[self.deferred_stash] = []

    def easy_state_select(self, simgr):
        keep, seen, split = [], [], []
        for state in simgr.active:
            if state.addr not in seen:
                seen.append(state.addr)
                keep.append(state)
            else:
                split.append(state)
        simgr.active = keep
        # simgr.deferred.extend(split)

    def moved(self, state_list):
        keep, split = state_list[-50:], state_list[:-50]
        return keep, split

    def protect_merge(self, *args):
        states = list(args)
        # try:
        #     ret = self.simgr._merge_states(states)
        # except TimeoutError:
        #     raise TimeoutError
        # except:
        #     return states[-1]
        # del self.simgr
        return states[0]

    def step(self, simgr, stash='active', **kwargs):
        try:
            simgr = simgr.step(stash=stash, **kwargs)
        except TimeoutError:
            raise TimeoutError
        except Exception as e:
            handler_error(simgr, e)

        for s in simgr.active:
            s.downsize()

        if len(simgr.active) > 1.5 * self.last_active_num or self.last_active_num > 100:
            try:
                self.simgr = simgr
                simgr.merge(stash="active", merge_key=_merge_key, merge_func=self.protect_merge)
            except TimeoutError:
                raise TimeoutError
            except:
                traceback.print_exc()

        # self.easy_state_select(simgr)
        #
        if len(simgr.stashes['active']) > 200:
            simgr.split(from_stash='active', to_stash='deferred', limit=100)

        if len(simgr.stashes['deferred']) > 200:
            simgr.drop(stash='deferred')
        #     simgr.stashes['stashed'].extend(simgr.stashes['deferred'][:-150])
        #     simgr.stashes['deferred'] = simgr.stashes['deferred'][-150:]
        #     simgr.drop(stash='stashed')
        #
        # if len(simgr.stashes['deferred']) > 30:
        #     try:
        #         self.simgr = simgr
        #         simgr.merge(stash="deferred", merge_func=self.protect_merge)
        #         # self._random.shuffle(simgr.stashes["deferred"])
        #         simgr.drop(stash="stashed")
        #         simgr.drop(stash="unconstrained")
        #     except TimeoutError:
        #         raise TimeoutError
        #     except:
        #         traceback.print_exc()

        if len(simgr.stashes[stash]) == 0:
            if len(simgr.stashes[self.deferred_stash]) == 0:
                return simgr
            simgr.stashes[stash].append(simgr.stashes[self.deferred_stash].pop())
        self.last_active_num = len(simgr.active)

        return simgr


