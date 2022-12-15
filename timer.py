# file defining components to measure the resolution time

import time

# Class Timer is used to measure the time spent in the different components of the branch & cut resolution
class Timer:
    def __init__(self):
        self.TOTAL_TIME = 0.0
        self.start_total = 0.0
        self.end_total = 0.0

        self.TIME_DEFINING_MASTER = 0.0
        self.start_define = 0.0
        self.end_define = 0.0

        self.TOTAL_TIME_REACHING_TOLERANCE = 0.0
        self.start_toler = 0.0
        self.end_toler = 0.0

        self.TOTAL_TIME_CALLBACK = 0.0
        self.start_cb = 0.0
        self.end_cb = 0.0

        self.TOTAL_TIME_BB = 0.0
        self.start_bb = 0.0
        self.end_bb = 0.0

        self.TIME_BRANCHING = 0.0
        self.start_branch = 0.0
        self.end_branch = 0.0

    def start_resolution(self):
        self.start_total = time.time()

    def end_resolution(self):
        self.end_total=time.time()-self.start_total
        self.TOTAL_TIME+=self.end_total

    def start_defining(self):
        self.start_define=time.time()

    def end_defining(self):
        self.end_define=time.time()-self.start_define
        self.TIME_DEFINING_MASTER+=self.end_define

    def start_first_phase(self):
        self.start_toler=time.time()

    def end_first_phase(self):
        self.end_toler=time.time()-self.start_toler
        self.TOTAL_TIME_REACHING_TOLERANCE+=self.end_toler

    def enter_callback(self):
        self.end_branching()
        self.start_cb=time.time()

    def leave_callback(self):
        self.end_cb=time.time()-self.start_cb
        self.TOTAL_TIME_CALLBACK+=self.end_cb
        self.start_branching()

    def start_branching(self):
        self.start_branch=time.time()

    def end_branching(self):
        self.end_branch=time.time()-self.start_branch
        self.TIME_BRANCHING+=self.end_branch

    def start_branch_and_cut(self):
        self.start_bb=time.time()
        self.start_branching()

    def end_branch_and_cut(self):
        self.end_branching()
        self.end_bb=time.time()-self.start_bb
        self.TOTAL_TIME_BB+=self.end_bb

    def print_ti(self):
        print("TIMER")
        print("{} Total seconds spent solving the problem".format(self.TOTAL_TIME))
        print("     - {} Total seconds spent defining the master problem(s)".format(self.TIME_DEFINING_MASTER))
        print("     - {} Total seconds spent reaching the tolerance".format(self.TOTAL_TIME_REACHING_TOLERANCE))
        print("     - {} Total seconds spent solving the B&B".format(self.TOTAL_TIME_BB))
        print("     - {} Total seconds spent in callbacks".format(self.TOTAL_TIME_CALLBACK))
        print("         * {} Total seconds spent branching".format(self.TIME_BRANCHING))
