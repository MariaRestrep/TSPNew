# contains the classes and functions to parse the data plus some functions to execute the arguments parsed from arguments.py

import copy
import numpy as np
import pandas as pd

from exporter import *
from model_weekly import *
from branch_and_cut import *
from arguments import *
from pricing import *
from grammars import *
from grammars_2 import *

# default shifts and patterns parameters (values are changed after reading arguments)
NB_TIME_PERIODS = 0
MAX_HOUR = 0
MIN_HOUR = 0

fixed_cost = 13 # fixed cost per hour
agglo_var_cost_coef = 1 # variable cost coefficient increase when delivering in the agglomeration (variable cost depends on the vehicle)
agglo_ext_cost_coef = 1 # external cost coefficient increase when delivering in the agglomeration
external_cost_intra = 25
external_cost_agglo = external_cost_intra * agglo_ext_cost_coef
pattern_fixed_cost = 0  # fixed cost of assigning a pattern
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

SEED = 1 # seed to assign the same vehicles to couriers at each run

COLLECTION = "COLLECTION"

instances_dir="./instances/{}/".format(COLLECTION)
bench_dir="./benchmarks/{}".format(COLLECTION)

class TypeVehicle:

    def __init__(self, id_vehicle, idVS, name, maxWeight, maxHeight, avSpeed, threshold, vehicleWearCost,
                 pickupCost, deliveryCost):
        self.id_vehicle = int(id_vehicle)
        self.idVS = idVS
        self.name = name
        self.maxWeight = float(maxWeight)
        self.maxHeight = float(maxHeight)
        self.avSpeed = float(avSpeed)
        self.threshold = float(threshold)
        self.vehicleWearCost = float(vehicleWearCost)
        self.pickupCost = float(pickupCost)
        self.deliveryCost = float(deliveryCost)

class PostalCode:

    def __init__(self, postal_code, pc_id, latitude, longitude, zone):

        self.postal_code = int(postal_code)
        self.pc_id = int(pc_id)
        self.latitude = float(latitude)
        self.longitude = float(longitude)
        self.zone = int(zone)


class Employee:

    def __init__(self, id_employee, id_vehicle, start_availability, end_availability, fixed_cost, var_cost_intra, var_cost_aglo, capacity):

        self.id_employee = int(id_employee)
        self.id_vehicle = int(id_vehicle)
        self.start_availability = int(start_availability) # an integer from 0 to 23. It could also be a datetime type
        self.end_availability = int(end_availability) # an integer from 0 to 23. It could also be a datetime type
        self.fixed_cost = float(fixed_cost)
        self.var_cost_intra = float(var_cost_intra)
        self.var_cost_aglo = float(var_cost_aglo)
        self.capacity = float(capacity)

class Shift:

    def __init__(self, idShift, idShiftString, cost, workTime, maxHeight, shift, startPeriod, endPeriod, timesZero, timesOne, fixed, min_duration, max_duration, tws, twl):
        self.idShift = idShift
        self.idShiftString = idShiftString
        self.cost = cost
        self.workTime = workTime
        self.maxHeight = maxHeight
        self.shift = shift
        self.startPeriod = startPeriod
        self.endPeriod = endPeriod
        self.timesZero = timesZero
        self.timesOne = timesOne
        self.fixed = fixed
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.tws = int(tws)
        self.twl = int(twl)


class ODPair:

    def __init__(self, day, pc_pickup, pc_delivery, id_area1, id_area2, origin, nb_scenarios, demand_od_pair, distance, probability, zone):
        self.pc_pickup = int(pc_pickup)
        self.pc_delivery = int(pc_delivery)
        self.id_area1 = int(id_area1)
        self.id_area2 = int(id_area2)
        self.origin = origin
        self.nb_scenarios_d = [[0]*NB_TIME_PERIODS]*len(DAYS) # nb_scenarios[d]: nb scenarios per time period at day d
        self.nb_scenarios_d[day] = nb_scenarios
        self.demand_od_pair_d = [[]]*len(DAYS) # list list of size nb_scenarios_d[d][i] at each time period i at dat d
        self.demand_od_pair_d[day] = demand_od_pair
        self.distance = None
        self.probability_d = [[]]*len(DAYS)
        self.probability_d[day] = probability
        self.zone = zone

    def compute_distance(self, lat1, lon1, lat2, lon2):
        if(lon1 == lon2 and lat1 == lat2):
            self.distance = 1
        else:
            lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
            self.distance =round( 6371 * (math.acos(math.sin(lat1) * math.sin(lat2) + math.cos(lat1) * math.cos(lat2) * math.cos(lon1 - lon2))))

    def add_demand(self, day, nb_scenarios, demand_od_pair, probability):
        self.demand_od_pair_d[day] = demand_od_pair
        self.nb_scenarios_d[day] = nb_scenarios
        self.probability_d[day] = probability

# return shift corresponding to a day-off
def neutral_shift():
    return Shift(-1, "Day off", 0, 0, 0, [0]* NB_TIME_PERIODS, 0, 0, 0, 0, 0, 0, 0)

class Pattern:

    def __init__(self, id_pattern, shifts):
        self.id_pattern = id_pattern
        self.shifts = shifts
        self.weekly_hours = 0
        self.compute_weekly_hours()

    def compute_weekly_hours(self):
        self.weekly_hours = 0
        for s in self.shifts:
            self.weekly_hours += s.workTime

    def assign_shift(self, day, shift):
        self.shifts[day] = shift
        self.compute_weekly_hours()

    # return true if the period is a working period in the pattern for the day
    def is_working_period(self, day, period):
        return self.shifts[day].startPeriod <= period and self.shifts[day].endPeriod >= period and self.shifts[day].idShift!=-1

    #TODO: Extend to part time and full time? Grammars?
    # return true if the pattern is feasible (respect the weekly working rules: weekly hours + rest day)
    def is_feasible(self):
        rest_day = False
        for sh in self.shifts:
            if sh.idShift == -1:
                rest_day = True
        legal_hours = self.weekly_hours >= MIN_WORKING_HOURS and self.weekly_hours <= MAX_WORKING_HOURS #feasibility only in termns of legal hours
        return rest_day and legal_hours

class ProbData:

    def __init__(self):

        # read the data in filename
        self.vehicles = []
        self.postal_codes = []
        self.incompatibilities = []
        self.employees = []
        self.courierToVeh = []
        self.shifts = []
        self.shift_types = []
        self.od_pairs = []
        self.od_pairs_generated = {}
        self.patterns = []

    #TODO: change vehicles
    def add_vehicle(self, id_vehicle, idVS, name, maxWeight, maxHeight, avSpeed, threshold, vehicleWearCost,
                 pickupCost, deliveryCost):
        self.vehicles.append(TypeVehicle(id_vehicle, idVS, name, maxWeight, maxHeight, avSpeed, threshold, vehicleWearCost,
                                         pickupCost, deliveryCost))
        return self.vehicles[-1]

    def add_employee(self, id_employee, id_vehicle, start_availability, end_availability, fixed_cost, var_cost_intra, var_cost_aglo, capacity):
        self.employees.append(Employee(id_employee, id_vehicle, start_availability, end_availability, fixed_cost, var_cost_intra, var_cost_aglo, capacity))
        return self.employees[-1]

    def add_postal_code(self, postal_code, pc_id, latitude, longitude, zone):
        self.postal_codes.append(
            PostalCode(postal_code, pc_id, latitude, longitude, zone))
        return self.postal_codes[-1]

    def add_shift(self, idShift, idShiftString, cost, workTime, maxHeight, shift, startPeriod, endPeriod,
                  timesZero, timesOne, fixed, min_duration, max_duration, tws, twl):
        a = Shift(idShift, idShiftString, cost, workTime, maxHeight, shift, startPeriod, endPeriod,
                                 timesZero, timesOne, fixed, min_duration, max_duration, tws, twl)
        self.shifts.append(a)
        return self.shifts[-1]

    def add_od_pair(self, pc_pickup, pc_delivery, id_area1, id_area2, origin, nb_scenarios, demand_od_pair, distance, probability, zone):
        self.od_pairs.append(ODPair(pc_pickup, pc_delivery, id_area1, id_area2, origin, nb_scenarios, demand_od_pair, distance, probability, zone))

        return self.od_pairs[-1]

    def add_od_pair_day(self, day, pc_pickup, pc_delivery, id_area1, id_area2, origin, nb_scenarios, demand_od_pair, distance, probability, zone):
        try:
            data.od_pairs_generated[(pc_pickup, pc_delivery)].add_demand(day, nb_scenarios, demand_od_pair, probability)
        except:
            self.od_pairs.append(ODPair(day, pc_pickup, pc_delivery, id_area1, id_area2, origin, nb_scenarios, demand_od_pair, distance, probability, zone))
            data.od_pairs_generated[(pc_pickup, pc_delivery)]=self.od_pairs[-1]

        return self.od_pairs[-1]

    #TODO: Check this
    def add_pattern(self, shifts, check_feasibility=False):
        pat = Pattern(len(self.patterns), shifts)
        if check_feasibility:
            if pat.is_feasible():
                self.patterns.append(pat)
        else:
            self.patterns.append(pat)

    def generate_couriers(self, nb_couriers, full_avail=False):
        data.employees=[]
        random.seed(SEED)
        for i in range(int(nb_couriers)):
            start_availability = MIN_HOUR
            end_availability = MAX_HOUR
            availability_duration = random.choice(AVAILABILITIES_DURATION)
            if full_avail: availability_duration = NB_TIME_PERIODS
            if end_availability-availability_duration > 0:
                start_availability = random.choice(range(start_availability, 2+end_availability-availability_duration, 2))
            else:
                start_availability = 0
            end_availability = start_availability + availability_duration - 1
            veh = random.choice(data.vehicles)
            var_cost = veh.pickupCost + veh.deliveryCost
            data.add_employee(id_employee=i, id_vehicle=veh.id_vehicle,
                              start_availability=start_availability,
                              end_availability=end_availability, fixed_cost=fixed_cost+veh.vehicleWearCost,
                              var_cost_intra=var_cost, var_cost_aglo=var_cost * agglo_var_cost_coef,
                              capacity=veh.maxWeight)

    #TODO: Check when this is used
    def random_pattern(self, id_pattern):
        day_off = random.choice([0, 1, 2, 3, 4, 5, 6])
        shifts = [neutral_shift(), neutral_shift(), neutral_shift(), neutral_shift(), neutral_shift(), neutral_shift()]
        days = [i for i in range(6) if i != day_off]
        for i in days:
            shifts[i] = random.choice(data.shifts)
        return Pattern(id_pattern, shifts)

    def random_patterns(self, nb_patterns, seed):
        random.seed(seed)
        self.patterns = []
        id_pattern = 0
        while len(self.patterns) < nb_patterns:
            pattern = self.random_pattern(id_pattern)
            #TODO: have a parameter to change the weekly hours
            if pattern.weekly_hours >= 30 and pattern.weekly_hours <= 40:
                self.patterns.append(pattern)
                id_pattern += 1

    #todo: check from here the pattern generation
    #done
    def generate_patterns_from_shifts(self):
        # reindex the shifts
        id_shift=0
        for shift in data.shifts:
            shift.idShift=id_shift
            id_shift+=1

        # add resting day in shifts if not already present
        resting_day_added = False
        for shift in data.shifts:
            if shift.idShift == -1: resting_day_added=True
        if not resting_day_added: data.shifts.append(neutral_shift())

        # construct patterns (patterns are lists of id shifts)
        patterns = [[s.idShift] for s in self.shifts] # 1st day shifts
        for d in range(1, len(DAYS)): # for all other days
            new_patterns = []
            for id_pat in range(len(patterns)): # add each shift in all patterns
                pattern=patterns[id_pat]
                for id_sh in range(len(self.shifts)):

                    shift=self.shifts[id_sh]
                    if (shift.idShift != -1):
                        new_pattern=pattern.copy()
                        new_pattern.append(shift.idShift)
                        new_patterns.append(new_pattern)
                    elif (shift.idShift == -1 and (d == 1 or d >= 5)):
                        new_pattern = pattern.copy()
                        new_pattern.append(shift.idShift)
                        new_patterns.append(new_pattern)

            patterns = []
            for pattern in new_patterns: patterns.append(pattern)

        correct_patterns = []
        for pattern in patterns:
            if has_resting_day(pattern) and legal_working_hours(pattern) and max_rest_day(pattern) and position_rest(pattern):
                correct_patterns.append(pattern)

        id_pattern=0
        self.patterns=[]
        for pattern in correct_patterns:
            shifts=[]
            for id_shift in pattern: shifts.append(data.shifts[id_shift])
            pat=Pattern(id_pattern, shifts)
            #pat.compute_weekly_hours()
            self.patterns.append(pat)
            id_pattern+=1
            #print(pattern)


        #exit(1)

        #print ("correct patterns", len(correct_patterns))

        #print (correct_patterns)

    def filter_incompatible_patterns(self):
        #print("Filtering patterns")
        new_patterns=[]
        before_filter=len(self.patterns)
        for t in self.patterns:
            rest_day=False
            for sh in t.shifts:
                if sh.idShift==-1: rest_day=True
            if rest_day and t.weekly_hours >= MIN_WORKING_HOURS and t.weekly_hours <= MAX_WORKING_HOURS:
                new_patterns.append(t)
        #print(before_filter-len(new_patterns), "patterns filtered")
        self.patterns=new_patterns

def has_resting_day(pattern):
    rest=False
    for id_shift in pattern:
        if id_shift == -1:
            rest = True
    return rest

def max_rest_day(pattern):
    rest=False
    rd = 0
    for id_shift in pattern:
        if id_shift == -1:
            rd += 1
    if rd<=2: rest=True #TODO: Generalize!

    return rest

def position_rest(pattern):
    rest=False
    if pattern[0] != -1 and pattern[1] == -1:
        return rest
    elif pattern[6] != -1 and pattern[5] == -1:
        return rest
    else:
        rest = True
    return rest

def legal_working_hours(pattern):
    working_hours = 0
    for id_shift in pattern:
        if id_shift>=0:
            shift=data.shifts[id_shift]
            working_hours+=shift.workTime
    legal = working_hours >= MIN_WORKING_HOURS and working_hours <= MAX_WORKING_HOURS
    return legal


#TODO: to check the allocation of  days off
# make a pattern feasible wrt te weekly working rules (by adding or removing shifts)
def make_feasible(data, pattern):
    working=[0]*len(DAYS) # 1 if working on day d 0 otherwise
    sum_w = 0
    for d in range(len(DAYS)):
        if pattern.shifts[d].workTime > 0:
            working[d]=1
            sum_w+=pattern.shifts[d].workTime
    rest=0 in working
    # not enough working hours
    if sum_w < MIN_WORKING_HOURS:
        while sum_w<MIN_WORKING_HOURS:
            rest_days=[]
            for d in range(len(DAYS)):
                if not working[d]: rest_days.append(d)
            r=random.choice(rest_days)
            sh=random.choice(data.shifts)
            pattern.assign_shift(r, sh)
            sum_w+=sh.workTime
            working[r]=sh.workTime>0
    # too may working hours
    if sum_w > MAX_WORKING_HOURS:
        while sum_w > MAX_WORKING_HOURS:
            working_days=[]
            for d in range(len(DAYS)):
                if working[d]: working_days.append(d)
            r=random.choice(working_days)
            sh=pattern.shifts[r]
            pattern.assign_shift(r, neutral_shift())
            sum_w-=sh.workTime
            working[r]=0
    rest_days=[]
    for d in range(len(DAYS)):
        if not working[d]: rest_days.append(d)
    if len(rest_days)==0:
        r=random.choice(range(len(DAYS)))
        pattern.assign_shift(r, neutral_shift())

# solve the problem day per day to use daily shifts instead of weekly patterns
# and then make the solution feasible for the weekly problem
def solve_per_day_and_repair(timeout, obj_tolerance, abs_obj_tolerance):
    # add neutral shift

    # data.shifts.append(neutral_shift())

    assert data.shifts[-1].idShift == -1  # day off shift must be the last shift of the sequence
    neutral_shift = data.shifts[-1]
    weekly_param=model_parameters_weekly(data, [i for i in range(len(DAYS))], NB_TIME_PERIODS, external_cost_intra, external_cost_agglo, pattern_fixed_cost=pattern_fixed_cost)
    weekly_param.instanciate_parameters()
    cdata=copy.deepcopy(data)
    print("Number of couriers (data)", len(cdata.employees))
    shifts_used = [[-1 for c in cdata.employees] for d in DAYS]
    objective_values = [0.0 for d in DAYS]
    delta_cost = [0] * len(weekly_param.C)  # store the min cost increase to make the pattern feasible
    v_affectation={}

    shift_fixed_cost = pattern_fixed_cost/len(DAYS)

    total_fix_cost=0
    total_var_cost=0
    total_ext_cost=0

    repair_patterns=[]
    y_sol={}
    for d in range(len(DAYS)):
        print("Day ", d)
        # export the data from day d to the day 0 of the cdata object
        for id in range(len(data.od_pairs)):
            odp=cdata.od_pairs[id]
            odp.nb_scenarios_d = [[0] * NB_TIME_PERIODS]
            odp.demand_od_pair_d = [[]]
            odp.probability_d = [[]]
            odp.nb_scenarios_d[0] = data.od_pairs[id].nb_scenarios_d[d]
            odp.demand_od_pair_d[0] = data.od_pairs[id].demand_od_pair_d[d]
            odp.probability_d[0] = data.od_pairs[id].probability_d[d]

        # create one pattern for each shift, the patterns contain only shift for day 0
        cdata.patterns=[]
        for id_shift in range(len(cdata.shifts)):
            shift=cdata.shifts[id_shift]
            cdata.patterns.append(Pattern(shift.idShift, [shift]))

        # create the parameters used by the model to solve the weekly problem on day 0
        param = model_parameters_weekly(cdata, [0], NB_TIME_PERIODS, external_cost_intra, external_cost_agglo, pattern_fixed_cost=shift_fixed_cost)
        param.instanciate_parameters()

        prob, fix_cost, var_cost, ext_cost=run_weekly(param, log_output=False, obj_tolerance=obj_tolerance,
                                                      abs_obj_tolerance=abs_obj_tolerance, timeout=timeout/len(DAYS), decompose_costs=True)

        total_fix_cost+=fix_cost
        total_var_cost+=var_cost
        total_ext_cost+=ext_cost

        try:
            objective_values[d] = prob.objective_value

            for var in prob.iter_variables():
                if var.type=="X" and var.solution_value >= 0.99:
                    c, t = var.get_key()
                    shifts_used[d][c] = t
                if var.type=="V":
                    c, day, i, w, p=var.get_key()
                    v_affectation[(c, d, i, w, p)]=var.solution_value
                if var.type=="Y":
                    c, day, i, p = var.get_key()
                    y_sol[(c, d, i, p)]=var.solution_value
        except:
            print("Something went wrong with the model")
        del param
        del prob

    #release memory
    del cdata


    try:
        for c in weekly_param.C:
            shift_price=[2^16 for d in weekly_param.D] # store the cost of removing the shift of day d (0 if not shift)
            total_hours=0
            day_off=False
            nb_shifts_used=0
            pattern=[]
            for d in weekly_param.D:
                if shifts_used[d][c] >=0:
                    shift=data.shifts[shifts_used[d][c]]
                    pattern.append(shift)
                    f_cost=data.employees[c].fixed_cost
                    w_hours=shift.workTime
                    total_hours+=w_hours
                    if w_hours==0: day_off=True
                    else: nb_shifts_used+=1
                    affectation_cost=-f_cost*w_hours+shift_fixed_cost
                    for i in range(shift.startPeriod, shift.endPeriod):
                        for w in weekly_param.Omega_i_d[d][i]:
                            proba=weekly_param.proba[(d, i, w)]
                            affectation_cost+=proba*sum(
                                +v_affectation[(c, d, i, w, p)]*(weekly_param.c[(i, p)]
                                -v_affectation[(c, d, i, w, p)]*weekly_param.l[(c, i, p)])
                            for p in weekly_param.P)
                    shift_price[d]=affectation_cost
                else:
                    pattern.append(neutral_shift)
                print(shift_price)
            if not day_off:
                min_shift=int(numpy.argmin(shift_price))
                pattern[min_shift]=neutral_shift
                delta_cost[c]=shift_price[min_shift]
            delta_cost[c]+=-shift_fixed_cost*nb_shifts_used+pattern_fixed_cost
            pat=Pattern(len(data.patterns), pattern)
            for sh in pat.shifts:
                print("Shift", sh.idShift, sh.workTime)
            make_feasible(data, pat)
            data.patterns.append(pat)
            repair_patterns.append(pat.id_pattern)
        print(delta_cost, " / ", sum(delta_cost))
    except:
        print("Something went wrong while computing the delta cost")
    return objective_values, delta_cost, total_fix_cost, total_var_cost, total_ext_cost

#This is to run several instances
def run_bac(args, grammar_graph, timeout, epsilon_tolerance, obj_tolerance, abs_obj_tolerance, current_bench_dir):

    per_day_time=-1
    per_day_obj=-1
    fix_cost=-1
    var_cost=-1
    ext_cost=-1
    repair_obj=-1

    if args.day_heu:
        per_day_time=time.time()
        obj_per_day, delta_per_courier, fix_cost, var_cost, ext_cost = solve_per_day_and_repair(timeout, obj_tolerance, abs_obj_tolerance)
        per_day_time=time.time()-per_day_time
        per_day_obj=sum(obj_per_day)
        repair_obj=per_day_obj+sum(delta_per_courier)
        print("Per day objs:", per_day_obj, fix_cost, var_cost, ext_cost, sum([fix_cost, var_cost, ext_cost]), repair_obj)

    #print("Running Bac")
    parameters_time=time.time()
    param = model_parameters_weekly(data, list(range(len(DAYS))), NB_TIME_PERIODS, external_cost_intra, external_cost_agglo, pattern_fixed_cost=pattern_fixed_cost)
    param.instanciate_parameters()
    parameters_time=time.time()-parameters_time

    #print("Number of patterns: {}".format(len(data.patterns)))

    #print("Starting Bac resolution with {} patterns".format(len(data.patterns)))
    bac_problem = solve_bac(param, grammar_graph, args.gram, timeout=timeout, log_output=True, obj_tolerance=obj_tolerance,
                            abs_obj_tolerance=abs_obj_tolerance, epsilon_tolerance=epsilon_tolerance, decomp_costs=True)

    del param

    return bac_problem, per_day_time, per_day_obj, fix_cost, var_cost, ext_cost, repair_obj


def run_mean_value(timeout, epsilon_tolerance, obj_tolerance, abs_obj_tolerance, grammar):
    print("Running mean value")
    param_mean = model_parameters_weekly(data, list(range(len(DAYS))), NB_TIME_PERIODS, external_cost_intra, external_cost_agglo, pattern_fixed_cost=pattern_fixed_cost)

    param_mean.instanciate_mean_parameters()

    mean_problem=run_weekly(param_mean, log_output=False, timeout=timeout,
                            obj_tolerance=obj_tolerance, abs_obj_tolerance=abs_obj_tolerance, decompose_costs=False)

    mean_bac_time=mean_problem.timer.TOTAL_TIME
    mean_bac_gap=mean_problem.solve_details.gap

    first_stage_x = first_stage_y = {}
    for var in mean_problem.iter_variables():
        if var.type == "X":
            c, t = var.get_key()
            first_stage_x[(c, t)] = var.solution_value
        if var.type == "Y":
            c, d, i, p = var.get_key()
            first_stage_y[(c, d, i, p)] = var.solution_value
    del param_mean
    del mean_problem
    param = model_parameters_weekly(data, list(range(len(DAYS))), NB_TIME_PERIODS, external_cost_intra, external_cost_agglo, pattern_fixed_cost=pattern_fixed_cost)
    param.instanciate_parameters()
    mean_fs_problem = solve_bac(param, grammar_graph, grammar, timeout=timeout, fix_first_stage=True, first_stage_x=first_stage_x,
                                first_stage_y=first_stage_y, decomp_costs=True)
    mean_rec_obj=mean_fs_problem.objective_value
    mean_rec_fix=mean_fs_problem.fix
    mean_rec_var=mean_fs_problem.v_cost
    mean_rec_ext=mean_fs_problem.e_cost

    del param
    del mean_fs_problem

    return mean_bac_time, mean_bac_gap, mean_rec_obj, mean_rec_fix, mean_rec_var, mean_rec_ext

def execute_arguments(args):
    timeout=args.time
    epsilon_tolerance=args.eps
    obj_tolerance=1e-04*args.gap
    abs_obj_tolerance=1e-06*args.gap

    grammar_graph = []

    data.generate_couriers(NB_COURIERS, full_avail=FULL_AVAILABILITY)

    pattern_generation_time=time.time()

    if args.price_patterns or args.price_patterns_flex:

        # GENERATE PATTERNS WITH PRICING PROBLEM
        data.shifts.append(neutral_shift())
        mean_ext= (external_cost_intra + external_cost_agglo) / 2
        run_pattern_generation(data, len(DAYS), NB_TIME_PERIODS, fixed_cost, mean_ext)
        data.filter_incompatible_patterns()

    if args.full_patterns:

        if args.gram:
            #grammar_string = get_grammar_string_2(data)

            #grammar_string = get_grammar_string_3(data)

            grammar = readGrammar(data)

            dag = buildCYKGraph(len(DAYS), grammar)

            # Build the grammar graph
            for c in range(len(data.employees)):
                #print(c)
                # grammar_graph.append(build_dag(grammar_string, length_bounds, len(DAYS)))
                grammar_graph.append(buildCYKGraph(len(DAYS), grammar))
        else:
            data.generate_patterns_from_shifts()

    pattern_generation_time=time.time()-pattern_generation_time

    data_list=[]

    if not args.no_exp:
        distribution = ""
        if args.n: distribution = "normal"
        if args.g: distribution = "gamma"
        if args.u: distribution = "uniform"
        variability = ""
        if args.me: variability = "medium_var"
        if args.hi: variability = "high_var"
        current_bench_dir=create_bench_dir(bench_dir, args.instance_size, args.instance_number, distribution, variability)
        generation_type=""
        if args.full_patterns: generation_type="Exhaustive"
        if args.price_patterns or args.price_patterns_flex: generation_type="Pricing"

        data_list+=["COLLECTION", args.instance_size, distribution, variability, args.instance_number, len(DAYS),
                    NB_TIME_PERIODS, len(data.od_pairs), len(data.employees), len(data.shifts), min_shift_length,
                    max_shift_length, SHIFT_GAP, len(data.patterns), MIN_WORKING_HOURS, MAX_WORKING_HOURS,
                    generation_type, pattern_generation_time]

    else:
        current_bench_dir="trash/"

    if args.rec:
        #print("ENTER RECOURSE PROBLEM!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        # per day resolution + heuristic repair + bac recourse problem
        model, per_day_time, per_day_obj, per_day_fix_cost, per_day_var_cost, per_day_ext_cost, repair_obj = run_bac(args, grammar_graph,
            timeout, epsilon_tolerance, obj_tolerance, abs_obj_tolerance, current_bench_dir)
        ti=model.timer
        mov=model.objective_value
        mf=model.fix
        mv=model.v_cost
        me=model.e_cost
        mnwc=model.nb_working_couriers
        data_list+=[per_day_time, per_day_obj, per_day_fix_cost, per_day_var_cost, per_day_ext_cost, repair_obj,
                epsilon_tolerance, obj_tolerance, ti.TOTAL_TIME, ti.TIME_DEFINING_MASTER,
                    ti.TOTAL_TIME_REACHING_TOLERANCE, ti.TOTAL_TIME_BB, model.solve_details.gap,
                    mov, mf, mv, me, mnwc]
        del model
        # mean value problem solved + recourse problem with mean value first stage decisions
        mean_bac_time, mean_bac_gap, mean_rec_obj, mean_rec_fix, mean_rec_var, mean_rec_ext = (-1, -1, -1, -1, -1, -1)
        if args.ev:
            mean_bac_time, mean_bac_gap, mean_rec_obj, mean_rec_fix, mean_rec_var, mean_rec_ext=run_mean_value(
                timeout, epsilon_tolerance, obj_tolerance, abs_obj_tolerance, args.gram)
        data_list+=[mean_bac_time, mean_bac_gap, mean_rec_obj, mean_rec_fix, mean_rec_var, mean_rec_ext]
    else:
        # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        run_bac(args, grammar_graph, timeout, epsilon_tolerance, obj_tolerance, abs_obj_tolerance, current_bench_dir)

    if not args.no_exp:
        data_file=current_bench_dir+"data_string.txt"


        #print("exporting!!!!!!!!!", data_file)
        export_data_string(data_file, data_list)

    #print(data_list)

if __name__ == "__main__":

    #print("Parsing data")

    vehicle_file = "./data/VehicleParameters3.csv"
    postal_codes_file = "./data/PostalCodesWithCentroidsSQL.csv"

    args = argument_parser()

    instance_collection = "./instances/"
    isize=args.instance_size
    inb=args.instance_number

    distribution=""
    if args.n: distribution="normal"
    if args.g: distribution="gamma"
    if args.u: distribution = "uniform"
    variability=""
    if args.me: variability="medium_var"
    if args.hi: variability="high_var"

    instance_dir="{}{}/{}/instance_{}/".format(instances_dir, isize, distribution, inb)


    #TODO: remove the comment, this is just to test the two models
    #TODO: check the computation of the mean value problem

    #demand_files = [instance_dir + "stochastic_{}.csv".format(i) for i in range(len(DAYS))]
    demand_files = [instance_dir + "stochastic_{}.csv".format(i) for i in range(len(DAYS))]

    #create the class ProbData
    data = ProbData()

    #read the files with the information
    data_vehicle = pd.read_csv(vehicle_file)
    data_postal_codes = pd.read_csv(postal_codes_file)
    data_demands = []
    for d in range(len(DAYS)):
        data_demands.append(pd.read_csv(demand_files[d]))

    #print(data_vehicle)

    for index, row in data_vehicle.iterrows():
        data.add_vehicle(id_vehicle = index, idVS = row['vehicle_id'], name = row['vehicle_id'], maxWeight = row['maxWeightCapacity'],
                         maxHeight = 0, avSpeed = row['maxSpeed'], threshold = 0, vehicleWearCost = row['vehicleWearCost'],
                         pickupCost = row['pickupCost'], deliveryCost = row['deliveryCost'])

    for index, row in data_postal_codes.iterrows():
        zone_pc = 0
        postal_code = str(row['postal_code'])

        #to change
        if(postal_code[0] == '7'):
            zone_pc = 0
        else:
            zone_pc = 1

        data.add_postal_code(postal_code = row['postal_code'], pc_id = row['pc_id'], latitude=row['latitude'],
                             longitude=row['longitude'], zone = zone_pc)

    d=0
    parse_demand=time.time()
    tot_dem = 0
    for data_demand in data_demands:
        table = pd.pivot_table(data_demand, values='nbr_packages', index=['pc_pickup', 'pc_delivery'], columns = ['hour', 'id_scenario'],
                               aggfunc = np.sum)


        flattened_demand = pd.DataFrame(table.to_records()).fillna(0)

        table_2 = pd.pivot_table(data_demand, values='nbS', index=['pc_pickup', 'pc_delivery'],
                               columns=['hour'],
                               aggfunc=np.mean)

        flattened_nb_scenarios = pd.DataFrame(table_2.to_records()).fillna(0)

        MIN_HOUR = min(data_demand['hour'])
        MAX_HOUR = max(data_demand['hour'])
        NB_TIME_PERIODS = MAX_HOUR - MIN_HOUR + 1

        for index, row in flattened_demand.iterrows():

            latitude1 = longitude1 = latitude2 = longitude2 = zone1 = zone2 = -1
            id1 = id2 = None
            scenarios = [0] * NB_TIME_PERIODS #the number of scenarios per time period
            demand = [0] * NB_TIME_PERIODS #the demand per time period, per scenario (the number of scenarios can change at each time period
            prob = [0] * NB_TIME_PERIODS

        # create a new object from the class od_pair

            for i in data.postal_codes:
                if (i.postal_code == int(row['pc_pickup'])):
                    id1 = i.pc_id
                    zone1 = i.zone
                if (i.postal_code == int(row['pc_delivery'])):
                    id2 = i.pc_id
                    zone2 = i.zone
                if(id1 != None and id2!= None):
                    break

            # scenarios
            for j in range(0, NB_TIME_PERIODS):
                scenarios[j] = int(flattened_nb_scenarios.iloc[index,j+2])
                demand[j] = [0] * scenarios[j]
                prob[j] = [0] * scenarios[j]

            data.add_od_pair_day(d, pc_pickup = row['pc_pickup'], pc_delivery = row['pc_delivery'], id_area1 = id1, id_area2 = id2,
                             origin = 0, nb_scenarios = scenarios,
                             demand_od_pair = demand, distance = 0, probability = prob, zone = max(id1, id2))


        iter_demand = time.time()
        for index, row in data_demand.iterrows():
            i = data.od_pairs_generated[(row['pc_pickup'], row['pc_delivery'])]
            i.demand_od_pair_d[d][int(row['hour'])- MIN_HOUR][int(row['id_scenario'])-1] += row['nbr_packages']
            i.probability_d[d][int(row['hour'])- MIN_HOUR][int(row['id_scenario'])-1] = row['probability']
            if (d==0):
                tot_dem+=row['nbr_packages']


        d+=1
        #print(time.time() - iter_demand, "seconds iterating demand")

    #print(time.time() - parse_demand, "seconds parsing demand"," total demand", tot_dem)

    # exit(1)

    compute_distance=time.time()
    #compute the distance for each od pair
    for i in data.od_pairs:
        lat1 = lon1 = lat2 = lon2 = None
        for j in data.postal_codes:
            if (j.postal_code == i.pc_pickup):
                lat1 = j.latitude
                lon1 = j.longitude
            if (j.postal_code == i.pc_delivery):
                lat2 = j.latitude
                lon2 = j.longitude
            if(lat1 != None and lon2!= None):
                break
        i.compute_distance(lat1, lon1, lat2, lon2)

    #print(time.time() - compute_distance, "seconds computing distance")

    employee_file = instance_dir + "employees.csv"

    data_employee = pd.read_csv(employee_file)

    for index, row in data_employee.iterrows():
        NB_COURIERS = row["nb_drivers"]


    #print(NB_COURIERS, NB_TIME_PERIODS)

    # WEEKLY WORKING RULES TO CONSIDER
    #todo: Check this because the values are different when checking if a tour is feasible or not
    if args.full_patterns or args.price_patterns:
        min_shift_length = 6
        max_shift_length = 8
        SHIFT_GAP = 2
        AVAILABILITIES_DURATION = [NB_TIME_PERIODS]
        FULL_AVAILABILITY=True
        MIN_WORKING_HOURS = 32
        MAX_WORKING_HOURS = 40
    # if args.price_patterns_flex:
    #     min_shift_lengt = 8
    #     max_shift_lengt = 8
    #     SHIFT_GAP = 2
    #     AVAILABILITIES_DURATION = [NB_TIME_PERIODS]
    #     FULL_AVAILABILITY = True
    #     MIN_WORKING_HOURS = 30
    #     MAX_WORKING_HOURS = 40


    #Build the shifts
    id_shift = 0
    delay = 3
    for l in range(min_shift_length, max_shift_length + 1, SHIFT_GAP):
        for t in range(0, NB_TIME_PERIODS-(l-1), SHIFT_GAP):

            cost = 0
            shift = [0] * NB_TIME_PERIODS

            for i in range(t, t+l):
                shift[i] = 1
                cost += fixed_cost
            data.add_shift(idShift= id_shift, idShiftString="None", cost = cost, workTime = l, maxHeight = 0,
                           shift = shift, startPeriod = t, endPeriod = t+l-1, timesZero = 0, timesOne = 0,
                           fixed= 0, min_duration = 1, max_duration = 6, tws = 0, twl = 7)
            #print(shift, " c: ", cost)

            id_shift += 1

    if args.day_heu:
        data.add_shift(idShift=-1, idShiftString="Day off", cost=0, workTime=0, maxHeight=0,
                   shift=[0] * NB_TIME_PERIODS, startPeriod=0, endPeriod=0, timesZero=0, timesOne=0,
                   fixed=0, min_duration=0, max_duration=0, tws=0, twl=0)





    #print("Data parsed")

    # print("Couriers availability:", data.employees)
    #
    # for c in data.employees:
    #     print("{} -> {} ({})".format(c.start_availability, c.end_availability, c.end_availability-c.start_availability+1))

    #print("Number of od-pairs:", len(data.od_pairs), " number of shifts: ", len(data.shifts))

    #exit(1)

    execute_arguments(args)

