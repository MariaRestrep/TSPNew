# contains the classes and functions to parse the data plus some functions to execute the arguments parsed from arguments.py
import copy
import numpy as np
import pandas as pd
import math
import random

from exporter import *
from model_weekly import *
from branch_and_cut import *
from arguments import *

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
collection = "collection"

instances_dir="./instances/{}/".format(collection)
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

    # def generate_couriers(self, nb_couriers, full_avail=False):
    #     data.employees=[]
    #     random.seed(SEED)
    #     for i in range(int(nb_couriers)):
    #         start_availability = MIN_HOUR
    #         end_availability = MAX_HOUR
    #         availability_duration = random.choice(AVAILABILITIES_DURATION)
    #         if full_avail: availability_duration = NB_TIME_PERIODS
    #         if end_availability-availability_duration > 0:
    #             start_availability = random.choice(range(start_availability, 2+end_availability-availability_duration, 2))
    #         else:
    #             start_availability = 0
    #         end_availability = start_availability + availability_duration - 1
    #         veh = random.choice(data.vehicles)
    #         var_cost = veh.pickupCost + veh.deliveryCost
    #         data.add_employee(id_employee=i, id_vehicle=veh.id_vehicle,
    #                           start_availability=start_availability,
    #                           end_availability=end_availability, fixed_cost=fixed_cost+veh.vehicleWearCost,
    #                           var_cost_intra=var_cost, var_cost_aglo=var_cost * agglo_var_cost_coef,
    #                           capacity=veh.maxWeight)

#This is to run several instances
def run_bac(args, timeout, epsilon_tolerance, obj_tolerance, abs_obj_tolerance, current_bench_dir):

    per_day_time=-1
    per_day_obj=-1
    fix_cost=-1
    var_cost=-1
    ext_cost=-1
    repair_obj=-1

    #print("Running Bac")
    parameters_time=time.time()
    param = model_parameters_weekly(data, list(range(len(DAYS))), NB_TIME_PERIODS, external_cost_intra, external_cost_agglo, pattern_fixed_cost=pattern_fixed_cost)
    param.instanciate_parameters()
    parameters_time=time.time()-parameters_time

    #print("Number of patterns: {}".format(len(data.patterns)))

    #print("Starting Bac resolution with {} patterns".format(len(data.patterns)))
    bac_problem = solve_bac(param, timeout=timeout, log_output=True, obj_tolerance=obj_tolerance,
                            abs_obj_tolerance=abs_obj_tolerance, epsilon_tolerance=epsilon_tolerance, decomp_costs=True)

    del param

    return bac_problem, per_day_time, per_day_obj, fix_cost, var_cost, ext_cost, repair_obj


def execute_arguments(args):

    timeout=args.time
    epsilon_tolerance=args.eps
    obj_tolerance=1e-04*args.gap
    abs_obj_tolerance=1e-06*args.gap

    #data.generate_couriers(NB_COURIERS, full_avail=FULL_AVAILABILITY)

    pattern_generation_time=time.time()

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
        model, per_day_time, per_day_obj, per_day_fix_cost, per_day_var_cost, per_day_ext_cost, repair_obj = run_bac(args,
            timeout, epsilon_tolerance, obj_tolerance, abs_obj_tolerance, current_bench_dir)
        ti=model.timer
        mov=model.objective_value
        mf=model.fix
        mv=model.v_cost
        me=model.e_cost
        mnwc=model.nb_working_couriers
        max_tour_l = model.max_tour_l
        min_tour_l = model.min_tour_l
        average_tour_l = model.average_tour_l
        
        data_list+=[per_day_time, per_day_obj, per_day_fix_cost, per_day_var_cost, per_day_ext_cost, repair_obj,
                epsilon_tolerance, obj_tolerance, ti.TOTAL_TIME, ti.TIME_DEFINING_MASTER,
                    ti.TOTAL_TIME_REACHING_TOLERANCE, ti.TOTAL_TIME_BB, model.solve_details.gap,
                    mov, mf, mv, me, mnwc, max_tour_l, min_tour_l, average_tour_l]
        del model
        # mean value problem solved + recourse problem with mean value first stage decisions
        mean_bac_time, mean_bac_gap, mean_rec_obj, mean_rec_fix, mean_rec_var, mean_rec_ext = (-1, -1, -1, -1, -1, -1)

        data_list+=[mean_bac_time, mean_bac_gap, mean_rec_obj, mean_rec_fix, mean_rec_var, mean_rec_ext]


    if not args.no_exp:
        data_file = current_bench_dir + "results_" + str(args.instance_number) + ".txt"
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
        NB_COURIERS = row["nb_drivers"] + 2

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


    #data.add_shift(idShift=-1, idShiftString="Day off", cost=0, workTime=0, maxHeight=0, shift=[0] * NB_TIME_PERIODS,
                   #startPeriod=0, endPeriod=0, timesZero=0, timesOne=0, fixed=0, min_duration=0, max_duration=0, tws=0, twl=0)

    #print("Number of od-pairs:", len(data.od_pairs), " number of shifts: ", len(data.shifts))

    #exit(1)

    execute_arguments(args)

