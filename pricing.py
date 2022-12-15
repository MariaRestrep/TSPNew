from docplex.mp.model import Model
import math
import time
import numpy
import random

timeout_pricing=600

# requirements: staff requirement per day and time period, computed with function compute_required_couriers
def define_pattern_pricing(data, requirement, nb_days, nb_periods, fixed_cost, external_cost, rho, patterns):

    x_idx = [(c, t) for c in range(len(data.employees)) for t in range(len(patterns))]
    y_idx = [(d, i) for d in range(nb_days) for i in range(nb_periods)]

    master=Model()

    master.x=master.continuous_var_dict(x_idx, name="X", lb=0.0, ub=1.0)
    master.y_pos=master.continuous_var_dict(y_idx, name="Y+", lb=0.0)
    master.y_neg=master.continuous_var_dict(y_idx, name="Y-", lb=0.0)

    x=master.x
    y_pos=master.y_pos
    y_neg=master.y_neg

    for d in range(nb_days):
        for i in range(nb_periods):
            master.add_constraint(
                master.sum(
                    master.sum(
                        rho[(c, d, i, data.patterns[t].id_pattern)]*x[(c, t)]
                    for t in range(len(patterns)))
                for c in range(len(data.employees)))
                - y_pos[(d, i)] + y_neg[(d, i)]==requirement[(d, i)], ctname="cst1_{}_{}".format(d, i))

    for c in range(len(data.employees)):
        master.add_constraint(
            master.sum(
                x[(c, t)]
            for t in range(len(patterns))) <= 1, ctname="cst2_{}".format(c))

    f_cost=master.sum(
        master.sum(
            (data.employees[c].fixed_cost+data.employees[c].var_cost_intra)*x[(c, t)]*data.patterns[t].weekly_hours
        for t in range(len(patterns)))
    for c in range(len(data.employees)))
    overstaffing=master.sum(
        master.sum(
            y_pos[(d, i)]*fixed_cost
        for i in range(nb_periods))
    for d in range(nb_days))
    understaffing=master.sum(
        master.sum(
            y_neg[(d, i)]*external_cost
        for i in range(nb_periods))
    for d in range(nb_days))
    master.minimize(f_cost+overstaffing+understaffing)

    return master

# compute the number of required couriers per time period (r_di in the model)
def compute_required_couriers(data, nb_days, nb_periods):
    required_couriers={}
    mean_capacity=sum(v.maxWeight for v in data.vehicles)/len(data.vehicles)
    for d in range(nb_days):
        for i in range(nb_periods):
            mean_demand=0
            for odp in data.od_pairs:
                try:
                    nb_scen=odp.nb_scenarios_d[d][i]
                except:
                    nb_scen=0
                for scen in range(nb_scen):
                    proba=odp.probability_d[d][i][scen]
                    dem=odp.demand_od_pair_d[d][i][scen]
                    mean_demand+=math.ceil(dem/mean_capacity)*proba
            mean_demand=math.ceil(mean_demand)
            required_couriers[(d, i)]= mean_demand
    return required_couriers

def generate_random_shifts(data, nb_days, neutral_shift):
    shifts=[]
    shifts_list=list(range(len(data.shifts)))
    for d in range(nb_days):
        shifts.append(data.shifts[random.choice(shifts_list)])
    rest_day=False
    for sh in shifts:
        if sh.idShift==-1:
            rest_day=True
    if not rest_day:
        shifts[random.choice(range(nb_days))]=neutral_shift
    return shifts

# column generation procedure
# parameters: data (parser in main), nb_days, nb_periods, fixed_cost (cost per hour), external_cost (mean external cost)
# modify the list of patterns data.patterns
def run_pattern_generation(data, nb_days, nb_periods, fixed_cost, external_cost):

    last_obj=0
    # number of steps with no improvement
    nb_plateau=0
    max_plateau=100
    plateau_gap=0.1

    assert data.shifts[-1].idShift == -1 # day off shift must be the last shift of the sequence
    neutral_shift=data.shifts[-1]

    # initial patterns
    data.patterns=[]
    rest_day=0
    for ids in range(len(data.shifts)):
        sh=data.shifts[ids]
        shifts=[sh]*nb_days
        shifts[rest_day]=neutral_shift
        data.add_pattern(shifts)
        rest_day=(rest_day+1)%nb_days
    print(len(data.patterns), "initial patterns")

    # compute rho: 1 if di is a working hour in pattern t for courier c, 0 otherwise
    rho={}
    for c in range(len(data.employees)):
        for d in range(nb_days):
            for i in range(nb_periods):
                for t in range(len(data.patterns)):
                    employee=data.employees[c]
                    pattern=data.patterns[t]
                    rho[(c, d, i, pattern.id_pattern)]=(i>=employee.start_availability) and (i<=employee.end_availability) and \
                                      (i>=pattern.shifts[d].startPeriod) and (i<=pattern.shifts[d].endPeriod)

    # compute rho_shift: 1 if di is a working hour in shift s for courier c, 0 otherwise
    rho_shift={}
    for c in range(len(data.employees)):
        for d in range(nb_days):
            for i in range(nb_periods):
                for s in range(len(data.shifts)):
                    employee=data.employees[c]
                    shift=data.shifts[s]
                    rho_shift[(c, d, i, s)] = (i >= employee.start_availability) and (i <= employee.end_availability) and \
                        (i >= shift.startPeriod) and (i <= shift.endPeriod)

    patterns=list(range(len(data.patterns)))
    requirement=compute_required_couriers(data, nb_days, nb_periods)
    master=define_pattern_pricing(data, requirement, nb_days, nb_periods, fixed_cost, external_cost, rho, patterns)

    last_nb_patterns=len(data.patterns)

    # start the generation procedure
    stop=False
    previously_generated=[]
    start_pricing=time.time()
    while not stop:
        new_patterns = False
        master.solve()
        obj=master.objective_value
        beta={}
        sigma={}
        # get the dual values from the model
        for cst in master.iter_constraints():
            splitted=cst.name.split("_")
            if cst.name[3]=="1":
                d, i =int(splitted[1]), int(splitted[2])
                beta[(d, i)] = cst.dual_value
            if cst.name[3]=="2":
                c = int(splitted[1])
                sigma[c] = cst.dual_value

        nb_patterns=0
        total_prices=[]
        # compute a pattern for each courier
        for c in range(len(data.employees)):
            pat=[]
            prices=[]
            # choose the shift with the lowest reduce cost each day
            for d in range(nb_days):
                sh_id=-1
                sh_price=2^16
                # compute the reduced cost of each shift on day d
                for s in range(len(data.shifts)):
                    price=sum(((data.employees[c].fixed_cost + data.employees[c].var_cost_intra) - beta[(d, i)]) * rho_shift[
                        (c, d, i, s)] for i in range(nb_periods))-sigma[c]/nb_days
                    if price<sh_price:
                        sh_id=s
                        sh_price=price
                pat.append(sh_id)
                prices.append(sh_price)
            higher_price=numpy.argmax(prices)
            #higher_price = numpy.argmin(prices)
            shifts=[]
            total_price=sum(prices)
            # build the sequence of shift
            for ids in pat:
                if ids>=0:
                    sh=data.shifts[ids]
                    shifts.append(sh)
                else:
                    shifts.append(neutral_shift)
            rest_day=False
            # add a rest day if there is no rest day in the pattern
            for sh in shifts: rest_day=(sh.idShift==-1)or rest_day
            # set the rest day on the day with higer reducd cost
            if not rest_day:
                shifts[higher_price]=neutral_shift
                total_price-=prices[higher_price]
            # check that the pattern does not already exist
            if shifts not in previously_generated:
                before=len(data.patterns)
                # add the pattern if it is feasible
                data.add_pattern(shifts, check_feasibility=True)
                # if the pattern has been added
                if len(data.patterns)>before:
                    if not new_patterns:
                        previously_generated=[]
                        new_patterns=True
                    previously_generated.append(shifts)
            nb_patterns+=1
            total_prices.append(total_price)
        if abs(last_obj-obj) < plateau_gap:
            nb_plateau+=1
            if not new_patterns:
                # add a randomly generated pattern if none has been added to avoid cycling
                shifts=generate_random_shifts(data, nb_days, neutral_shift)
                data.add_pattern(shifts, check_feasibility=True)
        else:
            nb_plateau=0
        last_obj=obj
        if nb_plateau==max_plateau or time.time() - start_pricing>=timeout_pricing:
            stop=True
            print("Stop with {} patterns".format(len(patterns)))

        if nb_patterns==0:
            pass
        else:
            deb_define=time.time()

            # redefine rho for added patterns
            for c in range(len(data.employees)):
                for d in range(nb_days):
                    for i in range(nb_periods):
                        for t in range(last_nb_patterns, len(data.patterns)):
                            employee = data.employees[c]
                            pattern = data.patterns[t]
                            rho[(c, d, i, pattern.id_pattern)] = (i >= employee.start_availability) and (
                                        i <= employee.end_availability) and \
                                                (i >= pattern.shifts[d].startPeriod) and (
                                                            i <= pattern.shifts[d].endPeriod)
            last_nb_patterns=len(data.patterns)
            patterns = list(range(len(data.patterns))) #consider all patterns

            # here it would be more efficient to update the current model than creating a new one
            master=define_pattern_pricing(data, requirement, nb_days, nb_periods, fixed_cost, external_cost, rho, patterns)


