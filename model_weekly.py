import time
import timer
from docplex.mp.model import Model


# class model parameters weekly define the parameters of the models (complete and B&C)
# parameters: data a prob_data object containing the parsed data
#			  nb_days, nb_timesperiods, external_cost_intra, external_cost_agglo, pattern_fixed_cost
class model_parameters_weekly:
    def __init__(self, data, nb_days, nb_time_periods, external_cost_intra, external_cost_agglo,
                 pattern_fixed_cost=0):
        self.data = data
        self.nb_days = nb_days
        self.nb_time_periods = nb_time_periods
        self.external_cost_intra = external_cost_intra
        self.external_cost_agglo = external_cost_agglo
        self.nb_scenarios = []
        self.nb_scenarios_d = []
        self.D = []
        self.I = []
        self.P = []
        self.C = []
        self.Tc = [[]]
        self.Omega_i_d = [[]]
        self.fixed_costs = {}
        self.delta_d = {}
        self.alpha = {}
        self.sigma = {}
        self.l = {}
        self.c = {}
        self.beta = {}
        self.d = {}
        self.mu = {}
        self.proba = {}
        self.pattern_fixed_cost=pattern_fixed_cost

    def set_of_days(self):
        self.D = [i for i in range(len(self.nb_days))]

    def set_of_periods(self):
        self.I = [i for i in range(self.nb_time_periods)]

    def set_of_pairs(self):
        self.P = [i for i in range(len(self.data.od_pairs))]

    def set_of_couriers(self):
        self.C = [i for i in range(len(self.data.employees))]

    def set_of_patterns(self):
        #print("Set of patterns")
        self.Tc = []
        for c in self.C:
            courier = self.data.employees[c]
            patterns = []
            for t in range(len(self.data.patterns)):
                pattern = self.data.patterns[t]
                compatible = True
                for d in self.D:
                    shift = pattern.shifts[d]
                    if shift.idShift != -1:
                        compatible = compatible and (min(shift.endPeriod, courier.end_availability)-
                                                 max(shift.startPeriod, courier.start_availability)+1 == shift.workTime)
                if compatible:
                    patterns.append(t)
            self.Tc.append(patterns)

    def set_of_scenarios(self):
        # store the maximum number of scenarios over all od pairs for each day and each period
        nb_scenarios_d = [[0 for i in self.I] for d in self.D]
        for p in self.data.od_pairs:
            for d in self.D:
                for i in self.I:
                    try:
                        nb_scen=p.nb_scenarios_d[d][i]
                    except:
                        nb_scen=0
                    nb_scenarios_d[d][i] = max(nb_scenarios_d[d][i], nb_scen)
        self.Omega_i_d = [[[j for j in range(nb_scenarios_d[d][i])] for i in self.I] for d in self.D]

    def set_of_mean_scenarios(self):
        self.Omega_i_d = [[[0] for i in self.I] for d in self.D]

    def set_of_fixed_costs(self):
        for c in self.C:
            courier = self.data.employees[c]
            for t in self.Tc[c]:
                pattern = self.data.patterns[t]
                # self.fixed_costs[(c, t)] = pattern.weekly_hours*courier.fixed_cost + self.pattern_fixed_cost
                self.fixed_costs[(c, t)] = pattern.weekly_hours * 25
        #     print("courier ", c, " fixed cost ", courier.fixed_cost)
        #     print("pattern fixed cost ", self.pattern_fixed_cost)
        #
        # exit(1)



    def set_of_delta(self):
        #print("Set of delta")
        for c in self.C:
            for d in self.D:
                for i in self.I:
                    for t in self.Tc[c]:
                        pattern = self.data.patterns[t]
                        shift = pattern.shifts[d]
                        self.delta_d[(c, d, i, t)] = shift.shift[i]

    def set_of_alpha(self):
        #print("Set of alpha")
        for c in self.C:
            for i in self.I:
                self.alpha[(c, i)] = 1

    def set_of_sigma(self):
        for c in self.C:
            for i in self.I:
                for p in self.P:
                    self.sigma[(c, i, p)] = self.data.od_pairs[p].distance

    def set_of_variable_costs(self):
        for c in self.C:
            for i in self.I:
                for p in self.P:
                    courier = self.data.employees[c]
                    pair = self.data.od_pairs[p]
                    if pair.zone==0:
                        self.l[(c, i, p)] = courier.var_cost_intra
                    else:
                        self.l[(c, i, p)] = courier.var_cost_aglo
        #print("courier cost intra: ", courier.var_cost_intra, " courier cost agglo ", courier.var_cost_aglo)

    def set_of_external_costs(self):
        for i in self.I:
            for p in self.P:
                zone=self.data.od_pairs[p].zone
                if zone==0:
                    self.c[(i, p)] = self.external_cost_intra
                else:
                    self.c[(i, p)] = self.external_cost_agglo
        #print("external cost intra: ", self.external_cost_intra, " external cost agglo ", self.external_cost_agglo)

        #exit(1)

    def set_of_capacities(self):
        for c in self.C:
            for i in self.I:
                for p in self.P:
                    self.beta[(c, i, p)] = self.data.employees[c].capacity

    def set_of_demands(self):
        for d in self.D:
            for i in self.I:
                for w in self.Omega_i_d[d][i]:
                    for p in self.P:
                        try:
                            self.d[(d, i, w, p)] = self.data.od_pairs[p].demand_od_pair_d[d][i][w]
                        except:
                            self.d[(d, i, w, p)] = 0.0

    def set_of_mean_demands(self):
        for d in self.D:
            for i in self.I:
                for p in self.P:
                    mean_demand = 0
                    try:
                        nb_scen=self.data.od_pairs[p].nb_scenarios_d[d][i]
                    except:
                        nb_scen=0
                    for w in range(nb_scen):
                        dem=self.data.od_pairs[p].demand_od_pair_d[d][i][w]
                        proba=self.data.od_pairs[p].probability_d[d][i][w]
                        mean_demand+=dem*proba
                    self.d[(d, i, 0, p)] = mean_demand

    def set_of_mu(self):
        for c in self.C:
            for d in self.D:
                for i in self.I:
                    for w in self.Omega_i_d[d][i]:
                        for p in self.P:
                            #self.mu[(c, d, i, w, p)] = min(self.d[(d, i, w, p)], self.beta[(c, i, p)])
                            self.mu[(c, d, i, w, p)] = self.beta[(c, i, p)]
        # print("courier ",0, " time period ", 3, " o-d pair ", 1, self.beta[(0, 3, 1)])
        # print("courier ", 1, " time period ", 3, " o-d pair ", 1, self.beta[(1, 3, 1)])
        # print("courier ", 0, " time period ", 8, " o-d pair ", 4, self.beta[(0, 8, 4)])
        # print("courier ", 1, " time period ", 8, " o-d pair ", 4, self.beta[(1, 8, 4)])
        #
        #
        # exit(1)

    def set_of_probabilities(self):
        for d in self.D:
            for i in self.I:
                for w in self.Omega_i_d[d][i]:
                    a = 0.0
                    for p in self.P:
                        pair = self.data.od_pairs[p]
                        try:
                            a = max(a, pair.probability_d[d][i][w])
                        except:
                            pass
                    self.proba[(d, i, w)] = a

    # probabilities are all equal to one since there is only one scenario in the mean value problem
    def set_of_mean_probabilities(self):
        for d in self.D:
            for i in self.I:
                for w in self.Omega_i_d[d][i]:
                    self.proba[(d, i, w)]=1.0

    def instanciate_parameters(self):
        #print("Instanciating parameters")
        self.set_of_days()
        self.set_of_periods()
        self.set_of_pairs()
        self.set_of_couriers()
        self.set_of_patterns()
        self.set_of_scenarios()
        self.set_of_fixed_costs()
        self.set_of_delta()
        self.set_of_alpha()
        self.set_of_sigma()
        self.set_of_variable_costs()
        self.set_of_external_costs()
        self.set_of_capacities()
        self.set_of_demands()
        self.set_of_mu()
        self.set_of_probabilities()
        # self.print_weekly_parameters()
        # print("Parameters have been instanciated")

    # instanciate the parameters to solve the mean value problem
    def instanciate_mean_parameters(self):
        print("Instanciating mean parameters")
        self.set_of_days()
        self.set_of_periods()
        self.set_of_pairs()
        self.set_of_couriers()
        self.set_of_patterns()
        self.set_of_mean_scenarios()
        self.set_of_fixed_costs()
        self.set_of_delta()
        self.set_of_alpha()
        self.set_of_sigma()
        self.set_of_variable_costs()
        self.set_of_external_costs()
        self.set_of_capacities()
        self.set_of_mean_demands()
        self.set_of_mu()
        self.set_of_mean_probabilities()
        #self.print_weekly_parameters()
        print("Mean parameters have been instanciated")

    def print_weekly_parameters(self):
        print("Set of couriers", self.C)
        print("Set of days", self.D)
        print("Set of periods:", self.I)
        print("Number of patterns", len(self.data.patterns))
        print("Number of od-pairs", len(self.P))
        for c in self.C:
            print("Employee {}: {}".format(c, len(self.Tc[c])))




#Weekly model
def define_model_weekly(param):

    model = Model("ssla")

    # define the variables
    x_idx = [(c, t) for c in param.C for t in param.Tc[c]] #these are the children of the root node in the grammar
    y_idx = [(c, d, i, p) for c in param.C for d in param.D for i in param.I for p in param.P]
    v_idx = [(c, d, i, w, p) for c in param.C for d in param.D for i in param.I for w in param.Omega_i_d[d][i] for p in param.P]
    e_idx = [(d, i, w, p) for d in param.D for i in param.I for w in param.Omega_i_d[d][i] for p in param.P]

    model.x = model.continuous_var_dict(x_idx, lb=0.0, ub=1.0, name="X")
    model.y = model.binary_var_dict(y_idx, name="Y")
    model.v = model.continuous_var_dict(v_idx, name="V")
    model.e = model.continuous_var_dict(e_idx, name="E")

    x = model.x
    y = model.y
    v = model.v
    e = model.e

    for var in model.iter_variables():
        var_type=var.name[0]
        if var_type=="X": var.type="X"
        if var_type=="Y": var.type="Y"
        if var_type=="V": var.type="V"
        if var_type=="E": var.type="E"

    ### CONSTRAINTS ###

    # constraint (2) each courier is affected to at most one pattern
    for c in param.C:
        model.add_constraint(model.sum(x[c, t] for t in param.Tc[c]) <= 1,
                             "constraint_2_{}".format(c))

    # constraint (3) the courier is affected to an od-pair during a working period
    for c in param.C:
        for d in param.D:
            for i in param.I:
                delta = lambda t: param.delta_d[(c, d, i, t)]
                model.add_constraint((model.sum(y[(c, d, i, p)] for p in param.P) -
                    (model.sum(delta(t) * x[(c, t)] for t in param.Tc[c]))) == 0,
                    "constraint_3_{}_{}_{}".format(c, d, i))

    # constraint (5) constraint limiting the number of packages that an internal courier can deliver
    for c in param.C:
        for d in param.D:
            for i in param.I:
                for w in param.Omega_i_d[d][i]:
                    for p in param.P:
                        model.add_constraint((v[(c, d, i, w, p)] - (param.mu[(c, d, i, w, p)] * y[(c, d, i, p)])) <= 0,
                                         "constraint_5_{}_{}_{}_{}_{}".format(c, d, i, w, p))

    # constraint (6) constraint for the demand satisfaction
    for d in param.D:
        for i in param.I:
            for w in param.Omega_i_d[d][i]:
                for p in param.P:
                    model.add_constraint((model.sum(v[(c, d, i, w, p)] for c in param.C) + e[(d, i, w, p)]) == param.d[(d, i, w, p)],
                                         "constraint_6_{}_{}_{}_{}".format(d, i, w, p))

    ### OBJECTIVE ###
    model.fixed_cost = model.sum(model.sum(param.fixed_costs[(c, t)] * x[(c, t)] for t in param.Tc[c]) for c in param.C)
    model.scenario_cost = model.sum(model.sum(model.sum(param.proba[(d, i, w)] *(
        model.sum(model.sum(param.l[(c, i, p)]*v[(c, d, i, w, p)] for p in param.P) for c in param.C)
        + model.sum(param.c[(i, p)]*e[(d, i, w, p)] for p in param.P)
    ) for w in param.Omega_i_d[d][i]) for i in param.I) for d in param.D)
    model.minimize(model.fixed_cost + model.scenario_cost)

    return model

def run_weekly(param, log_output=False, information=False, timeout=1800, obj_tolerance=1e-04, abs_obj_tolerance=1e-06,
               decompose_costs=False):
    ti=timer.Timer()
    time_weekly = time.time()
    time_define_weekly = time.time()
    m = define_model_weekly(param)
    m.timer=ti
    time_define_weekly = time.time() - time_define_weekly
    time_solve_weekly = time.time()
    # set Tolerance
    m.parameters.mip.tolerances.mipgap.set(obj_tolerance)
    m.parameters.mip.tolerances.absmipgap.set(abs_obj_tolerance)
    m.parameters.benders.strategy=-1
    m.set_time_limit(timeout)
    m.solve(log_output=log_output)
    time_solve_weekly = time.time() - time_solve_weekly
    time_weekly = time.time() - time_weekly
    if information: print(m.print_information())
    print("Optimal:", m.objective_value)
    print("Total Time for complete weekly model {} ({} sec. Defining / {} sec. solving )".format(
        time_weekly, time_define_weekly, time_solve_weekly))
    fix_cost=0
    var_cost=0
    ext_cost=0
    if decompose_costs:
        for var in m.iter_variables():
            if var.name[0]=="X":
                c, t=var.get_key()
                fix_cost+=param.fixed_costs[(c, t)]*var.solution_value
            if var.name[0]=="V":
                c, d, i, w, p=var.get_key()
                var_cost+=param.l[(c, i, p)]*var.solution_value*param.proba[(d, i, w)]
            if var.name[0]=="E":
                d, i, w, p=var.get_key()
                ext_cost+=param.c[(i, p)]*var.solution_value*param.proba[(d, i, w)]
        return m, fix_cost, var_cost, ext_cost
    return m

