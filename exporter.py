# file containing functions to export results
import os

# create the benchmark directory for the instance or make it current if it already exists
def create_bench_dir(bench_dir, instance_size, instance_number, distribution, variability):

    current_bench_dir="{}/{}/{}/".format(bench_dir, instance_size, distribution)
    instance_name="instance_{}".format(instance_number)
    if instance_name not in os.listdir(current_bench_dir):
        os.mkdir(current_bench_dir+instance_name)
    current_bench_dir+="instance_{}/".format(instance_number)
    return current_bench_dir


# export the data in data_list as a csv string into a csv file
#data list:
    # Instances: Instance COLLECTION name, Distribution, Variability, id,
    # Instance parameters: Orders, Distribution (normal or gamma), Variability (medium or high), id, Days, Time periods,
    #                      OD pairs,
    # Staffing parameters: Number of couriers, Number of Shifts (depends on Min hours, Max hours and Interval),
    #                      Min hours (shift duration), Max hours (shift duration), Interval (hours between 2 shifts of the same duration),
    #                      Number of Patterns, min whours (minimum weekly hours), max whours (maximum weekly hours),
    #                      Gen. method (method to generate patterns), Gen. time (time to generate patterns),
    # Per day resolution (solve each day independantly):
    #                      Per day time, Per day obj (total cost), Per day fix (fixed cost), Per day var (variable cost),
    #                      Per day ext (external cost),
    #                      Per day repair (objective after making the per day solution feasible for a week,
    # Recourse two-stage resolution:
    #                      Resolution parameters: Epsilon, Gap tolerance,
    #                      Resolution details: Total time, Rec def time (time defining the model),
    #                      P1 time (time reaching epsilon), P2 time (B&C time), Rec gap (Gap at the end of the resolution),
    #                      Costs: Rec obj, Rec fix, Rec var, Rec ext,
    #                      Rec couriers: Number of couriers (number of x variables at 1)
    # Expected value resolution:
    #                      EV time , EV gap,
    #                      EV obj, EV fix, EV var, EV ext
def export_data_string(file, data_list):
    io=open(file, "a")
    cpt=0
    for item in data_list:
        cpt+=1
        if cpt>=len(data_list):
            tail="\n"
        else:
            tail=","
        io.write(str(item)+tail)
    io.close()