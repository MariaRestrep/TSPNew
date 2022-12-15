import os

def export_csv(bench_collection, csv_file):
    head_string="Collection, Orders, Distribution, Variability, id, Days, Time periods, OD pairs, couriers, Shifts, " \
                "Min hours, Max hours, Interval, Patterns, Min whours, max whours, Gen. method, Gen. time, Per day time," \
                " Per day obj, Per day fix, Per day var, Per day ext, Per day repair, Epsilon, Gap tolerance, " \
                "Total time, Rec def time, P1 time, P2 time, Rec gap, Rec obj, Rec fix, Rec var, Rec ext, Rec couriers, " \
                "Mean time, Mean gap, Mean obj, Mean fix, Mean var, Mean ext\n"
    csv_string=head_string
    size_list=os.listdir(bench_collection)
    for i in size_list:
        distrib_list=os.listdir(bench_collection+"{}/".format(i))
        for d in distrib_list:
            var_list=os.listdir(bench_collection+"{}/{}/".format(i, d))
            for v in var_list:
                instance_list=os.listdir(bench_collection+"{}/{}/{}".format(i, d, v))
                print(instance_list)
                for n in instance_list:
                    string_file=bench_collection+"{}/{}/{}/{}/data_string.txt".format(i, d, v, n)
                    try:
                        str="void"
                        io=open(string_file, 'r')
                        while str!="":
                            str=io.readline()
                            csv_string+=str
                        io.close()
                    except Exception as e:
                        print(e)
    io=open(csv_file, "w")
    io.write(csv_string)
    io.close()

collection_dir="collection/"
csv_file=collection_dir+"bench_results.csv"
print(collection_dir)
export_csv(collection_dir, csv_file)