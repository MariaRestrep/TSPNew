src:
	main.py: contains the classes and functions to parse the data plus some functions to execute the arguments parsed from arguments.py
	model_weekly.py: contains a class of parameters used to define the complete weekly model and in the branch_and_cut.py file
	branch_and_cut.py: contains function to define and solve the two-stage recourse problem with the branch and cut 
	pricing.py: constains function related to the pricing 
	arguments.py: contains the argument parser for main.py
	timer.py: contains function to measure the time spent in the resolution process
	exporter.py: contains function to export the results 
		
	
	Arguments:
		instance_size: the number of orders of the instance to parse
		instance_number: the number of the instance 
		-n, -g: the distrubution of the instance (resp. normal, gamma) (default normal)
		-me, -hi: the variability of the instance (resp. medium, high) (default medium)
		-t: timeout (default 1800) for the B&C
		-eps: epsilon tolerance for the first phase (default 0.05)
		-c: the number of couriers to consider 
		--rec: solve the instance with the recourse solution
		--ev: solve the instance with the mean value problem and resolve the recourse problem with the first stage decisions taken
		--day_heu: solve the problem independatly for each day (with the shifts) and make the solution heuristacly feasible for a week
		--no_exp: disable the export (default False)
		--full_patterns, --price_pattern_flex, --price_patterns: define the method to generate patterns and the weekly working rules
			(weekly working rules corresponding to each method can be manualy changed in main.py)
			
	exemple of command line: main.py 50 0 -n -me --rec --ev --day_heu -c 10 --price_patterns -t 3600 
		solve the instance located at Instances/collection_name/50/normal/medium/instance_0/ 
		with price_patterns with the recourse problem, the expect value problem and the per day resolution + heuristic with 10 couriers
			
	Weekly working rules:
		Shifts:
			min_shift_length
			max_shift_length
			shift_gap: interval between 2 shifts of the same duration
		Patterns:
			min_working_hours, max_working_hours
		Couriers:
			AVAILABILITIES_DURATION: list of possible availability duration per day to assign to each couriers (randomly assigned) 
				(can reduce greatly the number of compatible patterns with a courier, and thus can have an important impact on the problem)
			FULL_AVAILABILITY: ignore the availability (couriers are considered always available and can thus work in every pattern)
			
			
	Parameters (in main.py):
		fixed_cost: cost per hour of a courier (total fixed cost also depends on the vehicle)
		agglo_var_cost_coef: variable cost coefficient to increase the cost of couriers deliveries in agglomeration
		agglo_ext_cost_coef: external cost coefficient to increase the cost of external couriers deliveries in agglomeration
		external_cost_intra: external cost per package intra 
		external_cost_agglo: external cost per package in agglomeration (depends on external_cost_intra and agglo_ext_cost_coef) 
		pattern_fixed_cost: fixed cost for the affectation of a pattern to a courier 
		DAYS: the set of days
		SEED: set the seed for the random components to have the same data on all resolution (vehicle affectation and periods availability of couriers are random)
		COLLECTION: the name of the collection in which the instance is located (in instances/)
		
		
data:
	VehicleParameters.csv
	PostalCodesWithCentroidsGen.csv
	OrderDistributionWeekly.csv
	OrderDistributionWeeklyHighVar.csv
	
Instances: 
	main.py: generate a collection of instances 
		parameters:
			* NB_TIME_PERIODS: the number of time periods on each day
			* NB_DAYS: the number of days 
			* collection_name: the name of the collection (directory name in instances/ that will contain the data generated)
			* total_order_list: a list of orders 
			* distribution_type_list: a list of distribution type
			* variability_type_list: a list of variability type
			* instances_per_class: the number of instances to generate for each class
		Generate a collection (collection_name) of instances_per_class instances for class of instance (a combination of orders, distribution, variability)
			
	
	orders_per_time_period.py: file used to generate the demand
	
	collection: a collection of 10 instances for each class
		* instances are regrouped per class of instance: collection/orders/distribution/variability/
		* each instance of the same class is identified as a number, instance data are located in collection/orders/distribution/variability/instance_number/
			where instance_number/ is a directory containing a csv file for the demand on each day (stochastic_0.csv is the demand on day 0, ...)
			
Benchmarks:
	Directory used to export the results
	Follow the same structure as Instances: the results of an instance are exported to benchmarks/collection/orders/distribution/variability/instance_number/
	
	Results:
		results of an instane are exported in benchmarks/collection/orders/distribution/variability/instance_number/data_string.csv
	where each row in data_string.csv corresponds to a resolution of this instance in the format (explained in more details in exporter.py):
		Collection, Orders, Distribution, Variability, id, Days, Time periods, OD pairs, couriers, Shifts, Min hours, Max hours, Interval, Patterns, Min whours, max whours, 
		Gen. method, Gen. time, Per day time, Per day obj, Per day fix, Per day var, Per day ext, Per day repair, Epsilon, Gap tolerance, Total time, Rec def time, P1 time, P2 time, 
		Rec gap, Rec obj, Rec fix, Rec var, Rec ext, Rec couriers, Mean time, Mean gap, Mean obj, Mean fix, Mean var, Mean ext
	
	bench_csv.py  can be used to compute a single csv file from all csv file of a collection of instances
	