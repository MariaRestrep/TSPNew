import argparse

def argument_parser():
    parser = argparse.ArgumentParser(description='Argument parser.')
    parser.add_argument('instance_size', type=int, help="Instance size (mandatory)")
    parser.add_argument('instance_number', type=int, help="Instance number (mandatory)")
    distrib_group = parser.add_mutually_exclusive_group()
    distrib_group.add_argument('-n', action='store_true', help="Normal distribution", default=True)
    distrib_group.add_argument('-g', action='store_true', help="Gamma distribution", default=False)
    distrib_group.add_argument('-u', action='store_true', help="Uniform distribution", default=False)
    var_group = parser.add_mutually_exclusive_group()
    var_group.add_argument('-me', action='store_true', help="Medium variability", default=True)
    var_group.add_argument('-hi', action='store_true', help="High variability", default=False)
    parser.add_argument('-t', type=int, dest='time', help="Timeout (default=1800)", default=1800) # the timeout
    parser.add_argument('-eps', type=float, default=0.05, help="First phase epsilon tolerance")
    parser.add_argument('-gap', type=int, default=1, help="Coefficient (>=1) multiplying the base mip gap (e-04)")
    parser.add_argument('-c', type=int, default=30, help="Number of couriers")
    parser.add_argument('--gram', default=True, action='store_true', help="Solve the problem with grammars")
    parser.add_argument('--rec', default=True, action='store_true', help="Solve the recourse problem")
    parser.add_argument('--day_heu', default=False, action='store_true', help="Per day resolution + heuristic repair")
    parser.add_argument('--ev', default=False, action='store_true', help="Expected value solution")
    parser.add_argument('--no_exp', default=False, action='store_true', help="Do not export results")
    pattern_group = parser.add_mutually_exclusive_group()
    pattern_group.add_argument('--full_patterns', default=False, action='store_true', help='Generate patterns exhaustively')
    pattern_group.add_argument('--price_patterns_flex', default=False, action='store_true', help='Generate patterns with pricing problem high shift flexibility')
    pattern_group.add_argument('--price_patterns', default=False, action='store_true', help="Generate patterns with pricing problem and low shift flexibility")
    args=parser.parse_args()
    return args

