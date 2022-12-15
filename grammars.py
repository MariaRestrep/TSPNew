from lark import Lark, tree
import numpy as np
import pandas as pd
import networkx as nx
import pygraphviz
import matplotlib.pyplot as plt
import lark.parsers.cyk as cyk
from lark.parsers.cyk import *

TOLERANCE = 0.0001

#cyk._parse = newparse_cyk

#include the preferred grammar
#it can be read from a text file
#TODO: generalize for as many shifts as wanted and for the preferred structure
# def get_grammar_string(data):
#     grammar_string = f"""
#         start: s
#         s : r f r | r f | f r
#         r : R ~ 1..2
#         f : d d_0 | d | n n_0 | n | a a_0 | a
#         d_0 : n n_0 | n | a a_0 | a
#         n_0 : d d_0 | d | a a_0 | a
#         a_0 : d d_0 | d | n n_0 | n
#         d : S0 ~ 3..5
#         n : S1 ~ 2..3
#         a : S2 ~ 2..3
#         R  : "r"
#         {get_shift_terminals(data)}
#                     """
#     return grammar_string

def get_grammar_string(data):
    grammar_string = f"""
        start: s
        s : r f r | r f | f r
        r : R ~ 1..1
        f : S0 ~ 1..6
        R  : "r"
        {get_shift_terminals(data)}
                    """
    print(grammar_string)
    return grammar_string


def get_grammar_string_2(data):
    print("GRAMMAR 2")
    grammar_string = f"""
        start: s 
        s : r f r | r f | f r 
        r : R ~ 1..2     
        {get_production_full_shifts(data)}
        {get_production_change_shifts(data)}
        {get_production_shifts(data)}
        R  : "r"  
        {get_shift_terminals(data)}
                    """
    print (grammar_string)
    return grammar_string

#{get_production_shifts(data)}

def get_production_shifts(data):
    result = ""
    for i in range(len(data.shifts)):
        result += f"s_{i} : S{i} ~ {data.shifts[i].min_duration}..{data.shifts[i].max_duration}"
        #result += f"s_{i} : S{i} ~ {data.shifts[i].min_duration}..{data.shifts[i].max_duration}"
        #result += f"s_{i} : S{i} ~ 2..3"

        if i < len(data.shifts) - 1:
            result += "\n"
    return result


def get_production_full_shifts(data):
    result = "f : "
    for i in range(len(data.shifts)):
        result += f"s_{i} s_{i}_0 | s_{i}"
        if i < len(data.shifts)-1:
            result += " | "
    return result


def get_production_change_shifts(data):
    result = ""
    for i in range(len(data.shifts)):
        k = 0
        result += f"s_{i}_0 : "
        for j in range(len(data.shifts)):
            if i != j:
                if k > 0:
                    result += " | "
                result += f"s_{j} s_{j}_0 | s_{j}"
                k += 1
        if i < len(data.shifts)-1: result += "\n"
    return result


#TODO: generalize
def get_shift_terminals(data):
    result = ""
    for i in range(len(data.shifts)):
        result += f"""S{i} : "{i}" \n"""

    return result


#build the directed acyclic graph contaning all the and-nodes and or-nodes
def build_dag(grammar_string, length_bounds, max_length):

    #print(grammar_string)
    length_bounds['start'] = (max_length, max_length)

    #call the parser
    p = Lark(grammar_string, parser="cyk")

    cnf_grammar = p.parser.parser.parser.grammar
    table, trees, graph = build_dag_of_parse_trees(max_length, cnf_grammar, p, length_bounds)
    #print("Nodes before removal: ", graph.number_of_nodes())
    remove_nodes(graph) #remove unnecessary nodes
    #print("Nodes after removal: ", graph.number_of_nodes())
    return graph


def remove_nodes(graph):
    nodes_to_remove = []
    for n in graph.nodes():
        if graph.in_degree(n) == 0 and not (n.node_type == "or" and n.symbol == "start"):
            # print (n)
            nodes_to_remove.append(n)

    for node in nodes_to_remove:
        for (source, target) in graph.out_edges(node):
            if graph.in_degree(target) == 1:
                nodes_to_remove.append(target)

        graph.remove_node(node)


def build_dag_of_parse_trees(n_periods, g, p, length_bounds):
    graph = nx.DiGraph()

    # The CYK table. Indexed with a 2-tuple: (start pos, end pos)
    table = defaultdict(set)
    # Top-level structure is similar to the CYK table. Each cell is a dict from
    # rule name to the best (lightest) tree for that rule.
    trees = defaultdict(dict)
    # Populate base case with existing terminal production rules

    for i in range(n_periods):
        for terminal, rules in g.terminal_rules.items():
            for rule in rules:
                table[(i, i)].add(rule)

                if (rule.lhs not in trees[(i, i)] or
                        rule.weight < trees[(i, i)][rule.lhs].weight):
                    trees[(i, i)][rule.lhs] = RuleNode(rule, [terminal], weight=rule.weight)
                    #print (i, " ", terminal, rule.lhs.name)
                    # print ()
                add_terminal_or_node(graph, i, rule, p.get_terminal(rule.rhs[0].name).pattern.value)
                print("i ", i, " rule ", rule, " pattern: ", p.get_terminal(rule.rhs[0].name).pattern.value)


    #print("***************************")
    # return
    # Iterate over lengths of sub-sentences
    for l in xrange(2, n_periods + 1):
        # Iterate over sub-sentences with the given length
        for i in xrange(n_periods - l + 1):
            # Choose partition of the sub-sentence in [1, l)
            for p in xrange(i + 1, i + l):
                span1 = (i, p - 1)
                span2 = (p, i + l - 1)
                for r1, r2 in itertools.product(table[span1], table[span2]):
                    for rule in g.nonterminal_rules.get((r1.lhs, r2.lhs), []):
                        # here: check rule
                        print ("start: ", i, " lenght: ", l, " span 1 ", span1, " span 2 ", span2)
                        if not check_rule_indices(rule, i, l, length_bounds):
                            continue

                        table[(i, i + l - 1)].add(rule)
                        r1_tree = trees[span1][r1.lhs]
                        r2_tree = trees[span2][r2.lhs]
                        rule_total_weight = rule.weight + r1_tree.weight + r2_tree.weight
                        # if (rule.lhs not in trees[(i, i + l - 1)]
                        #    or rule_total_weight < trees[(i, i + l - 1)][rule.lhs].weight):
                        trees[(i, i + l - 1)][rule.lhs] = RuleNode(rule, [r1_tree, r2_tree], weight=rule_total_weight)
                        # print (i, " ", trees[(i, i + l - 1)][rule.lhs].rule)
                        add_and_or_node_and_link(graph, i, l, p - i, rule)

    return table, trees, graph


def get_work_leaf_nodes(graph):
    return [node for node in graph if (graph.out_degree(node) == 0 and node.symbol.isnumeric())]


def add_terminal_or_node(graph, start_period, rule, pattern):
    or_terminal = NodeInfo("or", start_period, 1, rule.lhs.name, pattern)
    graph.add_node(or_terminal)
    # print(type(rule.lhs.name))
    # print(rule)
    #print("symbol or terminal", or_terminal.symbol, "pattern ", pattern)
    # graph.nodes[or_terminal]['group'] = 'or1'


class NodeInfo:
    def __init__(self, node_type, start_period, length, lhs, symbol = None, rhs=None, length_first=None, solution=None, id = None):
        self.node_type = node_type
        self.start_period = start_period
        self.length = length
        if symbol is None:
            self.symbol = lhs
        else:
            self.symbol = symbol
        self.lhs = lhs
        self.rhs = rhs
        self.length_first = length_first
        self.solution = 0
        self.id = None

    def __repr__(self):
        if self.node_type == "or":
            return f" O {self.lhs} {self.start_period},{self.length}"
        else:
            return f" A {self.lhs} {self.start_period},{self.length},{self.length_first},{self.rhs}"

    def __eq__(self, other):
        return self.__repr__() == other.__repr__()

    def __hash__(self):
        return hash(self.__repr__())


def check_rule_indices(rule, index, length, length_bounds):
    if rule.lhs.name in length_bounds:
        lb, ub = length_bounds[rule.lhs.name]
        if length < lb or length > ub:
            return False
    return True



def add_and_or_node_and_link(graph, start_period, length, length_first, rule):
    # print(type(rule), rule, start_period, length)

    parent_or_node = NodeInfo("or", start_period, length, rule.lhs.name)

    and_node = NodeInfo("and", start_period, length, rule.lhs.name, rule.lhs.name, rule.rhs[0].name + rule.rhs[1].name,
                        length_first)

    graph.add_edge(parent_or_node, and_node)  # automatically adds nodes

    graph.nodes[and_node]['group'] = 'and' + str(length)

    length_second = length - length_first

    or_node_1 = NodeInfo("or", start_period, length_first, rule.rhs[0].name)
    graph.add_edge(and_node, or_node_1)
    graph.nodes[or_node_1]['group'] = 'or' + str(length_first)

    if length > length_first:
        or_node_2 = NodeInfo("or", start_period + length_first, length_second, rule.rhs[1].name)
        graph.add_edge(and_node, or_node_2)
        graph.nodes[or_node_2]['group'] = 'or' + str(length_second)


def build_mip_component_from_dag(m, graph, flow_size=None, personalized=False):
    for node in graph:

        # Flow size denotes the number of employees assigned to the tours
        # If flow size is None, the model will choose the number of tours to use
        # If the model is personalized, the decision variables are binary, otherwise they are integer
        if flow_size is not None and graph.in_degree(node) == 0:
            var = m.integer_var(lb=flow_size, ub=flow_size)
        else:
            if personalized:
                var = m.binary_var()
            else:
                var = m.integer_var()
        graph.nodes[node]['var'] = var

    for node in graph:
        if node.node_type == 'or':
            ##child-constraint
            if graph.out_degree(node) > 0:
                m.add_constraint(graph.nodes[node]['var'] == m.sum(
                    graph.nodes[and_node]['var'] for and_node in graph.successors(node)))
            if graph.in_degree(node) > 0:
                m.add_constraint(graph.nodes[node]['var'] == m.sum(
                    graph.nodes[and_node]['var'] for and_node in graph.predecessors(node)))

#postprocessing to build the schedules and to check if everyhting is OK
#TODO: check why it is printing a weird symbol...
def build_schedules(m, data, graph, personalized = None):

    #initialize the schedules with rest time (or days-off)
    schedules = ["r"] * len(data.employees)
    for e in range(len(data.employees)):
        schedules[e] = ["r"] * int(data.periods_grammar)


    for e in m.employees:
        schedules_built = 0
        stack = []

        id = 0
        for node in graph[e]:
            node.solution = graph[e].nodes[node]['var'].solution_value
            node.id = id
            id += 1 #improve

            # if node.solution > 0:
            #     print(graph[e].nodes[node]['var'].solution_value, node.start_period, node.length, node.node_type)


    #     while schedules_built < data.number_of_employees:
        while schedules_built < 1:
            id_node = -1
            for node in graph[e]:
                #print("degree", graph.in_degree(node))
                if graph[e].in_degree(node) == 0:
                    #print("in degree ", graph.in_degree(node))
                    for ch_root in graph[e].successors(node):
                        #print("sol", graph.nodes[ch_root]['var'].x)
                        if(ch_root.solution > TOLERANCE):
                            #print(graph[e].nodes[ch_root]['var'].solution_value, ch_root.start_period, ch_root.length, ch_root.node_type)
                            id_node = ch_root.id
                            #print("*** id node", id_node)
                            break
                        else:
                            continue
                        break

            assert(id_node != -1)
            assert(len(stack) == 0)

            #print("id node", id_node)

            stack.append(id_node)
            schedules_built += 1

            while len(stack) > 0:

                current_and_id = stack.pop()
                print(current_and_id)
                #find the node
                for node in graph[e]:
                    if node.id == current_and_id:
                        current_and_node = node
                        node.solution = node.solution - 1
                        break
                #print("successors current and", graph.out_degree(current_and_node), current_and_node.start_period, current_and_node.length)


                if graph[e].out_degree(current_and_node) == 1: #leaf
                    #print("leaf")

                    current_or_node = graph.successors(current_and_node)
                    #find the terminal

                    #print("terminal2!!!!!", current_or_node.symbol, current_or_node.length, current_or_node.start_period, current_or_node.lhs)

                else: #if  children (child) of the and node is not a leaf
                    #print("out degree", graph.out_degree(current_and_node))
                    for node in graph[e].successors(current_and_node): #left and right children

                        #print("node out", graph.out_degree(node))

                        if(graph[e].out_degree(node) == 0):
                            #print("terminal!!!!!", node.symbol, node.length, node.start_period, node.lhs)
                            schedules[e][node.start_period] = node.symbol

                        for node2 in graph[e].successors(node):
                            #print("node 2 out", graph.out_degree(node2))
                            if(node2.solution > 0):
                                stack.append(node2.id)
                                #print("node2 id", node2.id)
                                break
                    #print("len_stack", len(stack))

    return schedules
