from docplex.mp.model import Model

class Production:
    def __init__(self, left, length, right, minSpan, maxSpan, tws, twl):
        self.left = int(left)
        self.length = int(length)
        self.right = right
        self.minSpan = int(minSpan)
        self.maxSpan = int(maxSpan)
        self.tws = int(tws)
        self.twl = int(twl)
        self.cost = 0.0

class GrammarString:
    def __init__(self, nbProductions, nbTerminals, nbNonTerminals):
        self.nbProductions = int(nbProductions)
        self.nbTerminals = int(nbTerminals)
        self.nbNonTerminals = int(nbNonTerminals)

        self.productions = []

    def addProduction(self, prod):
        self.productions.append(prod)
        self.nbProductions+=1



class LeafNode:
    def __init__(self, terminal, position):
        self.terminal = int(terminal) #see if it need to be a terminal
        self.position = int(position)

class OrNode:
    def __init__(self, nbAncestors, nbChildren, cost, idN, position, span, span2, symbol):

        self.cost = float(cost)
        self.idN = int(idN)
        self.id = 0
        self.position = int(position)
        self.span = int(span)
        self.span2 = int(span2)
        #self.nodeType = int(type) #0->leaf, 1->inner
        self.symbol = int(symbol)
        self.leaf = LeafNode(symbol, position)
        self.decisionVariable = None

        self.nbAncestors = int(nbAncestors)
        self.ancestors = []

        self.nbChildren = int(nbChildren)
        #self.inner = InnerNode()
        self.children = []

    def addAncestors(self, andNode):
        self.ancestors.append(andNode)
        self.nbAncestors+=1

    def addChildren(self, andNode):
        self.children.append(andNode)
        self.nbChildren+=1

class AndNode:
    def __init__(self, leftChild, rightChild, parent, production, idN):
        self.leftChild =leftChild
        self.rightChild = rightChild
        self.parent = parent #an or-node
        self.production = production
        self.idN = 0
        self.id = 0

        self.decisionVariable = None


#see if we need this
class InnerNode:
    def __init__(self):
        self.children = []


class GrammarGraph:
    def __init__(self):

        self.nbLeaves = 0
        self.nbOrNodes = 0
        self.NbNodes = 0
        self.sequenceLenght = 0
        self.postOrderNodes = []
        self.root = None

    def addPostOrderNode(self, node):
        self.postOrderNodes.append(node)
        self.nbOrNodes += 1


def get_grammar_string_3(data):
    print("GRAMMAR 3")
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



def readGrammar(data):
    # # TODO: read the string
    # # this is temporary
    # nbTerminals = 4
    # nbNonTerminals = 6
    # nbProductions = 9
    #
    # g = GrammarString(nbProductions, nbTerminals, nbNonTerminals)
    #
    # # W->WW self, left, length, right, minSpan, maxSpan, tws, twl, cost
    # ps_w1 = Production(8, 1, [0], 0, 5, 0, 5)
    # g.addProduction(ps_w1)
    #
    # # W->WW self, left, length, right, minSpan, maxSpan, tws, twl, cost
    # ps_w2 = Production(8, 1, [1], 0, 5, 0, 5)
    # g.addProduction(ps_w2)
    #
    # # B->b self, left, length, right, minSpan, maxSpan, tws, twl, cost
    # ps_b = Production(9, 1, [2], 0, 5, 0, 5)
    # g.addProduction(ps_b)
    #
    # # B->b self, left, length, right, minSpan, maxSpan, tws, twl, cost
    # ps_r = Production(5, 1, [3], 0, 5, 0, 5)
    # g.addProduction(ps_r)
    #
    # # S->RF self, left, length, right, minSpan, maxSpan, tws, twl, cost
    # ps_1 = Production(4, 2, [5, 6], 0, 5, 0, 5)
    # g.addProduction(ps_1)
    #
    # # S->FR self, left, length, right, minSpan, maxSpan, tws, twl, cost
    # ps_2 = Production(4, 2, [6, 5], 0, 5, 0, 5)
    # g.addProduction(ps_2)
    #
    # # F->XW self, left, length, right, minSpan, maxSpan, tws, twl, cost
    # ps_f = Production(6, 2, [7, 8], 0, 5, 0, 5)
    # g.addProduction(ps_f)
    #
    # # F->XW self, left, length, right, minSpan, maxSpan, tws, twl, cost
    # ps_x = Production(7, 2, [8, 9], 0, 5, 0, 5)
    # g.addProduction(ps_x)
    #
    # # W->WW self, left, length, right, minSpan, maxSpan, tws, twl, cost
    # ps_w = Production(8, 2, [8, 8], 0, 5, 0, 5)
    # g.addProduction(ps_w)

    #TODO: read the string
    #this is temporary
    nbTerminals = len(data.shifts) + 1
    nbNonTerminals = 4
    nbProductions = 0
    periods_grammar = 7

    g = GrammarString(nbProductions, nbTerminals, nbNonTerminals)

    # F -> w
    for i in range(len(data.shifts)):

        ps = Production(len(data.shifts) + 3, 1, [i],
                                               data.shifts[i].min_duration, data.shifts[i].max_duration,
                                               data.shifts[i].tws, data.shifts[i].twl)
        g.addProduction(ps)


    # F->FF self, left, length, right, minSpan, maxSpan, tws, twl, cost
    ps_FF = Production(len(data.shifts)+ 3, 2, [len(data.shifts) + 3, len(data.shifts) + 3],
                       0, periods_grammar, 0, periods_grammar)
    g.addProduction(ps_FF)


    # R->r self, left, length, right, minSpan, maxSpan, tws, twl, cost
    ps_R = Production(len(data.shifts) + 2, 1, [len(data.shifts)],
                       0, 1, 0, periods_grammar)
    g.addProduction(ps_R)


    # Q->FR self, left, length, right, minSpan, maxSpan, tws, twl, cost
    ps_Q = Production(len(data.shifts) + 4, 2, [len(data.shifts) + 3, len(data.shifts) + 2],
                       0, periods_grammar, 0, periods_grammar)
    g.addProduction(ps_Q)

    # S->RF self, left, length, right, minSpan, maxSpan, tws, twl, cost
    ps_s1 = Production(len(data.shifts) + 1, 2, [len(data.shifts) + 2, len(data.shifts) + 3],
                      0, periods_grammar, 0, periods_grammar)
    g.addProduction(ps_s1)

    # S->RQ self, left, length, right, minSpan, maxSpan, tws, twl, cost
    ps_s2 = Production(len(data.shifts) + 1, 2, [len(data.shifts) + 2, len(data.shifts) + 4],
                      0, periods_grammar, 0, periods_grammar)
    g.addProduction(ps_s2)

    #S->FR self, left, length, right, minSpan, maxSpan, tws, twl, cost
    ps_s3 = Production(len(data.shifts) + 1, 2, [len(data.shifts) + 3, len(data.shifts) + 2],
                      0, periods_grammar, 0, periods_grammar)
    g.addProduction(ps_s3)


    return g

def buildCYKGraph(sequenceLength, grammar):

    #Parser CYK
    orNodesList = []
    id = 0
    for i in range(sequenceLength):
        for c in range(grammar.nbTerminals):
            orNode = OrNode(0, 0, 0.0, id, i, 1, 0, c)
            orNode.leaf.terminal = c
            orNode.leaf.position = i
            orNodesList.append(orNode)

            # print("create orNode leaf in position", i, "terminal ", c, " id", id )
            id+=1


    for i in range(sequenceLength):
        for p in grammar.productions:
            if(p.length == 1): #TODO: check later if the production has permission
                #check if the production is within the time window
                if(i >= p.tws and i < p.tws + p.twl):
                    #print("production left ", p.left, " right ", p.right[0])

                    #find the parent of the production
                    parent = find_orNode(orNodesList, i, 0, p.left)
                    if parent == None:
                        parent = OrNode(0, 0, 0.0, id, i, 1, 0, p.left)

                        #orNode.inner.succ = []
                        orNodesList.append(parent)

                        #print("create orNode parent of leaf", p.right[0], " in position", i, "symbol ", p.left)

                        id += 1

                    child = find_orNode(orNodesList, i, 0, p.right[0])
                    assert(child != None)

                    #create an and-node among the children of the parent
                    andNode = AndNode(child, None, parent, p, id)

                    # print("create orNode parent of leaf", p.right[0], " in position", i, "symbol ",
                    #       p.left, p.minSpan, p.maxSpan, p.tws, p.twl)

                    id += 1

                    parent.addChildren(andNode)
                    child.addAncestors(andNode)



    #TODO: check why the function has permission was implemented
    hasPermission = True

    # // j: span of the rule - 1
    # // prod: Production
    # // i: Beginning of the production
    # // k: The first half of the rule spans form i to i + k - 1 and the second half spans from i = k to i + j - 1
    for j in range(1, sequenceLength):
        for p in grammar.productions:
            if (p.length == 2 and p.minSpan - 1 <= j and j < p.maxSpan):
                for i in range(sequenceLength - j):
                    if (i >= p.tws and i+j < p.tws + p.twl):
                        orNode = find_orNode(orNodesList, i, j, p.left)
                        for k in range(j):
                            #there is an assert weird
                            if hasPermission:# TODO: check later if the production has permission

                                # print("enter", "j", j, "k", k, "i", i)
                                # print("left", i, k, p.right[0])
                                # print("right", i + k + 1, j - k - 1, p.right[1])

                                left = find_orNode(orNodesList, i, k, p.right[0])
                                right = find_orNode(orNodesList, i + k + 1, j - k - 1, p.right[1])

                                if right != None:
                                    if left != None:
                                        #print("found both")
                                        if orNode == None:
                                            #nbAncestors, nbChildren, cost, idN, position, span, span2, symbol
                                            orNode = OrNode(0, 0, 0.0, id, i, j+1, j, p.left)
                                            orNodesList.append(orNode)

                                            id+=1

                                        #Let prod be A -> BC. We create an and-node with left child B and right child C.
                                        #leftChild, rightChild, parent, production, id
                                        andNode = AndNode(left, right, orNode, p, id)
                                        id+=1

                                        orNode.addChildren(andNode)

                                        #add the andNode to the set of ancestors of left and right
                                        left.addAncestors(andNode)
                                        right.addAncestors(andNode)

                                        assert(orNode.children[orNode.nbChildren - 1].idN == andNode.idN)
                                        assert(andNode.parent.idN == orNode.idN)
                                        assert(andNode.leftChild.ancestors[andNode.leftChild.nbAncestors -1].idN == andNode.idN)
                                        assert(andNode.rightChild.ancestors[
                                                andNode.rightChild.nbAncestors - 1].idN == andNode.idN)



    return  buildDAG(orNodesList, sequenceLength, grammar)

    # for node in orNodesList:
    #     print(node.position, node.span2, node.symbol)


def buildDAG(orNodesList, sequenceLength, grammar):

    dag = GrammarGraph()
    dag.sequenceLenght = sequenceLength

    dag.root = find_orNode(orNodesList, 0, sequenceLength - 1, grammar.nbTerminals)
    assert(dag.root != None) #if the root is != None there must be at least one parsing tree

    tableOrNodes = [None] * (2 * sequenceLength)
    top = 0
    tableOrNodes[top]= dag.root

    dag.root.cost = 1.0
    dag.nbOrNodes = 0

    while top >= 0:
        orNode = tableOrNodes[top]
        #print("id OR NODE", orNode.id, " cost", orNode.cost, dag.nbOrNodes)
        assert (orNode.cost == 1.0)
        if(orNode.id < orNode.nbChildren):
            andNode = orNode.children[orNode.id]
            orNode.id+=1
            assert(andNode.rightChild == None or andNode.rightChild.cost != 1.0) #The graph is acyclic
            if (andNode.rightChild != None and andNode.rightChild.cost == 0.0):
                andNode.rightChild.cost = 1.0
                top+=1
                tableOrNodes[top] = andNode.rightChild
                assert (top < 2 * sequenceLength)
                assert (andNode.rightChild.id == 0)

            assert (andNode.leftChild != None)
            assert (andNode.leftChild.cost != 1.0)
            if(andNode.leftChild.cost == 0.0):
                andNode.leftChild.cost = 1.0 #Tag the node as visited
                top += 1
                tableOrNodes[top] = andNode.leftChild
                assert (top < 2 * sequenceLength)
                assert (andNode.leftChild.id == 0)
        else:
            orNode.cost = 2.0
            top = top - 1
            orNode.id = dag.nbOrNodes
            dag.addPostOrderNode(orNode)

            #check
            for node in orNode.children:
                assert(node.leftChild.cost == 2.0)
                assert(node.rightChild == None or node.rightChild.cost == 2.0)


    #Remove all ancestors that cannot be part of a valid parsing tree
    for node in orNodesList:
        if(node != None and node.cost > 0.0):
            for i in reversed(range(node.nbAncestors)):
                if(node.ancestors[i].parent.cost == 0):
                    node.nbAncestors = node.nbAncestors -1
                    node.ancestors[i] = node.ancestors[node.nbAncestors]
            if(node.nbAncestors!=len(node.ancestors)):
                #print("problem !!!!!!!", node.nbAncestors, len(node.ancestors))
                #delete the ancestors
                for  i in reversed(range(node.nbAncestors, len(node.ancestors))):
                    del node.ancestors[i]
                #print("verifying !!!!!!!", node.nbAncestors, len(node.ancestors))



    #Delete nodes that have been removed from the graph
    for node in orNodesList:
        if (node != None and node.cost == 0.0):
            # print("node position ", node.position, " span", node.span, " nb anc", node.nbAncestors, " nb chil",
            #       node.nbChildren, "nbAnce2 ", len(node.ancestors), "nbCh", len(node.children), "symbol", node.symbol)

            for i in reversed(range(node.nbChildren)):
                del node.children[i]


    #Give an id to each and-node
    nextId = dag.nbOrNodes
    for i in range(dag.nbOrNodes):
        orNode = dag.postOrderNodes[i]
        for andNode in orNode.children:
            andNode.id = nextId
            nextId+=1
    dag.nbNodes = nextId
    assert(dag.postOrderNodes[dag.nbOrNodes -1].id == dag.root.id)

    # print("Nb nodes", dag.nbNodes)
    # print("Nb or nodes", dag.nbOrNodes)
    # print("Nb and nodes", dag.nbNodes-dag.nbOrNodes)
    # print("Nb leaves", dag.nbLeaves)
    #
    #
    # print("Root ancestors", dag.root.nbAncestors, "children", dag.root.nbChildren)

    return dag

def build_mip_component_from_dag(m, graph, param, c, flow_size=None, personalized=False):

    #decision variable declaration
    # I am not including the variables associated with the leaves

    # Flow size denotes the number of employees assigned to the tours
    # If flow size is None, the model will choose the number of tours to use
    # If the model is personalized, the decision variables are binary, otherwise they are integer

    # flow_size = 44

    if flow_size is not None:  # the root node
        var = m.continuous_var(lb=flow_size, ub=flow_size)
        graph.root.decisionVariable = var
    else:
        var = m.continuous_var()

        graph.root.decisionVariable = var


    for node in graph.postOrderNodes:
        for n in node.children:
            if n.decisionVariable == None:
                if personalized:
                    var = m.continuous_var()
                else:
                    var = m.continuous_var()
                n.decisionVariable  = var

    #Constraints

    #children of the root node
    m.add_constraint(graph.root.decisionVariable == m.sum(andNode.decisionVariable for andNode in graph.root.children))

    leaves = 0
    # rest of the or nodes excluding the root and the leaves
    for node in graph.postOrderNodes:
        if node.nbAncestors > 0 and node.nbChildren > 0:
            # print("node position ", node.position, " span", node.span, " nb anc", node.nbAncestors, " nb chil",
            #        node.nbChildren, "nbAnce2 ", len(node.ancestors), "nbCh", len(node.children))
            m.add_constraint(
                m.sum(andNodeParent.decisionVariable for andNodeParent in node.ancestors) ==
                m.sum(andNodeChild.decisionVariable for andNodeChild in node.children))


        elif node.nbChildren == 0 and node.leaf.terminal < len(param.shifts):  # the leaves
            m.add_constraint(m.x[c, node.leaf.position, node.leaf.terminal]==m.sum(andNodeParent.decisionVariable for andNodeParent in node.ancestors))

            #print(node.leaf.terminal, " ", node.leaf.position)
            # m.add_constraint(
            #     node.decisionVariable == m.sum(andNodeParent.decisionVariable for andNodeParent in node.ancestors))
            # print(node.leaf.terminal, " ", node.leaf.position)
            # print("node position ", node.position, " span", node.span, " nb anc", node.nbAncestors, " nb chil",
            #       node.nbChildren, "nbAnce2 ", len(node.ancestors), "nbCh", len(node.children))
            # for andNodeParent in node.ancestors:
            #     print(andNodeParent.decisionVariable, node.decisionVariable)
            leaves += 1


                #
                # for node in grammar_graph[c].postOrderNodes:
                #     if node.nbChildren == 0 and node.leaf.terminal < len(param.data.shifts):  # the leaves
                #         # m.add_constraint(node.decisionVariable == m.sum(andNodeParent.decisionVariable for andNodeParent in node.ancestors))
                #         model.add_constraint(node.decisionVariable == x[c, node.leaf.position, node.leaf.terminal])
                #
                #         print(node.leaf.terminal, " ", node.leaf.position)
                #         leaves += 1
                # print(leaves)

    # m.dump_as_lp("lpmodel_grammar")
    #
    # exit(1)
    #print(leaves)

#TODO: improve this search
def find_orNode(list, i, j, symbol):

    for node in list:
        if (node.position == i and node.span2 == j and node.symbol == symbol):
            return node

    return None
