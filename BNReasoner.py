from typing import Union, List, Tuple
from BayesNet import BayesNet, nx
from copy import deepcopy
import pandas as pd
import matplotlib.pyplot as plt

# Utility functions 
def check_single(variable: Union[str, List[str]]) -> List[str]:
    """ Checks if the variable is a single variable and returns a list containing the variable if it is.

    Args:
        variable (Union[str, List[str]]): Either a single variable or a list of variables.

    Returns:
        List[str]: A list containing the variable, otherwise the variable list.
    """
    return variable if type(variable) == list else [variable]

# BNReasoner class
class BNReasoner:
    def __init__(self, net: Union[str, BayesNet]) -> None:
        """ Initializes a BNReasoner object.
        
        Args:
            net (Union[str, BayesNet]): Either file path of the bayesian network in BIFXML format or BayesNet object
        """
        if type(net) == str:
            # constructs a BN object
            self.bn = BayesNet()
            # Loads the BN from an BIFXML file
            self.bn.load_from_bifxml(net)
        else:
            self.bn = net

    def marginalize(self, variable: str, f1: pd.DataFrame) -> pd.DataFrame:
        """ Sums out a given variable, given the joint probability distribution with other variables.
        Args:
            variable (str): A string indicating the variable that needs to be summed-out
            f1 (pd.DataFrame): A factor containing the variable that needs to be summed-out

        Returns:
            pd.DataFrame: A conditional probability table with the specified variable summed-out
        """
        # get other variables 
        variables = f1.columns.tolist()
        variables = list(filter(lambda var: var != "p" and var != variable, variables))

        # Compute the new cpt and return it
        return f1.groupby(variables)["p"].sum()

    def ordering(self, heuristic: str):
        """Computes an ordering for the elimination of a given variable. Two heuristics can be chosen: min-fill and min-degree.

        Args:
            cpt (pd.DataFrame): _description_
            heuristic (str): _description_
        """
        print(self.bn.get_interaction_graph())

    def has_path(self, graph: nx.DiGraph, x: str, y: List[str], visited: List[str]) -> bool:
        """ Checks (recursively) if there is a path from x to any node in y in a given graph.
        Args:
            graph (nx.DiGraph): The graph to check.
            x (str): The starting node.
            y (List[str]): The target node(s).
        Returns:
            bool: True if there is a path from x to y, False otherwise.
        """
        # Cycle through all predecessors and successors of x
        for n in list(graph.neighbors(x)) + list(graph.predecessors(x)):
            if n in y:
                return True
            if n in visited:
                # If we already visited this node, then skip it
                continue
            else:
                # Otherwise add it to the visited list and check if there is a path from n to y
                visited.append(n)
                if self.has_path(graph, n, y, visited):
                    return True
        return False  

    def _prune(self, graph: nx.DiGraph, x: List[str], y: List[str], z: List[str])\
          -> Tuple[nx.reportviews.NodeView, nx.reportviews.OutEdgeView]:
          """ Applies the d-separation algorithm to a graph by pruning all leaf nodes not in x, y or z
              and removing all edges that are outgoing from z.
          Args:
              graph (nx.DiGraph): An acyclic directed graph representing the BN.
              x (List[str]): A list containing all nodes in x.
              y (List[str]): A list containing all nodes in y.
              z (List[str]): A list containing all nodes in z.
          Returns:
              Tuple[nx.reportviews.NodeView, nx.reportviews.OutEdgeView]: A tuple containing the nodes and edges of the graph prior to pruning.
          """
          prev_n, prev_e = deepcopy(graph.nodes), deepcopy(graph.edges)
          # Remove all leaf nodes that are not in x, y or z
          graph.remove_nodes_from([n for n in graph.nodes if n not in x + y + z and not any(True for _ in graph.neighbors(n))])
          # Remove all edges that are outgoing from z
          graph.remove_edges_from(list(graph.edges(z)))
          return list(prev_n), list(prev_e)

    def d_separated(self, x:  Union[str, List[str]], y:  Union[str, List[str]], z: Union[str, List[str]]) -> bool:
        """ Checks whether x is d-separated from y given z.

        Args:
            x (Union[str, List[str]]): The specified variable x or a list of variables.
            y (Union[str, List[str]]): The specified variable y or a list of variables.
            z (Union[str, List[str]]): The specified variable z or a list of variables.

        Returns:
            bool: True if x is d-separated from y given z, False otherwise.
        """
        # Check if z is a list or a single variable
        x, y, z = check_single(x), check_single(y), check_single(z)
        # Create a copy of the BN we can use for pruning
        graph = deepcopy(self.bn.structure)

        # Apply the d-separation algorithm exhaustively
        prev_nodes, prev_edges = self._prune(graph, x, y, z)
        while list(graph.nodes) != prev_nodes and list(graph.edges) != prev_edges:
            prev_nodes, prev_edges = self._prune(graph, x, y, z)
        
        # x is d-separated from y given z if there is no path from x to y in the pruned graph
        return not any(self.has_path(graph, u, y, [u]) for u in x)
    
    def independent(self, x:  Union[str, List[str]], y:  Union[str, List[str]], z: Union[str, List[str]]) -> bool:
        """ Checks whether x is independent from y given z.

        Args:
            x (Union[str, List[str]]): The specified variable x or a list of variables.
            y (Union[str, List[str]]): The specified variable y or a list of variables.
            z (Union[str, List[str]]): The specified variable z or a list of variables.

        Returns:
            bool: True if x is independent from y given z, False otherwise.
        """
        return self.d_separated(x, y, z)
    
    def f_multiply(self, f1: pd.DataFrame, f2: pd.DataFrame) -> pd.DataFrame:
        """ Multiplies two given factors together.

        Args:
            f1 (pd.DataFrame): The first specified factor.
            f2 (pd.DataFrame): The second specified factor.

        Returns:
            pd.DataFrame: The resulting factor from the product of f1 and f2.
        """
        def multiply(var: List, r1: pd.Series, r2: pd.Series) -> float:
            var.append(r1.drop("p").values.tolist())
            return r1["p"] * r2["p"]
        new_f1 = []
        new_f2 = [[x[i] for x in f2.drop(columns="p").values.tolist() * len(f1.drop(columns="p").values)]for i in range(len(f2.drop(columns="p").columns))]
        p = [multiply(new_f1, r1, r2) for _, r1 in f1.iterrows() for _, r2 in f2.iterrows()]
        return pd.DataFrame(new_f1, columns=f1.drop(columns="p").columns).assign(p=p, **dict(zip(f2.drop(columns="p"), new_f2)))

        

    def network_prune(self, query: Union[str, List[str]], evidence: Union[str, List[str]]):
        """ prunes the current network such that it can answer the given query

        Args:
            query Union[str, List[str]]: The query to be answered.
            evidence Union[str, List[str]]: The evidence, on which basis the query can be answered.

        Returns:
            BNReasoner: A new BNReasoner with a pruned graph and updated values.
        """
        graph = deepcopy(self)
        e = check_single(evidence)
        q = check_single(query)
        instance = {}
        for val in e: instance[val] = True
        #prune edges
        for node in e:
            graph.bn.structure.remove_edges_from([x for x in graph.bn.structure.edges if (x[0]==node and x[1] not in e+q)])
        for i in graph.bn.get_all_variables():
            graph.bn.update_cpt(i, graph.bn.reduce_factor(pd.Series(instance), self.bn.get_cpt(i)))
        nodeList = []
        #prune nodes
        for node in graph.bn.structure.nodes:
            if (node in e+q): continue
            if graph.bn.get_children(node) == []: nodeList.append(node)
        if nodeList != []: graph.bn.structure.remove_nodes_from(nodeList)
        return graph

    def marginal_distribution(self, query: Union[str, List[str]], evidence: Union[str, List[str]]):
        #Reduce factors wrt e
        #Compute joint marginal
        #Sum out q
        #return joint marginal divided by sum out q
        
        if evidence == []: return self.bn.get_cpt(query)
        else: print("too much evidence")
    

if __name__ == "__main__":
    bnr = BNReasoner("testing/lecture_example.BIFXML")
    cpts = bnr.bn.get_all_cpts()
    #print(cpts)
    #print(cpts['Rain?'])
    #print(cpts['Wet Grass?'])
    #print(bn.f_multiply(cpts['Winter?'], cpts['Wet Grass?']))
    g = bnr.network_prune('Wet Grass?', ['Winter?', 'Rain?'])
    #g.bn.draw_structure()
    #print(bnr.marginal_distribution('Rain?', []))
    #bnr.bn.draw_structure()

