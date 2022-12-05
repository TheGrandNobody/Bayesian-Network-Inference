from typing import Union, List, Tuple
from BayesNet import BayesNet, nx
from copy import deepcopy

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
    
    def has_path(self, graph: nx.DiGraph, x: str, y: List[str], visited: List[str]) -> bool:
        """ Checks (recursively) if there is a path from x to y in a given graph.

        Args:
            graph (nx.DiGraph): The graph to check.
            x (str): The specified variable x.
            y (List[str]): The specified variable y.
            visited (List[str]): A list of visited nodes.

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

    def prune(self, query: Union[str, list[str]], evidence: Union[str, list[str]]):
        graph = deepcopy(self.bn.structure)
        # delete leaf nodes
        counter = 0
        nodeList = []
        edgeList = []
        for node in graph.nodes:
            if (node in query) or (node in evidence):
                continue
            for edge in graph.edges:
                if node == edge[1]:
                    counter += 1
            if counter == 0: nodeList.append(node)
            else: counter = 0
        
        for node in nodeList:
            graph.remove_node(node)

        #delete edges that follow from evidence
        for item in check_single(evidence):
            for edge in graph.edges:
                if item == edge[0]:
                    edgeList.append(edge)
        for edge in edgeList:
            graph.remove_edge(*edge)

        return graph


if __name__ == "__main__":
    bn = BNReasoner("testing/lecture_example.BIFXML")
    #print(bn.d_separated("Rain?", "Sprinkler?", "Winter?"))
    bn.bn.draw_structure()
    bn.prune("Rain?", "Winter?")
    bn.bn.draw_structure()
    