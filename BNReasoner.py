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
        """ Checks if there is a path from x to y in a given graph.

        Args:
            graph (nx.DiGraph): The graph to check.
            x (str): The specified variable x.
            y (List[str]): The specified variable y.
            visited (List[str]): A list of visited nodes.

        Returns:
            bool: True if there is a path from x to y, False otherwise.
        """
        # Check if there is a path from x to y
        for n in list(graph.neighbors(x)) + list(graph.predecessors(x)):
            if n in y:
                return True
            if n in visited:
                continue
            else:
                visited.append(n)
                if self.has_path(graph, n, y, visited):
                    return True
        return False

    def d_sep(self, graph: nx.DiGraph, x: List[str], y: List[str], z: List[str]) \
        -> Tuple[nx.reportviews.NodeView, nx.reportviews.OutEdgeView]:
        prev_n, prev_e = deepcopy(graph.nodes), deepcopy(graph.edges)
        # Remove all leaf nodes that are not in x, y or z
        graph.remove_nodes_from([n for n in graph.nodes if n not in x + y + z and not any(True for _ in graph.neighbors(n))])
        # Remove all edges that are not in z
        graph.remove_edges_from(list(graph.edges(z)))
        return list(prev_n), list(prev_e)

    def d_separated(self, x:  Union[str, List[str]], y:  Union[str, List[str]], z: Union[str, List[str]]) -> bool:
        """ Checks whether x is d-separated from y given z.

        Args:
            x (Union[str, List[str]]): The specified variable x.
            y (Union[str, List[str]]): The specified variable y.
            z (Union[str, List[str]]): Either a single variable or a list of variables.

        Returns:
            bool: True if x is d-separated from y given z, False otherwise.
        """
        # Check if z is a list or a single variable
        x, y, z = check_single(x), check_single(y), check_single(z)
        # Create a copy of the BN we can use for pruning
        graph = deepcopy(self.bn.structure)

        # Apply the d-separation algorithm exhaustively
        prev_nodes, prev_edges = self.d_sep(graph, x, y, z)
        while list(graph.nodes) != prev_nodes and list(graph.edges) != prev_edges:
            prev_nodes, prev_edges = self.d_sep(graph, x, y, z)
        
        # Check if there is a path from x to y
        return not any(self.has_path(graph, u, y, [u]) for u in x)

if __name__ == "__main__":
    bn = BNReasoner("testing/lecture_example.BIFXML")
    print(bn.d_separated("Rain?", "Sprinkler?", "Winter?"))
    #bn.bn.draw_structure()