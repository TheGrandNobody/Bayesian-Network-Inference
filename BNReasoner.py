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
            # Constructs a BN object
            self.bn = BayesNet()
            # Loads the BN from an BIFXML file
            self.bn.load_from_bifxml(net)
        else:
            self.bn = net

    def marginalize(self, variable: str, f1: pd.DataFrame) -> pd.DataFrame:
        """ Sums-out a given variable, given a factor containing it.
        Args:
            variable (str): A string indicating the variable that needs to be summed-out
            f1 (pd.DataFrame): A factor containing the variable that needs to be summed-out

        Returns:
            pd.DataFrame: A conditional probability table with the specified variable summed-out
        """
        return f1.groupby([c for c in f1.columns.tolist() if c not in ("p", variable)])["p"].sum().reset_index()
    
    def maximize(self, variable: str, f1: pd.DataFrame) -> pd.DataFrame:
        """ Maximizes-out a given variable, given a factor containing it.
        Args:
            variable (str): A string indicating the variable that needs to be maximized-out.
            f1 (pd.DataFrame): A factor containing the variable that needs to be maximized-out.

        Returns:
            pd.DataFrame: A conditional probability table with the specified variable maximized-out and their corresponding extended factors.
        """
        # Compute the CPT with the maximum probability for the given variable
        new = f1.groupby([c for c in f1.columns.tolist() if c not in ("p", variable) and "ext. factor" not in c])["p"].max().reset_index()
            # Find any previous extended factors present in the table
        prev_factors = [c for c in f1.columns.tolist() if "ext. factor" in c]
        # Compute the new extended factor for the new CPT and add it to the table
        ext_factor = pd.merge(f1, new, on=['p'], how='inner').rename(columns={variable: "ext. factor " + variable})[f"ext. factor {variable}"]
        return new.assign(**dict(f1[prev_factors]), **{f"ext. factor {variable}": ext_factor}) if prev_factors else new.assign(**{f"ext. factor {variable}": ext_factor})

    def new_edges(self, neighbours: List[List[str]], graph: nx.DiGraph) -> List[List[str]]:
        """Returns a list of new edges that arise by removing a given variable in the graph.

        Args:
            variable (str): The variable that is removed from the graph
            neighbours (List[List[str]]): List of neighbours (both neighbours and predecessors)
            graph (nx.DiGraph): A directed graph representing the Bayesian network

        Returns:
            List[List[str]]: A list of new edges that arise by removing a given variable in the graph
        """
        return [(var[1], var2[1]) for var, i in zip(neighbours, range(0, len(neighbours)-1)) for var2 in neighbours[i+1:] if not any(var2[1] in sublist for sublist in nx.edges(graph, var[1]))]
        

    def ordering(self, heuristic: str, to_eliminate: List[str]):
        """Computes an ordering for the elimination of a list of variables. Two heuristics can be chosen to decide the order of the list: min-fill and min-degree.

        Args:
            heuristic (str): "f" for min-fill heuristic, "e" for min-edge heuristic.
            to_eliminate (List[str]): List of variables to eliminate.

        Returns:
            list: List of variables to eliminate, with ordering decided by min-fill(f) or min-edge(e) heuristic. 
        """

        # list for new order and get interaction graph current BN
        new_order = []
        G =  self.bn.get_interaction_graph()
        # Draw interaction graph and save
        positions = nx.spring_layout(G)
        nx.draw(G, positions, with_labels = True)
        
        # create dict with variables (key) and a list of corresponding new edges(when variable is removed)(value)
        new_edges = {var: self.new_edges(var, list(nx.edges(G, var)), G) for var in to_eliminate}

        # Find best variable to eliminate until all variables were eliminated
        while len(to_eliminate) > 0:
            # choose heuristic
            if heuristic == "e":
                # make dict with [variable] -> (all edges), and get variable with least amount of edges from dict
                all_edges = {key: list(nx.edges(G, key)) for key in to_eliminate}
                next = min(all_edges.items(), key = lambda x: len(x[1]))[0]
            elif heuristic == "f":
                # get variable whose deletion would add the fewest new interactions to the ordering
                next = min(new_edges.items(), key = lambda x: len(x[1]))[0]

            # remove variable from to_eliminate, add to new_order
            to_eliminate.remove(next), new_order.append(next)

            # find edges that need to be connected after removing node and connect them
            [G.add_edge(edge[0], edge[1]) for edge in new_edges[next]]
            
            # update new_edges, remove all new edges that contain "next" in them, remove all new edges for variable "next"
            [new_edges[var].remove(new_edge) for var, edges in deepcopy(new_edges).items() for new_edge in edges if next in new_edge]
            del new_edges[next]

            # remove node from graph
            G.remove_node(next)       
        return new_order

    def elim_var(self, variables, bn):
        pass

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
        new_f2 = [[x[i] for x in f2.drop(columns="p").values.tolist() * len(f1.drop(columns="p").values)] for i in range(len(f2.drop(columns="p").columns))]
        p = [multiply(new_f1, r1, r2) for _, r1 in f1.iterrows() for _, r2 in f2.iterrows()]
        return pd.DataFrame(new_f1, columns=f1.drop(columns="p").columns).assign(**dict(zip(f2.drop(columns="p"), new_f2), p=p))

    def network_prune(self, query: Union[str, List[str]], evidence: Union[str, List[str]]):
        graph = deepcopy(self.bn.structure)
        e = check_single(evidence)
        # Prune edges
        for node in e:
            graph.remove_edges_from([e for e in graph.edges if e[0]==node])
            # Apply reduced factor
        
        nodeList = []
        # Prune nodes
        for node in graph.nodes:
            if (node in check_single(query)) or (node in e): continue
            counter = 0
            for edge in graph.edges:
                if node == edge[1]: counter += 1
            if counter == 0: nodeList.append(node)
        if nodeList != []: graph.remove_nodes_from(nodeList)
        return graph

if __name__ == "__main__":
    bn = BNReasoner("testing/lecture_example.BIFXML")
    print(bn.bn.get_cpt("Wet Grass?"))
    print(bn.bn.get_cpt("Sprinkler?"))
    print(bn.f_multiply(bn.bn.get_cpt("Wet Grass?"), bn.bn.get_cpt("Sprinkler?")))
