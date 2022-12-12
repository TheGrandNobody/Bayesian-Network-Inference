from typing import Dict, Type, Union, List, Tuple, Callable
from BayesNet import BayesNet, nx
from copy import deepcopy
import pandas as pd

# Utility functions 
def check_single(variable: Union[str, List[str]]) -> List[str]:
    """ Checks if the variable is a single variable and returns a list containing the variable if it is.

    Args:
        variable (Union[str, List[str]]): Either a single variable or a list of variables.

    Returns:
        List[str]: A list containing the variable, otherwise the variable list.
    """
    return variable if type(variable) == list else [variable]

def chain(factors: List[Type[pd.DataFrame]], j: int, element: Union[str, Type[pd.DataFrame]], func: Callable[[Union[str, Type[pd.DataFrame]],Type[pd.DataFrame]], None]) -> None:
    """For a list of factors and either a variable or a factor, calls a given function and appends the result to a list.

    Args:
        factors (List[Type[pd.DataFrame]]): A list of factors.
        j (int): The index of the factor to be used as an argument for the function.
        element (Union[str, Type[pd.DataFrame]]): Either a variable or a factor.
        func (Callable[[Union[str, Type[pd.DataFrame]],Type[pd.DataFrame]], None]): A function that takes either a variable or a factor, and a factor as arguments and returns a factor.
    """
    factors.append(func(element, factors[j]))

# BNReasoner class
class BNReasoner:
    """ A class for performing inference on a given Bayesian Network."""
    def __init__(self, net: Union[str, Type[BayesNet]]) -> None:
        """ Initializes a BNReasoner object.
        
        Args:
            net (Union[str, Type[BayesNet]]): Either file path of the bayesian network in BIFXML format or BayesNet object
        """
        if type(net) == str:
            # Constructs a BN object
            self.bn = BayesNet()
            # Loads the BN from an BIFXML file
            self.bn.load_from_bifxml(net)
        else:
            self.bn = net

    def marginalize(self, variable: str, f1: Type[pd.DataFrame]) -> Type[pd.DataFrame]:
        """ Sums-out a given variable, given a factor containing it.
        Args:
            variable (str): A string indicating the variable that needs to be summed-out
            f1 (Type[pd.DataFrame]): A factor containing the variable that needs to be summed-out

        Returns:
            Type[pd.DataFrame]: A conditional probability table with the specified variable summed-out
        """
        return f1.groupby([c for c in f1.columns.tolist() if c not in ("p", variable)])["p"].sum().reset_index()
    
    def maximize(self, variable: str, f1: Type[pd.DataFrame]) -> Type[pd.DataFrame]:
        """ Maximizes-out a given variable, given a factor containing it.
        Args:
            variable (str): A string indicating the variable that needs to be maximized-out.
            f1 (Type[pd.DataFrame]): A factor containing the variable that needs to be maximized-out.

        Returns:
            Type[pd.DataFrame]: A conditional probability table with the specified variable maximized-out and their corresponding extended factors.
        """
        # Check if there is only one variable left in the table
        variables = [c for c in f1.columns.tolist() if c != "p" and "ext. factor" not in c]
        if len(variables) == 1:
            # If so, return the row with the maximum probability and the variable as an extended factor
            return f1.iloc[[f1['p'].idxmax()]].drop(columns=variable).assign(**{"ext. factor "+ variable: f1[variable]})

        # For tables with more than one variable, compute the CPT with the maximum probability for the given variable
        new = f1.groupby([c for c in variables if c != variable])["p"].max().reset_index()
        # Find any previous extended factors present in the table
        prev_factors = [c for c in f1.columns.tolist() if "ext. factor" in c]
        # Compute the new extended factor for the new CPT and add it to the table
        ext_factor = pd.merge(f1, new, on=['p'], how='inner').rename(columns={variable: "ext. factor " + variable})[f"ext. factor {variable}"]
        return new.assign(**dict(f1[prev_factors]), **{f"ext. factor {variable}": ext_factor}) if prev_factors else new.assign(**{f"ext. factor {variable}": ext_factor})

    def new_edges(self, neighbours: List[List[str]], graph: Type[nx.DiGraph]) -> List[List[str]]:
        """ Returns a list of new edges that arise by removing a given variable in the graph.

        Args:
            neighbours (List[List[str]]): List of neighbours (both neighbours and predecessors)
            graph (Type[nx.DiGraph]): A directed graph representing the Bayesian network

        Returns:
            List[List[str]]: A list of new edges that arise by removing a given variable in the graph
        """
        return [(var[1], var2[1]) for var, i in zip(neighbours, range(0, len(neighbours)-1)) for var2 in neighbours[i+1:] if not any(var2[1] in sublist for sublist in nx.edges(graph, var[1]))]
        
    def ordering(self, heuristic: str, to_eliminate: List[str]) -> List[str]:
        """Computes an ordering for the elimination of a list of variables. 
        Two heuristics can be chosen to decide the order of the list: min-fill and min-degree.

        Args:
            heuristic (str): "f" for min-fill heuristic, "d" for min-degree heuristic.
            to_eliminate (List[str]): List of variables to eliminate.

        Returns:
            list: List of variables to eliminate, with ordering decided by min-fill(f) or min-edge(e) heuristic. 
        """
        # List for new order and get interaction graph current BN
        new_order = []
        g =  self.bn.get_interaction_graph()
        # Draw interaction graph and save it
        positions = nx.spring_layout(g)
        nx.draw(g, positions, with_labels = True)
        
        # Create dict with variables (key) and a list of corresponding new edges(when variable is removed)
        new_edges = {var: self.new_edges(list(nx.edges(g, var)), g) for var in to_eliminate}

        # Find the best variable to eliminate until all variables are eliminated
        while len(to_eliminate) > 0:
            # Pick a heuristic
            if heuristic == "d":
                # Fetch the variable with the least amount of edges
                all_edges = {key: list(nx.edges(g, key)) for key in to_eliminate}
                next = min(all_edges.items(), key = lambda x: len(x[1]))[0]
            elif heuristic == "f":
                # Fetch the variable whose deletion would add the fewest new interactions
                next = min(new_edges.items(), key = lambda x: len(x[1]))[0]
            # Add the variable to the ordering
            to_eliminate.remove(next), new_order.append(next)
            # Find edges that need to be connected after removing the node
            [g.add_edge(edge[0], edge[1]) for edge in new_edges[next]]
            # Update new_edges, remove any edges that are no longer valid and remove the node
            [new_edges[var].remove(new_edge) for var, edges in deepcopy(new_edges).items() for new_edge in edges if next in new_edge]
            del new_edges[next]
            g.remove_node(next)       
        return new_order

    def elim_var(self, variables: List[str]) -> Type[pd.DataFrame]:
        """ Applies variable elimination to a BN for a given list of variables to eliminate.

        Args:
            variables (List[str]): An (ordered) list of variables to eliminate from a BN.. 

        Returns:
            prior (pd.Dataframe): The resulting prior after applying variable elimination.
        """
        cpt_tables = self.bn.get_all_cpts()
        # before: choose variable from list
        # after: Take the first variable in the list, apply the chain rule, and marginalize the variable
        for i, var in enumerate(variables):
            # before: multiply all factors (f) containing the variable
            # after: Fetch all factors containing the current variable and multiply them together
            factors = [k for k, v in cpt_tables.items() if var in v.columns]
            s = [cpt_tables[factors[0]]]
            if len(factors) > 1:
                [chain(s, j - 1, cpt_tables[factors[j]], self.f_multiply) for j in range(1, len(factors))]
            elif len(factors) != 1:
                continue
            # before: marginalize out the variable to obtain a new factor 
            # after: Eliminate the variable from the factor
            factor = self.marginalize(var, s[-1])
            # Remove old factors from cpts (so that the algorithm knows that they are summed out)
            cpt_tables = {k: v for k, v in cpt_tables.items() if k not in factors}
            # Add the new factor as a cpt
            cpt_tables[str(factor.drop("p", axis=1).columns.to_list()) + str(i)] = factor.assign(p=factor["p"]) 

        # FIRST
        # for element, i in zip(cpt_tables.keys(), range(0, len(cpt_tables)))
        # vs
        # for i, element in enumerate(cpt_tables.keys()) <-- this is better
        # THEN
        #for i, key in enumerate(cpt_tables.keys()):
        #    if i == 0:
        #        new_factor = cpt_tables[key]
        #    else:
        #        new_factor = self.f_multiply(new_factor, cpt_tables[key])
        # VS
        # p = [cpt_tables[list(cpt_tables.keys())[0]]]
        # [chain(p, j, list(cpt_tables.keys())) for j in range(1, len(list(cpt_tables.keys())))]

        # Multiply all remaining cpts to get the final factor and return it
        s = [cpt_tables[list(cpt_tables.keys())[0]]]
        [chain(s, j, cpt_tables[key], self.f_multiply) for j, key in enumerate(list(cpt_tables.keys())[1:])]
        
        return s[-1]
        
    def has_path(self, graph: Type[nx.DiGraph], x: str, y: List[str], visited: List[str]) -> bool:
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

    def _prune(self, graph: Type[nx.DiGraph], x: List[str], y: List[str], z: List[str])\
          -> Tuple[Type[nx.reportviews.NodeView], Type[nx.reportviews.OutEdgeView]]:
          """ Applies the d-separation algorithm to a graph by pruning all leaf nodes not in x, y or z
              and removing all edges that are outgoing from z.
          Args:
              graph (Type[nx.DiGraph]): An acyclic directed graph representing the BN.
              x (List[str]): A list containing all nodes in x.
              y (List[str]): A list containing all nodes in y.
              z (List[str]): A list containing all nodes in z.
          Returns:
              Tuple[Type[nx.reportviews.NodeView], Type[nx.reportviews.OutEdgeView]]: A tuple containing the nodes and edges of the graph prior to pruning.
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
    
    def f_multiply(self, f1: Type[pd.DataFrame], f2: Type[pd.DataFrame]) -> Type[pd.DataFrame]:
        """ Multiplies two given factors together.

        Args:
            f1 (Type[pd.DataFrame]): The first specified factor.
            f2 (Type[pd.DataFrame]): The second specified factor.

        Returns:
            Type[pd.DataFrame]: The resulting factor from the product of f1 and f2.
        """
        f1, f2 = (f2, f1) if len(f1) < len(f2) else (f1, f2)
        shared = list(set(f1.columns) & set(f2.columns))
        p = [r1.drop("p").values.tolist() + r2.drop(shared).values.tolist() + [r1["p"] * r2["p"]]\
           for _, r1 in f1.iterrows() for _, r2 in f2.iterrows() if all(r1[var] == r2[var] for var in list(set(shared) - set(["p"])))]
        return pd.DataFrame(p, columns=f1.drop("p", axis=1).columns.to_list() + f2.drop(shared, axis=1).columns.to_list() + ["p"])

    def network_prune(self, query: Union[str, List[str]], evidence: Dict[str, bool]) -> Type[BayesNet]:
        """ Prunes the current network such that it can answer the given query

        Args:
            query Union[str, List[str]]: The query to be answered.
            evidence Dict[str, bool]): The specified evidence.

        Returns:
            Type[BayesNet]: A new BN with a pruned graph and updated values.
        """
        graph = deepcopy(self.bn)
        e, q = list(evidence.keys()), check_single(query)
        # Prune edges
        [graph.structure.remove_edges_from([x for x in graph.structure.edges if (x[0]==node and x[1] not in e + q)]) for node in e]
        [graph.update_cpt(i, graph.reduce_factor(pd.Series(evidence), self.bn.get_cpt(i))) for i in graph.get_all_variables()]
        # Prune nodes
        nodeList = [node for node in graph.structure.nodes if not (node in e + q or graph.get_children(node))]
        if nodeList: graph.structure.remove_nodes_from(nodeList)
        return graph

    def marginal_distribution(self, query: Union[str, List[str]], evidence: Union[str, List[str]]) -> float:
        """ Provides a marginal distribution given a query and an evidence

        Args:
            query: Union[str, List[str]]: The specified query.
            evidence: Union[str, List[str]]: The specified evidence.

        Returns:
            float: The probability of the result
        """
        #Reduce factors wrt e
        #Compute joint marginal
        #Sum out q
        #return joint marginal divided by sum out q
        q = check_single(query)
        newR = deepcopy(self)
        qReasoner = newR.network_prune(query, evidence)
        print([x for x in qReasoner.get_all_variables() if x not in q])
        a = newR.elim_var([x for x in qReasoner.get_all_variables() if x not in q])
        aVal = a.at[len(a)-1,'p']
        print(aVal)
        if len(q) == 1:
            b = newR.marginalize(query, qReasoner.get_cpt(query))
            bVal = b.at[len(b)-1,'p']
        #else:
        return aVal/bVal
    
    def m_a_p(self, query: Union[str, List[str]], evidence: Dict[str, bool]) -> Type[pd.DataFrame]:
        """ Provides the maximum a posteriori probability given a query and an evidence

        Args:
            query: Union[str, List[str]]: The specified query.
            evidence: Union[str, List[str]]:

        Returns:
            Type[pd.DataFrame]: The probability of the result
        """
        query = check_single(query)
        # Prune the network
        bn = BNReasoner(deepcopy(self).network_prune(query, evidence))
        # Eliminate all variables except the query
        pr_query = [bn.elim_var(bn.ordering("f", [x for x in bn.bn.get_all_variables() if x not in query]))]
        # Max-out the query variables
        [chain(pr_query, i, query[i], self.maximize) for i in range(len(query))]
        return pr_query[-1]
    
    def m_e_p(self, evidence: Dict[str, bool]) -> Type[pd.DataFrame]:
        """ Provides the maximum expected probability given some evidence

        Args:
            evidence: Union[str, List[str]]:

        Returns:
            Type[pd.DataFrame]: The probability of the result
        """
        return self.m_a_p([variable for variable in self.bn.get_all_variables() if variable not in evidence], evidence).\
          assign(**{f"ext. factor {k}":v for k, v in evidence.items()})

if __name__ == "__main__":
    bn = BNReasoner("testing/earthquake.BIFXML")
    bn.bn.draw_structure()
