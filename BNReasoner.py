from typing import Union, List
from BayesNet import BayesNet
from copy import deepcopy
import pandas as pd


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

    def marginalization(self, variable: str, cpt: pd.DataFrame):
        """Sums out a given variable, given the joint probability distribution with other variables.
        Args:
            variable (str): a string indicating the variable that needs to be summed-out
            cpt (pd.DataFrame): A cpt containing the variable that needs to be summed-out

        Returns:
            pd.DataFrame: cpt where variable is summed-out
        """

        # get other variables 
        variables = cpt.columns.tolist()
        variables = list(filter(lambda var: var != "p" and var != variable, variables))

        # make new cpt and return
        new_cpt = cpt.groupby(variables)["p"].sum()

        return new_cpt

    def ordering(self, heuristic: str):
        """Computes an ordering for the elimination of a given variable. Two heuristics can be chosen: min-fill and min-degree.

        Args:
            cpt (pd.DataFrame): _description_
            heuristic (str): _description_
        """
        print(self.bn.get_interaction_graph())
        
    def check_single(self, variable: Union[str, List[str]]) -> List[str]:
        """ Checks if the variable is a single variable and returns a list containing the variable if it is.

        Args:
            variable (Union[str, List[str]]): Either a single variable or a list of variables.

        Returns:
            List[str]: A list containing the variable, otherwise the variable list.
        """
        return variable if type(variable) == list else [variable]

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
        x, y, z = self.check_single(x), self.check_single(y), self.check_single(z)
        edges = [e for e in self.bn.structure.edges if e not in self.bn.structure.edges[z]]

        nodes = [n for n in self.bn.structure.nodes if n not in x + y + z and self.bn.structure.neighbors(n) == []]
        edges = [e for e in self.bn.structure.edges if e not in self.bn.structure.edges[z]]
            
      

if __name__ == "__main__":
    bn = BNReasoner("testing/lecture_example.BIFXML")
    bn.ordering("min-fill")

