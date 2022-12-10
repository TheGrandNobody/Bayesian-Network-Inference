import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element, ElementTree, SubElement
from typing import List
import random

def generate_probability(values: List[int]) -> None:
    """ Generate two random probabilities adding up to 1 and add them to the list of probabilities.

    Args:
        values (List[int]): The list of values to add the probability to.
    """
    first = float(random.randint(1,99)) / 100
    values.append(str(first))
    values.append(str(1 - first))

def new(var: Element, name: str, text: str) -> None:
        """ Generates a given sub element.

        Args:
            var (Element): The element to add the new sub element to.
            text (str): The text of the sub element.
        """
        variable = SubElement(var, name)
        variable.text = text

def new_variable(network: Element, name: str) -> str:
    """ Create a new variable element and add it to the network element.

    Args:
        network (Element): The network element to add the variable to.
        name (str): The name of the variable.
    """

    variable = SubElement(network, "VARIABLE", {"TYPE": "nature"})
    variable_name = SubElement(variable, "NAME")
    variable_name.text = name
    [new(variable, "OUTCOME", value) for value in ["True", "False"]]

    return name

def new_table(network: Element, prior: str, given: List[str]) -> None:
    definition = SubElement(network, "DEFINITION")
    new(definition, "FOR", prior)
    [new(definition, "GIVEN", g) for g in given]
    values = []
    [generate_probability(values) for _ in range(len(given) + 1)]
    new(definition, "TABLE", " ".join(values))

def new_file(path: str, node: int, edges: bool, nodes: bool) -> None:
    # Create the root element
    root = Element("BIF", {"VERSION": "0.3"})

    # Create the network element and add it to the root
    network = Element("NETWORK")
    root.append(network)

    # Create the name element
    name = SubElement(network, "NAME")
    name.text = path

    # Create the variables
    vars = [new_variable(network, f"N{i}") for i in range(node)]

    # Create the tables
    [new_table(network, name, [] if nodes and not edges else \
      (vars[:i] if name not in vars[:i] else vars[:i] - [name])) for i, name in enumerate(vars, start=1)]
    
    ElementTree(root).write(open(path, "wb"))

def create_files(n: int, path: str, edges: bool, nodes: bool) -> None:
    """ Create n BIFXML files and places them in a given path. The BIFXML testcases are generated using either increasing number of edges and/or nodes.

    Args:
        n (int): The number of files to create.
        path (str): The name of the folder in which to place the files.
        edges (bool): Whether to increase the number of edges over the files.
        nodes (bool): Whether to increase the number of nodes over the files.
    """
    [new_file(f"{path}/testcase{i}.BIFXML", i if nodes else n, edges, nodes) for i in range(2, n + 1)]

if __name__ == "__main__":
    create_files(10, "test_cases", False, True)