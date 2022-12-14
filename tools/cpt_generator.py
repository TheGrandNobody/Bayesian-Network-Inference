import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element, ElementTree, SubElement
from typing import List
import random
import math

# The path to the folder in which to place the files
PATH = "test_cases/experiment/"
# The number of files to create
NUM_FILES = 15
# Whether to create test cases with an increasing number of edges
# The number of edges increase either accordingly with Pascal's triangle (n! / 2 * (n - 2)!) or linearly with the nodes
EDGES = True
# Whether to create test cases with an increasing number of nodes
# If only NODES is True, the number of edges will be 0
NODES = False
# The number of nodes to use for the test cases if NODES is False
NUM_NODES = 15
# Whether to increase the number of edges according to Pascal's triangle
PASCAL = False
# Whether to have a equal ratio of root nodes to normal nodes.
DIV = False

def generate_probability(values: List[int]) -> None:
    """ Generate two random probabilities adding up to 1 and add them to the list of probabilities.

    Args:
        values (List[int]): The list of values to add the probability to.
    """
    first = random.randint(1,99) / 100
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
    # Create a variable element
    variable = SubElement(network, "VARIABLE", {"TYPE": "nature"})
    # Give the variable a name
    variable_name = SubElement(variable, "NAME")
    variable_name.text = name
    # Create the outcomes (True/False) for the variable
    [new(variable, "OUTCOME", value) for value in ["True", "False"]]

    return name

def new_table(network: Element, prior: str, given: List[str]) -> None:
    """ Create a new CPT for a given prior and a list of evidence elements.

    Args:
        network (Element): The network element to add the CPT to.
        prior (str): The name of the prior variable.
        given (List[str]): The list of evidence variables.
    """
    # Create the definition element
    definition = SubElement(network, "DEFINITION")
    # Create the for row
    new(definition, "FOR", prior)
    # Create the given rows
    [new(definition, "GIVEN", g) for g in given]
    # Generate probabilities for each row
    values = []
    [generate_probability(values) for _ in range((2 ** (len(given) + 1) // 2))]
    new(definition, "TABLE", " ".join(values))

def new_file(path: str, node: int, edge: int, edges: bool, nodes: bool, pascal: bool, div: bool) -> None:
    """ Create a new BIFXML file for a given number of nodes and a given configuration.

    Args:
        path (str): The path of the folder to add the file to.
        node (int): The number of nodes present for this BN.
        edge (int): The number of edges present for this BN (if only edges increase).
        edges (bool): If this is an edge-only test case (or both).
        nodes (bool): If this is a node-only test case (or both).
        pascal (bool): If the number of edges should increase according to Pascal's triangle.
        div (bool): If there should be an equal ratio of root nodes to other nodes.
    """
    # Create the root element
    root = Element("BIF", {"VERSION": "0.3"})

    # Create the network element and add it to the root
    network = Element("NETWORK")
    root.append(network)

    # Create the name element
    name = SubElement(network, "NAME")
    name.text = f"Test{node}N{'T' if nodes else 'F'}N{'T' if edges else 'F'}E"

    # Create the variables
    vars = [new_variable(network, f"N{i}") for i in range(node)]
    # Create the tables
    [new_table(network, name, [] if nodes and not edges else \
      [var for var in vars[:i] if var not in name] if pascal\
      else [vars[i-1]] if i > len(vars) - edge + (1 if edge <= 1 else 0) else []) for i, name in enumerate(vars, start=0)]
    
    ElementTree(root).write(open(path, "wb"))

def create_files(n: int, path: str, edges: bool, nodes: bool, pascal: bool, div: bool) -> None:
    """ Create n BIFXML files and places them in a given path. The BIFXML testcases are generated using either increasing number of edges and/or nodes.

    Args:
        n (int): The number of files to create.
        path (str): The name of the folder in which to place the files.
        edges (bool): Whether to increase the number of edges over the files.
        nodes (bool): Whether to increase the number of nodes over the files.
        pascal (bool): Whether to increase the number of edges over the files using Pascal's triangle.
        div (bool): Whether to have an equal number of root nodes and other nodes.
    """
    [new_file(f"{path}{i if nodes else NUM_NODES}N{int(math.factorial(i) / (2 * math.factorial(i - 2))) if pascal else i if not div else i // 2  if edges else 0}E.BIFXML",\
         i if nodes else NUM_NODES, i, edges, nodes, pascal, div) for i in range(2, n + 1)]

if __name__ == "__main__":
    create_files(NUM_FILES, PATH, EDGES, NODES, PASCAL, DIV)