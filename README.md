# Bayesian-Network-Inference
This repository introduces a set of tools which can be used for inference with Bayesian Networks (BN). The BayesNet class allows one to represent BNs using an acyclic directed graph, while the BNReasoner class allows one to analyze BNs with ease by computing factors through the application of a list of various inference techniques. 

Additionally, there is a conditional probability table (CPT) generator available for creating a number of CPTs with specific number of nodes and/or edges.
## List of inference techniques
* Network Pruning
* d-Separation/Conditional Independence
* Marginalization
* Maximization
* Factor multiplication
* Ordering
* Maximum A-posteriori (MAP) Computation
* Most Probable Explanation (MEP) Computation
## User Instructions
To load a BN, one must load a .BIFXML file using a BNReasoner object an example is given in the *unit_tests* folder, which you can run using:
```
python3 unit_test/unit.py
```
To generate CPTs, you must first edit the path, number of files and edge/node parameters in *cpt_generator.py*. Setting only the edge parameter to True will keep a constant number of nodes equal to the NUM_NODES parameter and gradually increase the number. Setting only the nodes parameter to True will create CPTs where all nodes are independent, but gradually increase per file. Setting both parameters to True will increase both nodes and edges gradually. 

The number of edges increase accordingly with the inner numbers of Pascal's triangle, determined by the formula:
$$
\begin{align}
  \text{Number of edges}\ =\ (n! / 2 * (n - 2)!)
\end{align}
$$
Where n is the number of nodes.