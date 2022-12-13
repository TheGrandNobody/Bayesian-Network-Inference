# Bayesian-Network-Inference
This repository introduces a set of tools which can be used for inference with Bayesian Networks (BN). The BayesNet class allows one to represent BNs using an acyclic directed graph, while the BNReasoner class allows one to analyze BNs with ease by computing factors through the application of a list of various inference techniques. 

Additionally, there is a conditional probability table (CPT) generator available for creating a number of CPTs with specific number of nodes and/or edges.
## List of inference techniques
* Network Pruning
  * **function:** network_prune()
* d-Separation/Conditional Independence
  * **function:** d_separated(), independent()
* Marginalization
  * **function:** marginalize()
* Maximization
  * **function:** maximize()
* Factor multiplication
  * **function:** f_multiply()
* Ordering
  * **function:** ordering()
* Variable Elimination
  * **function:** elim_var()
* Marginal Distribution Computation
  * **function:** marginal_distribution()
* Maximum A-posteriori (MAP) Computation
  * **function:** m_a_p()
* Most Probable Explanation (MEP) Computation
  * **function:** m_e_p()
## User Instructions
To load a BN, one must first load a .BIFXML file using a BNReasoner object. Then, the BN can be visualized using the BayesNet object of the BNReasoner class, and any of the above inference techniques can be used by calling their corresponding functions. An example is given in the *unit_tests* folder, which you can run using:
```
python3 unit_test/unit.py
```
To generate CPTs, you must first edit the path, number of files and edge/node parameters in *cpt_generator.py*. Setting only the edge parameter to True will keep a constant number of nodes equal to the NUM_NODES parameter and gradually increase the number. Setting only the nodes parameter to True will create CPTs where all nodes are independent, but gradually increase per file. Setting both parameters to True will increase both nodes and edges gradually. 

For any given number nodes __*n*__, the number of edges increases accordingly with the inner numbers of Pascal's triangle, determined by the formula:

$$
\begin{align}
  \text{Number of edges}\ =\ (n! / 2 * (n - 2)!)
\end{align}
$$