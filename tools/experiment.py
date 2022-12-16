# Script used to run experiment (Task 2) Bayesian Networks
# MSc Artificial Intelligence VU - 2022
# Knowledge Representation
import os
from BNReasoner import BNReasoner, chain
import time
import csv
import re

def run(exp_type: int)->None:
    """Measures computing time for two experiments. The first experiment measures for naive summing out versus variable elimination.
    The second measures two heuristics for ordering: min-fill and min-degree. 
    For the first experiment the heuristic is set on min-fill by default.
    By default, all variables but one are removed during variable elimination for any bayesian network.
    Saves results in csv.
    Args:
        exp_type (int): 1 for variable elimination versus naive summing out, 2 for min-fill versus min-degree. 
        If anything else given then 1, min-fill versus min-degree.
    """
    # open file for results
    f = open('../results/exp2_v.csv', 'w')

    writer = csv.writer(f)
    if exp_type == 1:
        writer.writerow(["variable elimination", "naive summing out", "node count", "edge count"])
    else:
        writer.writerow(["min-fill", "min-degree", "node count", "edge count"])

    # loop through all files
    for file in os.listdir("../test_cases/experiment/"):
        print(file)
        bn = BNReasoner("../test_cases/experiment/" + file)
        counts = re.split("[E N]", file)
        print(counts)
        node_count = counts[0]
        edge_count = counts[1]
        to_eliminate = bn.bn.get_all_variables()[:-1]

        if exp_type == 1:
            # Perform variable elimination and measure runtime
            start = time.time()
            bn.elim_var(bn.ordering("f", to_eliminate))
            runtime_1 = time.time() - start 
            
            # Perform naive summing out and measure run time
            start = time.time()
            # Multiply all CPTs
            factors = list(bn.bn.get_all_cpts().keys())
            s = [bn.bn.get_cpt(factors[0])]
            joint_pr = [[chain(s, i, bn.bn.get_cpt(factors[i + 1]), bn.f_multiply) for i in range(len(factors[1:]))][-1]]
            # Sum out all variables
            [chain(joint_pr, i, to_eliminate[i], bn.marginalize) for i in range(len(to_eliminate))]
            runtime_2 = time.time() - start
        elif exp_type == 2:
            # measure time for min-fill
            ordering = bn.ordering("f", to_eliminate)
            start = time.time()
            bn.elim_var(ordering)
            runtime_1 = time.time() - start 
            #measure time for min-degree
            ordering = bn.ordering("d", to_eliminate)
            start = time.time()
            bn.elim_var(ordering)
            runtime_2 = time.time() - start 
        else:
            start = time.time()
            ordering = bn.ordering("f", to_eliminate)
            runtime_1 = time.time() - start 

            start = time.time()
            ordering = bn.ordering("d", to_eliminate)
            runtime_2 = time.time() - start 


        # Save results (run time) in csv
        writer.writerow([runtime_1, runtime_2,node_count, edge_count])

if __name__== "__main__":
    run(2)