# Script used to run experiment (Task 2) Bayesian Networks
# MSc Artificial Intelligence VU - 2022
# Knowledge Representation
import os
from BNReasoner import BNReasoner, chain
import time
import csv

def run():
    # open file for results
    f = open('results/results.csv', 'w')
    writer = csv.writer(f)
    writer.writerow("variable elimination, naive summing out, node count, edge count")
    # loop through all files
    for file in os.listdir("/home/m_rosa/AI/KR/Bayesian-Network-Inference/test_cases"):
        bn = BNReasoner(file)
        node_count = len(bn.bn.get_all_variables()) # use the file name
        edge_count = None # use the file name
        to_eliminate = []
        
        # perform variable elimination and measure runtime
        start = time.time()
        bn.elim_var(bn.ordering(to_eliminate))
        runtime_ve = time.time() - start 

        # perform naive summing out and measure run time
        runtime_ns = None
        start = time.time()
        # Multiply all CPTs
        factors = list(bn.bn.get_all_cpts().values())
        s = [bn.bn.get_cpt(factors[0])]
        joint_pr = [[chain(s, i, factors[i + 1], bn.f_multiply) for i in range(len(factors[1:]))][-1]]
        # Sum out all variables
        [chain(joint_pr, i, to_eliminate[i], bn.marginalize) for i in range(len(to_eliminate))]
        runtime_ns = time.time() - start

        # save results (run time) in csv
        writer.writerow(f"{runtime_ve}, {runtime_ns},{node_count}, {edge_count}")

if __name__== "__main__":
    pass