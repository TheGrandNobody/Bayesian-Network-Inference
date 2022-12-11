# Script to run experiment (Task 2) Bayesian Networks
# MSc Artificial Intelligence VU - 2022
# Knowledge Representation

# imports
import os
from BayesNet import BayesNet
from BNReasoner import BNReasoner
import time
import csv


def run():
    # open file for results
    f = open('results/results.csv', 'w')
    writer = csv.writer(f)
    writer.writerow("variable elimination, naive summing out, node count, edge count")

    # loop through all files
    for file in os.listdir("/home/m_rosa/AI/KR/Bayesian-Network-Inference/testing"):
        bn = BNReasoner(file)
        node_count = len(bn.bn.get_all_variables())
        edge_count = None
        to_eliminate = []
        
        # perform variable elimination and measure runtime
        start = time.time()
        bn.elim_var(bn.ordering(to_eliminate))
        runtime_ve = time.time() - start 

        # perform naive summing out and measure run time
        runtime_ns = None
        
        # save results (run time) in csv
        writer.writerow(f"{runtime_ve}, {runtime_ns},{node_count}, {edge_count}")





if __name__== "__main__":
    pass







