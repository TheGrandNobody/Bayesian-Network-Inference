# Script used to run experiment (Task 2) Bayesian Networks
# MSc Artificial Intelligence VU - 2022
# Knowledge Representation
import os
from BayesNet import BayesNet
from BNReasoner import BNReasoner
import time
import csv
from typing import List
import pandas as pd

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
        def chain(s: List[pd.DataFrame], i: int, factors: List[pd.DataFrame], func: function):
            s.append(func(factors[i], s[i - 1]))
        start = time.time()
        # Multiply all CPTs
        factors = list(bn.bn.get_all_cpts().values())
        s = [bn.bn.get_cpt(factors[0])]
        joint = [chain(s, i, factors, bn.f_multiply) for i in range(1, len(factors))][-1]
        # Sum out all variables
        factors = bn.bn.get_all_variables()
        s = [bn.marginalize(factors[0], joint)]
        [chain(s, i, factors, bn.marginalize) for i in range(1, len(factors))]
        runtime_ns = time.time() - start

        # save results (run time) in csv
        writer.writerow(f"{runtime_ve}, {runtime_ns},{node_count}, {edge_count}")

if __name__== "__main__":
    pass







