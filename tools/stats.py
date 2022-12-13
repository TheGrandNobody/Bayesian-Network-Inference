# implements statistics for 2 experiments

# imports
import os
import csv
import pandas as pd
import statistics as stats
from scipy.stats import iqr
from pingouin import ttest, wilcoxon, homoscedasticity, normality

# Some functions to return percentiles to use in descriptive statistics later

# min
def min(x):
    return x.quantile(0)

# 50th Percentile
def q25(x):
    return x.quantile(0.25)

# 90th Percentile
def q75(x):
    return x.quantile(0.75)

# max 
def max(x):
    return x.quantile(1)

def run_heuristics(file):
    # read results and put in dict
    
    data = pd.read_csv("../results/"+ file)

    # data = pd.DataFrame.from_dict(data)
    print(data)

    # give desctriptive stats parametric
    print("parametric\n", data.agg(['mean', 'std']).round(4))


    # give descriptive stats non-parametric
    print("non parametric data\n", data.agg(['median', min, q25, q75, max]))
    
    # test assumptions dependent t-test
    print(homoscedasticity(data.drop(["edge count", "node count"], axis = 1), method = "levene", alpha = 0.05))
    print(normality(data["min-fill"], method = "shapiro", alpha = 0.05))


    # perform dependent t-test/ wilcoxon
    print(ttest(data["min-fill"], data["min-degree"], paired = True, alternative = "two-sided", confidence = 0.95))
    print(wilcoxon(data["min-fill"], data["min-degree"]))

    # test assumptions logistic regression

    # perform logistic regression 
    





if __name__== "__main__":
    run_heuristics("results_heuristic.csv")