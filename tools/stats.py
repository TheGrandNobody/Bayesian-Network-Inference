# implements statistics for 2 experiments

# imports
import pandas as pd
import statistics as stats
from scipy.stats import iqr
from pingouin import ttest, wilcoxon, homoscedasticity, normality
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib
import numpy as np



exp_1 = "min-fill"
exp_2 = "min-degree"

# exp_1 = "variable elimination"
# exp_2 = "naive summing out"

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

def testing(file):

    # read results and put in pandas dataframe
    data = pd.read_csv("../results/"+ file)

    # data = pd.DataFrame.from_dict(data)
    print(data)

    # give desctriptive stats parametric
    print("parametric\n", data.agg(['mean', 'std', "min", "max"]).round(4))


    # give descriptive stats non-parametric
    print("non parametric data\n", data.agg(['median', min, q25, q75, max]))
    
    # test assumptions dependent t-test
    print(homoscedasticity(data.drop(["edge count", "node count"], axis = 1), method = "levene", alpha = 0.05))
    print(normality(data[exp_1], method = "shapiro", alpha = 0.05))
    print(normality(data[exp_2], method = "shapiro", alpha = 0.05))


    # perform dependent t-test/ wilcoxon
    print(ttest(data[exp_1], data[exp_2], paired = True, alternative = "two-sided", confidence = 0.95))
    print(wilcoxon(data[exp_1], data[exp_2]))

    # boxplot
    plt.subplots(figsize = ( 6 , 4 ))
    sns.set(font_scale=1)
    plot = sns.boxplot(data=data.drop(["edge count", "node count"], axis = 1), saturation=0.8, showfliers = True, palette = "muted", orient="v")
    plot.set(yscale = "log")
    plt.ylabel('runtime (s)')
    fig = plot.get_figure()
    fig.savefig("boxplot.png")

################################################################################################################################################### 

def regression (file, dependent, independent, transform):
    """_summary_

    Args:
        file (csv): give file with data from experiment.py
        dependent(str): "min-fill" or "min-degree"
        independent (str): independent variables ("node count or edge count")
    """
    # read data  and put in pandas dataframe
    data = pd.read_csv("../results/"+ file)

    # # perform linear regression 

    # remove where edges are 0 (for regression with amount of edges) or only keep where edges are 0 (for regression with amount of nodes)
    if independent == ["edge count"]:
        data = data[data["node count"] == 15]
    elif independent == "node count":
        data = data[data[independent] == 0]
    
    print(data)

    if transform == True:
        # transform data (only for regression with amount of edges)
        data[dependent] = np.log(data[dependent])

    # defining the dependent and independent variables
    xtrain = data[independent]
    xtrain = sm.add_constant(xtrain)
    ytrain = data[dependent]

    print(xtrain)
    print(ytrain)
    # regression
    log_reg = sm.OLS(ytrain, xtrain).fit()
    print(log_reg.summary())
    
    # graph model (predictions versus actual data)

    # what to plot
    predicted_counts = data["edge count"]
    actual_counts = ytrain

    plt.clf()
    fig = plt.figure()
    fig.set_size_inches(15,7)
    fig.suptitle(f"The effect of the number of edges on the runtime of variable elimination ({dependent})", size = 20)
    plt.scatter(x=predicted_counts, y=actual_counts, marker='.')
    plt.xlabel('Amount of edges', size = 15)
    plt.ylabel('runtime variable elimination (s)', size = 15)
    plt.savefig("scatter.png")


if __name__== "__main__":
    # testing("exp2_v.csv")
    regression("exp2_v_2.csv", "min-fill", ["edge count"], transform = False)
