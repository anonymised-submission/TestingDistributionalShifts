import numpy as np
from multiprocessing import Pool, cpu_count
import statsmodels.api as sm
from tqdm import tqdm
from itertools import product
import pandas as pd

# Load files from parent folders
import os
import sys
try:sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
except NameError: print("Cannot load testing module")
from wrapper_resampler import ShiftedTester

# Help functions
def coin(n, p=0.5): return 1*(np.random.uniform(size=(n,1)) < p)
def cb(*args): return np.concatenate(args, axis=1)  # Col bind
def to_r(x): return robj.FloatVector(x)

np.random.seed(1)

# Simulate data from a Gaussian SCM
def scm(n, q=0.1):
    X1 = coin(n, q)
    return cb(X1)

q = 0.1
p = 0.9

# Weight
def weight(X):
    return (X*p/q + (1-X)*(1-p)/(1-q)).ravel()

# Test if proportion of 1 is larger than p - 0.1 (this is true in target dist, where proportion is p, but may fail in resample)
def T(X):
    return 1*(sm.stats.ztest(X, value=p-0.1, alternative="smaller")[1] < 0.05)[0]

# Define rates for resampling
def rate(pow, c=1):
    def f(n): return c*n**pow
    return f

# Loop parameters
pow_range = [a/20 for a in range(4, 20)]
n_range = [int(10**(x/2)) for x in range(4, 11)]
rep_range = [False, True]
combinations = list(product(n_range, pow_range, rep_range))

def conduct_experiment(i=None):
    out = []
    for n, pow, repl in combinations:
        X = scm(n, q)
        if rate(pow)(n) >= 3 and (rate(pow)(n) <= n):
            try:
                psi = ShiftedTester(weight, T, rate(pow), replacement=repl, reject_retries=5000, verbose=True)
                out.append(psi.test(X))

            except:
                # Catch errors from test statistic
                print(f"Error occurred {pow}, {n}")
                out.append(np.nan)
        else:
            # print(f"Sampling {pow}, {n}")
            out.append(np.nan)
    return out

## Conduct multiple experiments with multiprocessing and export to R for plotting:
if __name__ == '__main__':
    repeats = 200

    # Multiprocess
    pool = Pool(cpu_count()-2)
    res = np.array(
        list(tqdm(pool.imap_unordered(conduct_experiment, range(repeats)), total=repeats)))
    pool.close()

    # Count non-nas, to be used for binomial confidence intervals
    counts = (~np.isnan(res)).sum(axis=0)
    res = np.nansum(res, axis=0)

    # Pack as data frame
    df = pd.DataFrame(
        [(x/c, *v, c) for x, v, c in zip(res, combinations, counts)],
        columns=["RejectRate", "n", "Power", "SamplingScheme", "Count"])

    # Export to R for ggplotting
    df['RejectRate'] = df["RejectRate"].replace(np.NaN, "NA")
    df.to_csv("experiment-a2-binary.csv")
