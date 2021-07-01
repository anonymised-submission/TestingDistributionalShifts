import numpy as np
from multiprocessing import Pool, cpu_count
import statsmodels.api as sm
from scipy.stats import norm
from tqdm import tqdm
from itertools import product
import pandas as pd
from statsmodels.stats.proportion import proportion_confint

import os
import sys

try: sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
except NameError: print("Cannot load testing module")
from wrapper_resampler import ShiftedTester

np.random.seed(1)

# Help functions
def e(n=1): return np.random.normal(size=(n, 1))  # Gaussian noise
def cb(*args): return np.concatenate(args, axis=1)  # Col bind
p = norm.pdf

# Simulate data from a Gaussian SCM
def scm(n, causal_effect=0):
    H = e(n)
    X1 = e(n)
    X2 = X1 + H + e(n)
    X3 = X2 + 2*e(n)
    X4 = causal_effect*X1 + X3 + H + e(n)
    return cb(H, X1, X2, X3, X4)

def weight(X):
    num = p(X[:, 3], loc=X[:,3].mean(), scale=1.0)
    denom = p(X[:, 3], loc=X[:,2], scale=2)
    return num/denom

# Test function: Regress X4 ~ X1 and get p-value
def T(X): return (sm.OLS(X[:, 4], sm.tools.add_constant(X[:, 1])).fit().pvalues[1] < 0.05)*1

# Parameters for choice-of-m algorithm
tune_m_repeats = 25
cutoff = np.quantile(np.random.uniform(size=(1000, tune_m_repeats)).min(axis=1), 0.05)


# Loop parameters
causal_effects = [0, 0.3]#np.linspace(0, 5, num=21)
n_range = [int(10**(x/2)) for x in range(4, 10)]
tests = {"LinReg": T}
m_choices = ["heuristic", "sqrt"]

combinations = list(product(n_range, causal_effects, m_choices))

## Wrap as function to apply multiprocessing
def conduct_experiment(i=None):
    out = []
    for n, c_e, m_choice in combinations:
        X = scm(n, causal_effect=c_e)
        # Do not do anything if m < 5 or (m>n and not replacement)
        try:
            psi = ShiftedTester(weight, tests["LinReg"], rate=None, replacement="NO-REPL-reject", reject_retries=100)
            m = psi.tune_m(X, j_x = [3], j_y=[2], gaussian=True, cond = [X[:,3].mean()], m_factor=1.3,
                           p_cutoff=cutoff, repeats=tune_m_repeats, replacement=False,
                           m_init=int(np.sqrt(n))) if m_choice == "heuristic" else None
            out.append(psi.test(X, m=m))
        except:
            # Catch errors from test statistic
            print(f"Error occurred {c_e}, {n}")
            out.append(np.nan)
    return(out)

## Conduct multiple experiments with multiprocessing and export to R for plotting:
if __name__ == '__main__':
    repeats = 1000

    # Multiprocess
    pool = Pool(cpu_count()-2)
    res = np.array(
        list(tqdm(pool.imap_unordered(conduct_experiment, range(repeats)), total=repeats)))
    pool.close()

    # Count non-nas, to be used for binomial confidence intervals
    counts = (~np.isnan(res)).sum(axis=0)
    nans = np.isnan(res).sum(axis=0)
    res = np.nansum(res, axis=0)


    df = pd.DataFrame(
        [(x/(repeats-n), *v, *proportion_confint(x, repeats-n, method="binom_test"), n) for x, v, n in zip(res, combinations, nans)],
        columns=["alpha", "n", "Causal_Effect", "m_choice", "Lower", "Upper", "NoNans"])

    # Export to R for ggplotting
    df['alpha'] = df["alpha"].replace(np.NaN, "NA")
    df.to_csv("experiment-dormant-continuous.csv")
