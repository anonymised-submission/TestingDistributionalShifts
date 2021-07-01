from collections import namedtuple
import numpy as np
from multiprocessing import Pool, cpu_count
import statsmodels.api as sm
from scipy.stats import ttest_ind, mannwhitneyu
from tqdm import tqdm
from itertools import product
import pandas as pd
from statsmodels.stats.proportion import proportion_confint
from statsmodels.stats.weightstats import zconfint
from kernel_two_sample_test import kernel_two_sample_test

from wrapper_resampler import ShiftedTester

# create data object
from Policy import RandomPolicy, LinearPolicy

# set seed
np.random.seed(0)

Data = namedtuple('Data', 'X R A')
# number of actions
n_actions = 4
# number of context features
n_contexts = 3
# reward function params
bR = np.random.normal(size=(n_actions, 3)) * 2


# Simulate data from a Gaussian SCM
def scm(n, policy):
    X = np.random.normal(size=(n, n_contexts))
    A = policy.get_actions(X)
    full_R = bR.dot(X.T).T
    R = full_R[np.arange(n), A]
    return Data(X, R, A)


# gen training data
random_policy = RandomPolicy(n_actions)
n_train = 10000
train_data = scm(n_train, random_policy)

# train a target policy
reg_model = [sm.OLS(train_data.R[train_data.A == i], sm.add_constant(train_data.X[train_data.A == i])).fit().params for
             i in range(n_actions)]
reg_model = np.stack(reg_model)


# Define rates for resampling
def rate(pow, c=1):
    def f(n): return c * n ** pow

    return f


def t_test(X1, X2):
    _, p = ttest_ind(X1[:, -1], X2[:, -1])
    return p < .05


def MMD_test(X1, X2):
    _, _, p = kernel_two_sample_test(X1[:, [-1]], X2[:, [-1]])
    return p < .05


def mannwhiteney(X1, X2):
    _, p = mannwhitneyu(X1[:, [-1]], X2[:, [-1]], alternative='two-sided')
    return p < .05


pow = 0.5
tests = {"MMD Test": MMD_test}
policy_effects = np.linspace(1, 40, 6)
combinations = list(product(policy_effects, tests.keys()))


## Wrap as function to apply multiprocessing
def conduct_experiment(i=None):
    out = []
    for effect, test_fn in combinations:
        data = scm(n_train * 3, random_policy)
        # construct a target policy
        max_bR = np.abs(bR).sum(1).argmax()
        taret_weights = np.ones(shape=(n_actions,))
        taret_weights[max_bR] = effect
        target_policy = RandomPolicy(n_actions, weights=taret_weights)
        X = np.hstack([data.X, data.A[:, np.newaxis], data.R[:, np.newaxis]])

        weightG1 = lambda X: target_policy.get_prob(X[:, :-2], X[:, -2].astype(int))
        weightG2 = lambda X: np.ones(X.shape[0])
        # get mean_diff
        weights = weightG1(X)
        mean_diff = np.mean(X[:, -1] * weights)
        # Do not do anything if m < 5 or (m>n and not replacement)
        if rate(pow)(n_train) >= 5 and (rate(pow)(n_train) <= n_train):
            psiG1 = ShiftedTester(weightG1, MMD_test, rate(pow), replacement=False)
            psiG2 = ShiftedTester(weightG2, MMD_test, rate(pow), replacement=False)
            X1m = psiG1.resample(X[:int(X.shape[0] / 2)])
            X2m = psiG2.resample(X[int(X.shape[0] / 2):])
            out.append((mannwhiteney(X1m, X2m), MMD_test(X1m, X2m), mean_diff))
    return out


## Conduct multiple experiments with multiprocessing and export to R for plotting:
if __name__ == '__main__':
    repeats = 1000

    # Multiprocess
    pool = Pool(cpu_count() - 2)
    res = np.array(
        list(tqdm(pool.imap_unordered(conduct_experiment, range(repeats)), total=repeats)))
    pool.close()

    alpha_man = res[:, :, 0]
    alpha_MMD = res[:, :, 1]
    mean_diff = res[:, :, 2]


    def get_df_alpha(alpha):
        # # Count non-nas, to be used for binomial confidence intervals
        counts = (~np.isnan(alpha)).sum(axis=0)
        alpha = np.nansum(alpha, axis=0)
        # Pack as data frame
        df_alpha = pd.DataFrame(
            [(x / c, *v, *proportion_confint(x, c, method="binom_test")) for x, v, c in
             zip(alpha, combinations, counts)],
            columns=["alpha", "Policy_Effect", "Type", "Lower", "Upper"])
        return df_alpha

    df_man = get_df_alpha(alpha_man)
    df_MMD = get_df_alpha(alpha_MMD)
    # mean diff
    df_md = pd.DataFrame(
        [(mean_diff[:, i].mean(), eff, *zconfint(mean_diff[:, i])) for i, eff in enumerate(policy_effects)],
        columns=["mean_effect", "Policy_Effect", "Lower", "Upper"])

    df = df_man.join(df_MMD, rsuffix='_mmd')
    df = df.join(df_md, rsuffix='_me')

    # Export to R for ggplotting
    df['alpha'] = df["alpha"].replace(np.NaN, "NA")
    df.to_csv("experiment-offpolicy-var.csv")
