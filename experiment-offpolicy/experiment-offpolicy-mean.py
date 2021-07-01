from collections import namedtuple
import numpy as np
from multiprocessing import Pool, cpu_count
from scipy.stats import wilcoxon, ttest_1samp, mannwhitneyu
from tqdm import tqdm
from itertools import product
import pandas as pd
from statsmodels.stats.proportion import proportion_confint
from statsmodels.stats.weightstats import zconfint


from kernel_two_sample_test import kernel_two_sample_test
from wrapper_resampler import ShiftedTester

# create data object
from Policy import RandomPolicy, LinearPolicy

np.random.seed(11)

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


# Define rates for resampling
def rate(pow, c=1):
    def f(n): return c * n ** pow

    return f


def wilcoxon_test(X):
    _, p = wilcoxon(X[:, -1], alternative='greater')
    return p < .05


def mannwhiteney(X1, X2):
    _, p = mannwhitneyu(X1[:, [-1]], X2[:, [-1]], alternative='two-sided')
    return p < .05


def t_test(X):
    _, p = ttest_1samp(X[:, -1], 0, alternative='greater')
    return p


def MMD_test(X1, X2):
    _, _, p = kernel_two_sample_test(X1[:, [-1]], X2[:, [-1]])
    return p < .05


pow = 0.5
tests = {"Wilcoxon Test": wilcoxon_test}
policy_effects = list(np.linspace(0, 0.5, 6))
combinations = list(product(policy_effects, tests.keys()))

## Wrap as function to apply multiprocessing
def conduct_experiment(i=None):
    out = []
    for effect, test_fn in combinations:
        data = scm(n_train * 3, random_policy)
        # construct a target policy
        target_policy = LinearPolicy(bR * effect, bias=False, temp=2.)
        X = np.hstack([data.X, data.A[:, np.newaxis], data.R[:, np.newaxis]])
        weight = lambda X: target_policy.get_prob(X[:, :-2], X[:, -2].astype(int))
        # get mean_diff
        weights = weight(X)
        mean_diff = np.mean(X[:, -1] * weights)
        # Do not do anything if m < 5 or (m>n and not replacement)
        if rate(pow)(n_train) >= 5 and (rate(pow)(n_train) <= n_train):
            psi = ShiftedTester(weight, tests[test_fn], rate(pow), replacement="REPL-reject")
            out.append((psi.test(X), mean_diff))
    return out


def conduct_experiment_MMD(i=None):
    out = []
    for effect in policy_effects:
        data = scm(n_train * 3, random_policy)
        # construct a target policy
        target_policy = LinearPolicy(bR * effect, bias=False, temp=2.)
        X = np.hstack([data.X, data.A[:, np.newaxis], data.R[:, np.newaxis]])
        # split into two groups
        X_G1, X_G2 = X[:int(X.shape[0] / 2)], X[int(X.shape[0] / 2):]
        # group = np.random.choice(2, size=X.shape[0])
        weightG1 = lambda X: target_policy.get_prob(X[:, :-2], X[:, -2].astype(int))
        weightG2 = lambda X: np.ones(X.shape[0])
        # get mean_diff
        # Do not do anything if m < 5 or (m>n and not replacement)
        if rate(pow)(n_train) >= 5 and (rate(pow)(n_train) <= n_train):
            psiG1 = ShiftedTester(weightG1, mannwhiteney, rate(pow), replacement="REPL-reject")
            psiG2 = ShiftedTester(weightG2, mannwhiteney, rate(pow), replacement="REPL-reject")
            X1m, X2m = psiG1.resample(X_G1), psiG2.resample(X_G2)

            weights = weightG1(X_G1)
            mean_G1, mean_G2 = np.mean(X_G1[:, -1] * weights), np.mean(X_G2[:, -1])

            out.append((mannwhiteney(X1m, X2m), MMD_test(X1m, X2m), mean_G1 - mean_G2))
    return out


## Conduct multiple experiments with multiprocessing and export to R for plotting:
if __name__ == '__main__':
    repeats = 1000

    for two_sample in [True, False]:
        experiment_fn = conduct_experiment_MMD if two_sample else conduct_experiment
        # Multiprocess
        pool = Pool(cpu_count() - 2)
        res = np.array(
            list(tqdm(pool.imap_unordered(experiment_fn, range(repeats)), total=repeats)))
        pool.close()

        if two_sample:
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
            df.to_csv("experiment-offpolicy-MMD.csv")
        else:
            alpha = res[:, :, 0]
            mean_diff = res[:, :, 1]

            # # Count non-nas, to be used for binomial confidence intervals
            counts = (~np.isnan(alpha)).sum(axis=0)
            alpha = np.nansum(alpha, axis=0)
            # Pack as data frame
            df_alpha = pd.DataFrame(
                [(x / c, *v, *proportion_confint(x, c, method="binom_test")) for x, v, c in
                 zip(alpha, combinations, counts)],
                columns=["alpha", "Policy_Effect", "Type", "Lower", "Upper"])
            # mean diff
            df_md = pd.DataFrame(
                [(mean_diff[:, i].mean(), eff, *zconfint(mean_diff[:, i])) for i, eff in enumerate(policy_effects)],
                columns=["mean_effect", "Policy_Effect", "Lower", "Upper"])

            df = df_alpha.join(df_md, rsuffix='_me')

            # Export to R for ggplotting
            df['alpha'] = df["alpha"].replace(np.NaN, "NA")
            df.to_csv("experiment-offpolicy.csv")
