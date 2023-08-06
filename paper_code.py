import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pathlib

from tqdm import tqdm
from scipy.stats import bernoulli, binom, dirichlet, beta, multinomial, norm, lognorm, uniform
from datetime import datetime

RUN_NAME = pathlib.Path(__file__).name[:-3]
TINY_CONST = 1e-10
SHOW_PLOTS = False
SAVE_PLOTS = True
START_TIME = datetime.now().strftime("%Y%m%d-%H%M%S")
PLOTS_PATH = f'plots/{RUN_NAME}-{START_TIME}'
PLOT_SIZE = (22, 9)
PLOT_DPI = 600
SAVE_RESULTS = True
RESULTS_PATH = f'results/{RUN_NAME}-{START_TIME}'
USE_ORIGINAL_MODEL = False
USE_EXTENSION_LAMBDA = False
USE_EXTENSION_THETA = False
USE_REAL_DATA = False
SIMULATION_DATA = None  # 'simulation_data/data.npz'

# Script length variables
N_EXAMINEES = 500
N_ITERATIONS = 10000
N_REPEATS = 10

if int(os.environ.get('USE_TEST_MODE', 0)) == 1:
    print('Running in test mode...')
    SHOW_PLOTS = True
    SAVE_PLOTS = False
    SAVE_RESULTS = False
    PLOT_DPI = 100
    N_EXAMINEES = 50
    N_ITERATIONS = 1000
    N_REPEATS = 1

print(f'Running with {N_EXAMINEES} examinees, {N_ITERATIONS} iterations, {N_REPEATS} repeats')

if SAVE_PLOTS:
    print(f'Creating directory {PLOTS_PATH}...')
    os.makedirs(PLOTS_PATH, exist_ok=True)

if SAVE_RESULTS:
    print(f'Creating directory {RESULTS_PATH}...')
    os.makedirs(RESULTS_PATH, exist_ok=True)

plt.rcParams["figure.figsize"] = PLOT_SIZE
plt.rcParams["figure.dpi"] = PLOT_DPI

# Simulation parameters
mu_hyperparameter_1 = mu_hyperparameter_2 = 0.5
pi_hyperparams = np.array([1, 1]) if not USE_ORIGINAL_MODEL else np.array([0.01, 0.01])
pi_hyperparams_simulation = np.array([3, 3]) if not USE_ORIGINAL_MODEL else np.array([0.01, 0.01])

true_slipping = 0.3
true_guessing = 0.1

# ---------------------------------------------------------------------------------------------------------------------
# Q MATRICES
# ---------------------------------------------------------------------------------------------------------------------

# DE LA TORRE 2008
de_la_torre_strategy_a_q_matrix = np.array([
    [1, 1, 1, 1, 0],
    [1, 1, 1, 1, 0],
    [1, 0, 0, 0, 1],
    [1, 1, 0, 0, 1],
    [0, 0, 1, 1, 1],
    [0, 0, 1, 1, 1],
    [0, 0, 1, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 1, 0, 1, 1],
    [0, 1, 0, 1, 1],
    [1, 1, 1, 0, 0],
    [1, 1, 0, 0, 1],
    [1, 1, 0, 1, 0],
    [1, 0, 1, 0, 0],
    [1, 0, 1, 0, 1],
    [1, 0, 0, 1, 1],
    [0, 1, 1, 1, 0],
    [0, 1, 1, 0, 1],
    [0, 1, 0, 1, 1],
    [0, 0, 1, 1, 1],
])

de_la_torre_strategy_b_q_matrix = np.array([
    [0, 0, 0, 0, 1],
    [0, 0, 0, 0, 1],
    [0, 1, 0, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 1, 0],
    [0, 1, 1, 1, 0],
    [1, 1, 1, 1, 0],
    [1, 0, 0, 1, 1],
    [1, 1, 1, 0, 1],
    [1, 1, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 1, 0],
    [0, 1, 0, 0, 1],
    [1, 0, 1, 0, 0],
    [0, 0, 0, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 1],
    [1, 1, 0, 0, 0],
    [1, 0, 0, 1, 0],
    [1, 0, 0, 1, 0],
])

# SIMULATION STUDY 2
s2_a_q_matrix = np.array([
    [1, 1, 0, 0, 0],
    [1, 0, 1, 0, 0],
    [1, 0, 0, 1, 0],
    [1, 0, 0, 0, 1],
    [0, 1, 1, 0, 0],
    [0, 1, 0, 1, 0],
    [0, 1, 0, 0, 1],
    [0, 0, 1, 1, 0],
    [0, 0, 1, 0, 1],
    [0, 0, 0, 1, 1],
    [1, 1, 1, 0, 0],
    [1, 1, 0, 1, 0],
    [1, 1, 0, 0, 1],
    [1, 0, 1, 1, 0],
    [1, 0, 1, 0, 1],
    [1, 0, 0, 1, 1],
    [0, 1, 1, 1, 0],
    [0, 1, 1, 0, 1],
    [0, 1, 0, 1, 1],
    [0, 0, 1, 1, 1],
    [1, 1, 0, 0, 0],
    [1, 0, 1, 0, 0],
    [1, 0, 0, 1, 0],
    [1, 0, 0, 0, 1],
    [0, 1, 1, 0, 0],
    [1, 0, 0, 0, 0],
    [0, 1, 0, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 1, 0],
    [0, 0, 0, 0, 1],
])

s2_b_q_matrix = np.array([
    [0, 1, 0, 1, 1],
    [0, 0, 1, 1, 1],
    [0, 1, 1, 0, 1],
    [0, 1, 1, 1, 0],
    [1, 0, 0, 1, 1],
    [1, 1, 0, 0, 1],
    [1, 1, 0, 1, 0],
    [1, 0, 1, 0, 1],
    [1, 0, 1, 1, 0],
    [1, 1, 1, 0, 0],
    [0, 0, 0, 1, 1],
    [0, 1, 0, 0, 1],
    [0, 1, 0, 1, 0],
    [0, 0, 1, 0, 1],
    [0, 0, 1, 1, 0],
    [0, 1, 1, 0, 0],
    [1, 0, 0, 0, 1],
    [1, 0, 0, 1, 0],
    [1, 1, 0, 0, 0],
    [1, 0, 1, 0, 0],
    [1, 0, 0, 0, 0],
    [0, 1, 0, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 1, 0],
    [0, 0, 0, 0, 1],
    [0, 1, 1, 0, 0],
    [1, 0, 0, 0, 1],
    [1, 0, 0, 1, 0],
    [1, 1, 0, 0, 0],
    [1, 0, 1, 0, 0],
])

# SIMULATION STUDY 3
s3_a_q_matrix = np.array([
    [1, 1, 1, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 1],
    [0, 0, 1, 0, 0, 0, 0],
    [1, 0, 0, 1, 0, 0, 0],
    [0, 0, 1, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 1, 0],
    [1, 0, 0, 0, 0, 0, 1],
    [0, 1, 0, 1, 0, 0, 0],
    [1, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0],
    [1, 0, 0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 1],
    [0, 1, 1, 0, 0, 0, 0],
    [1, 1, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 1, 0, 0],
    [0, 1, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0],
    [1, 0, 0, 0, 0, 1, 0],
    [0, 0, 1, 0, 0, 0, 1],
    [0, 0, 0, 1, 1, 0, 0],
    [0, 0, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 0, 0, 1],
    [0, 0, 0, 0, 1, 1, 0],
    [0, 0, 0, 0, 1, 0, 1],
    [0, 0, 0, 0, 0, 1, 1],
    [1, 0, 0, 0, 0, 0, 0],
    [1, 1, 0, 1, 0, 0, 0],
    [1, 1, 0, 0, 1, 0, 0],
    [1, 1, 0, 0, 0, 1, 0],
    [1, 1, 0, 0, 0, 0, 1],
    [0, 1, 1, 1, 0, 0, 0],
    [0, 0, 1, 0, 1, 0, 1],
    [0, 0, 1, 0, 0, 1, 1],
    [0, 1, 1, 0, 0, 0, 1],
    [0, 0, 1, 1, 1, 0, 0],
    [0, 0, 0, 1, 0, 1, 1],
    [0, 0, 0, 1, 1, 1, 0],
])

s3_b_q_matrix = np.array([
    [1, 1, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0],
    [1, 0, 0, 1, 0, 0, 0],
    [1, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 1, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 1, 1, 1, 0],
    [1, 0, 0, 0, 0, 0, 0],
    [1, 0, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 0, 0, 0, 0],
    [1, 1, 0, 1, 0, 0, 0],
    [1, 1, 0, 0, 1, 0, 0],
    [1, 1, 0, 0, 0, 1, 0],
    [0, 0, 1, 0, 0, 1, 1],
    [0, 1, 1, 1, 0, 0, 0],
    [0, 0, 1, 0, 1, 0, 1],
    [1, 1, 0, 0, 0, 0, 1],
    [0, 1, 1, 0, 0, 0, 1],
    [0, 0, 1, 1, 1, 0, 0],
    [0, 0, 0, 1, 0, 1, 1],
    [0, 1, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 1, 0],
    [0, 1, 0, 0, 1, 0, 0],
    [0, 1, 0, 0, 0, 0, 1],
    [0, 0, 1, 1, 0, 0, 0],
    [0, 0, 1, 0, 1, 0, 0],
    [0, 1, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0],
    [0, 0, 0, 1, 0, 1, 0],
    [0, 0, 1, 0, 0, 0, 1],
    [0, 0, 0, 1, 0, 0, 1],
    [0, 0, 0, 0, 1, 1, 0],
    [0, 0, 0, 0, 1, 0, 1],
    [0, 0, 0, 0, 0, 1, 1]
])

if USE_REAL_DATA:
    # load the q matrix from fs_qmatrix.csv
    q_raw = pd.read_csv('fs_qmatrix.csv', header=0, index_col=[0, 1])
    q_a = q_raw.iloc[:, :7].values
    q_b = q_raw.iloc[:, 7:].values
    q = np.stack([q_a, q_b]).transpose([1, 2, 0])

else:
    # de la Torre 2008
    q = np.stack([de_la_torre_strategy_a_q_matrix, de_la_torre_strategy_b_q_matrix]).transpose([1, 2, 0])

    # Simulation Study 2
    # q = np.stack([s2_a_q_matrix, s2_b_q_matrix]).transpose([1, 2, 0])

    # Simulation Study 3
    # q = np.stack([s3_a_q_matrix, s3_b_q_matrix]).transpose([1, 2, 0])

# ---------------------------------------------------------------------------------------------------------------------
# SIMULATION PARAMETERS
# ---------------------------------------------------------------------------------------------------------------------

n_attributes = q.shape[1]  # K attributes
n_strategies = q.shape[2]  # M strategies
n_items = q.shape[0]  # J items, j-th item
n_examinees = N_EXAMINEES

slipping_trace_avg = np.ones([N_REPEATS, n_strategies])
guessing_trace_avg = np.ones([N_REPEATS, n_strategies])
slipping_traces = np.ones([N_REPEATS, N_ITERATIONS, n_strategies])
guessing_traces = np.ones([N_REPEATS, N_ITERATIONS, n_strategies])
pi_avg = np.ones(N_REPEATS)

# ---------------------------------------------------------------------------------------------------------------------
# SIMULATION THE EXAMINEES TAKING A TEST
# ---------------------------------------------------------------------------------------------------------------------

if USE_REAL_DATA:
    score = pd.read_csv('fs_score.csv', header=0, index_col=0).values
    n_examinees = score.shape[0]

else:
    if SIMULATION_DATA is not None and pathlib.Path(SIMULATION_DATA).exists():
        # load simulation data if it already exists
        print(f'Loading simulation data from {SIMULATION_DATA}...')
        saved_data = np.load(SIMULATION_DATA)
        alpha_sim = saved_data['alpha_sim']
        pi = saved_data['pi']
        pi_true = saved_data['pi']
        s_c = saved_data['s_c']
        g = saved_data['g']
        eta = saved_data['eta']
        score = saved_data['score']

    else:
        alpha_sim = np.array(bernoulli.rvs(0.5, size=(n_examinees, n_attributes)))
        pi = 1 - dirichlet.rvs(pi_hyperparams_simulation, size=1).flatten()
        pi_true = pi
        s_c = np.ones((n_items, n_strategies)) * (1 - true_slipping)
        g = np.ones((n_items, n_strategies)) * true_guessing

        print('pi for simulation:', pi)

        # build score from item response function
        eta = np.zeros(shape=(n_examinees, n_items, n_strategies))
        for i in range(n_examinees):
            for j in range(n_items):
                for m in range(n_strategies):
                    eta[i, j, m] = np.prod([alpha_sim[i, k] ** q[j, k, m] for k in range(n_attributes)])

        score = np.zeros(shape=(n_examinees, n_items))
        for i in range(n_examinees):
            for j in range(n_items):
                score[i, j] = bernoulli.rvs(
                    np.sum([pi[m] * s_c[j, m] ** eta[i, j, m] * g[j, m] ** (1 - eta[i, j, m]) for m in range(n_strategies)]))

        if SIMULATION_DATA is not None:
            # save simulation data
            print(f'Saving simulation data to {SIMULATION_DATA}...')
            os.makedirs(pathlib.Path(SIMULATION_DATA).parent.resolve(), exist_ok=True)
            np.savez(SIMULATION_DATA, alpha_sim=alpha_sim, pi=pi, s_c=s_c, g=g, eta=eta, score=score)

# ---------------------------------------------------------------------------------------------------------------------
# START WITH MCMC SAMPLING
# ---------------------------------------------------------------------------------------------------------------------

EM = N_ITERATIONS
BI = int(EM / 2)
repetitions = N_REPEATS  # number of repetitions

# track c, s, g
c_hat = np.zeros((repetitions, EM, n_examinees))
pi_hat = np.zeros((repetitions, EM, n_strategies))
mu_hat = np.zeros((repetitions, EM, n_strategies))
slipping = np.zeros((repetitions, EM, n_items, n_strategies))
guessing = np.zeros((repetitions, EM, n_items, n_strategies))
alpha_hat = np.zeros((repetitions, EM, n_examinees, n_attributes))
theta_hat = np.zeros((repetitions, EM, n_examinees))
lambda_0_hat = np.zeros((repetitions, EM, n_attributes))
lambda_1_hat = np.zeros((repetitions, EM, n_attributes))
mu_weight_hat = np.zeros((repetitions, EM))

for rep in range(repetitions):
    # Initial values
    alpha = np.array(bernoulli.rvs(0.5, size=(n_examinees, n_attributes)))
    mu = np.array(beta.rvs(mu_hyperparameter_1, mu_hyperparameter_2, size=n_strategies))
    pi = dirichlet.rvs(pi_hyperparams, size=1).flatten()
    s_c = 1 - np.array(beta.rvs(1, 2, size=(n_items, n_strategies)) * 0.4 + 0.1)
    g = np.array(beta.rvs(1, 2, size=(n_items, n_strategies)) * 0.4 + 0.1)
    c = np.argmax(multinomial.rvs(n=1, p=pi, size=n_examinees), axis=1)
    theta = np.array(beta.rvs(2, 2, size=n_examinees))
    mu_weight = uniform.rvs(0, 1)

    if USE_EXTENSION_LAMBDA:
        lambda_0 = np.array(norm.rvs(loc=0, scale=1, size=n_attributes))
        lambda_1 = np.array(lognorm.rvs(1, loc=1, scale=1, size=n_attributes))
    else:
        lambda_0 = np.array([-0.95, -1.42, -0.66, 0.5, -0.05])
        lambda_1 = np.array([1.34, 1.22, 1.08, 1.11, 0.97])

    # after burn-in variables
    bi_counter = 0
    alpha_sum = np.zeros((n_examinees, n_attributes))
    s_c_sum = np.zeros((n_items, n_strategies))
    g_sum = np.zeros((n_items, n_strategies))

    # for after the burn-in period
    alpha_diff = np.zeros((n_examinees, n_attributes))

    # Start the MCMC
    print('\nRepetition: ', rep)
    for WWW in tqdm(range(EM)):

        # -------------------------------------------------------------------------------------------------------------
        # METROPOLIS-HASTINGS SAMPLING
        # -------------------------------------------------------------------------------------------------------------

        if USE_EXTENSION_THETA:
            mu_weight_hyperparameter_1 = 1
            mu_weight_hyperparameter_2 = 1
            mu_weights = []

            for examinee in range(n_examinees):
                strategy_membership = c[examinee]
                mu_weight_new = beta.rvs(mu_weight_hyperparameter_1, mu_weight_hyperparameter_2)

                likelihood_alpha_given_theta = np.prod(
                    bernoulli.pmf(k=alpha[examinee, :], p=1 / (1 + np.exp(-1.7 * lambda_1 * (np.prod(theta) - lambda_0))))
                )

                probability_old = mu_weight * binom.pmf(np.sum(alpha[examinee, :]), n_attributes, mu[strategy_membership]) + (1 - mu_weight) * likelihood_alpha_given_theta
                probability_new = mu_weight_new * binom.pmf(np.sum(alpha[examinee, :]), n_attributes, mu[strategy_membership]) + (1 - mu_weight_new) * likelihood_alpha_given_theta

                likelihood_ratio = (probability_new * beta.pdf(mu_weight_new, mu_weight_hyperparameter_1, mu_weight_hyperparameter_2)) \
                    / (probability_old * beta.pdf(mu_weight, mu_weight_hyperparameter_1, mu_weight_hyperparameter_2))

                if likelihood_ratio >= np.random.rand():
                    mu_weights.append(mu_weight_new)

            mu_weight = np.mean(mu_weights)
            mu_weight_hat[rep, WWW] = mu_weight

        # -------------------------------------------------------------------------------------------------------------
        # GIBBS SAMPLING
        # -------------------------------------------------------------------------------------------------------------

        # draw pi using Gibbs Sampler (always accept, draw from conditional posterior)
        membership_counts = np.array([np.sum(c == m) for m in range(n_strategies)])
        # membership_counts = np.flip(membership_counts)
        if USE_ORIGINAL_MODEL:
            pi = 1 - dirichlet.rvs(pi_hyperparams + membership_counts, size=1).flatten()
        else:
            pi = dirichlet.rvs(pi_hyperparams + membership_counts, size=1).flatten()
        pi_hat[rep, WWW] = pi

        # draw c (strategy membership parameter), using Gibbs Sampler (always accept, draw from conditional posterior)
        # TODO parallelize
        for examinee in range(n_examinees):
            # p(c_i = m | all other parameters)
            Lc = np.ones(n_strategies)
            # TODO parallelize / vectorize
            for strategy in range(n_strategies):
                likelihood = 1
                for item in range(n_items):
                    eta = np.prod(alpha[examinee, :] ** q[item, :, strategy])

                    # some kinda likelihood
                    tem = (s_c[item, strategy] ** eta) * (g[item, strategy] ** (1 - eta))

                    #  p_ijm ^ u_ij * (1 - p_ijm) ^ (1 - u_ij)
                    p = tem if score[examinee, item] == 1 else 1 - tem
                    likelihood *= np.maximum(p, TINY_CONST)  # likelihood

                # L[strategy] = binom.pmf(np.sum(alpha[examinee, :]), n_attributes, mu[strategy])  # prior
                if USE_EXTENSION_THETA:
                    likelihood_alpha_given_theta = np.prod(
                        bernoulli.pmf(k=alpha[examinee, :], p=1 / (1 + np.exp(-1.7 * lambda_1 * (np.prod(theta) - lambda_0))))
                    )
                    prior = mu_weight * np.prod(bernoulli.pmf(alpha[examinee, :], mu[strategy])) + (1 - mu_weight) * likelihood_alpha_given_theta
                    # prior = mu_weight * binom.pmf(np.sum(alpha[examinee, :]), n_attributes, mu[strategy]) + (1 - mu_weight) * likelihood_alpha_given_theta
                else:
                    prior = np.prod(bernoulli.pmf(alpha[examinee, :], mu[strategy]))
                    # prior = binom.pmf(np.sum(alpha[examinee, :]), n_attributes, mu[strategy])
                # prior = np.prod(
                #     [bernoulli.pmf(alpha[examinee, attribute], prob_alpha) for attribute in range(n_attributes)])
                Lc[strategy] = prior * likelihood * pi[strategy]

            # c_hat[examinee, :] = 10 ** 5 * Lc
            # pp = Lc[1] / (Lc[0] + Lc[1])
            # print(WWW, examinee, pp, Lc, L, LL, p)
            c[examinee] = np.argmax(multinomial.rvs(1, Lc / np.sum(Lc)))  # np.random.binomial(1, pp)
            c_hat[rep, WWW, examinee] = c[examinee]

        # draw mu (strategy membership parameter), using Gibbs Sampler (always accept, draw from conditional posterior)
        num_attributes = np.sum(alpha, axis=1)
        attr_sum = [np.sum(num_attributes[c == m]) for m in range(n_strategies)]

        if USE_ORIGINAL_MODEL:
            mu = [
                np.random.beta(attr_sum[m] + mu_hyperparameter_1, alpha.size - attr_sum[m] + mu_hyperparameter_2)
                for m in range(n_strategies)
            ]
        else:  # USE_NORMAL_MODEL
            mu = [
                np.random.beta(attr_sum[m] + mu_hyperparameter_1, alpha[c == m].size - attr_sum[m] + mu_hyperparameter_2)
                for m in range(n_strategies)
            ]
        mu_hat[rep, WWW] = mu

        # -------------------------------------------------------------------------------------------------------------
        # METROPOLIS-HASTINGS SAMPLING
        # -------------------------------------------------------------------------------------------------------------

        # draw lambda
        if USE_EXTENSION_LAMBDA:
            # Pre-compute constant values
            theta_prod = np.prod(theta)
            binom_values = np.array([binom.pmf(np.sum(alpha[examinee, :]), n_attributes, mu[c[examinee]]) for examinee in range(n_examinees)])

            # Loop over attributes
            for attribute in range(n_attributes):
                lower_0 = lambda_0[attribute] - 1
                upper_0 = lambda_0[attribute] + 1
                lambda_0_new = uniform.rvs(loc=lower_0, scale=upper_0 - lower_0)

                lower_1 = lambda_1[attribute] - 1
                upper_1 = lambda_1[attribute] + 1
                lambda_1_new = uniform.rvs(loc=lower_1, scale=upper_1 - lower_1)

                p_old = 1 / (1 + np.exp(-1.7 * lambda_1[attribute] * (theta_prod - lambda_0[attribute])))
                p_new = 1 / (1 + np.exp(-1.7 * lambda_1_new * (theta_prod - lambda_0_new)))

                probability_old = bernoulli.pmf(k=alpha[:, attribute], p=p_old)
                probability_new = bernoulli.pmf(k=alpha[:, attribute], p=p_new)

                p_alpha_lambda_all = mu_weight * binom_values + (1 - mu_weight) * probability_old
                p_alpha_lambda_new_all = mu_weight * binom_values + (1 - mu_weight) * probability_new

                p_alpha_lambda = np.exp(np.sum(np.log(p_alpha_lambda_all)))
                p_alpha_lambda_new = np.exp(np.sum(np.log(p_alpha_lambda_new_all)))

                likelihood_0 = (p_alpha_lambda_new * norm.pdf(lambda_0_new, loc=0, scale=1)) / (p_alpha_lambda * norm.pdf(lambda_0[attribute], loc=0, scale=1))
                if likelihood_0 >= np.random.rand():
                    lambda_0[attribute] = lambda_0_new

                likelihood_1 = (np.prod(p_alpha_lambda_new) * lognorm.pdf(x=lambda_1_new, s=1, loc=0, scale=1)) / (np.prod(p_alpha_lambda) * lognorm.pdf(x=lambda_1[attribute], s=1, loc=0, scale=1))
                if likelihood_1 >= np.random.rand():
                    lambda_1[attribute] = lambda_1_new

            lambda_0_hat[rep, WWW] = lambda_0
            lambda_1_hat[rep, WWW] = lambda_1

        # draw theta
        if USE_EXTENSION_THETA:
            # TODO parallelize
            p_alpha_theta = 1
            p_alpha_theta_new = 1

            for examinee in range(n_examinees):
                strategy_membership = c[examinee]
                theta_new = norm.rvs(loc=theta[examinee], scale=1)

                # Ratio of likelihoods for the new and prior theta values
                # for attribute in range(n_attributes):
                # p_Theta = np.prod([bernoulli.pmf(k=alpha[examinee, attribute], p=1 / (1 + np.exp(-1.7 * lambda_1[attribute] * (theta[examinee] - lambda_0[attribute])))) for attribute in range(n_attributes)])
                p_theta = np.prod(bernoulli.pmf(k=alpha[examinee, :],
                                                p=1 / (1 + np.exp(-1.7 * lambda_1 * (theta[examinee] - lambda_0)))))
                # p_ThetaNew = np.prod([bernoulli.pmf(k=alpha[examinee, attribute], p=(1 / (1 + np.exp(-1.7 * lambda_1[attribute] * (theta_new - lambda_0[attribute]))))) for attribute in range(n_attributes)])
                p_theta_new = np.prod(bernoulli.pmf(k=alpha[examinee, :],
                                                    p=(1 / (1 + np.exp(-1.7 * lambda_1 * (theta_new - lambda_0))))))

                # probability = theta_weight * mu[strategy_membership] + (1 - theta_weight) * theta[examinee]
                # probability = mu_weight * binom.pmf(np.sum(alpha[examinee, :]), n_attributes, mu[strategy_membership]) + (1 - mu_weight) * p_theta_new
                # probability_old = mu_weight * binom.pmf(np.sum(alpha[examinee, :]), n_attributes, mu[strategy_membership]) + (1 - mu_weight) * p_theta

                # LLa = probability / \
                #       probability_old

                # tem = 1 / (1+ np.exp(-1.7*lambda_1[attribute]*(theta[examinee] - lambda_0[attribute])))
                # p_alpha_theta *= bernoulli.pmf(k = alpha[examinee, attribute],p = tem)

                # tem_new = 1 / (1+ np.exp(-1.7*lambda_1[attribute]*(theta_new - lambda_0[attribute])))
                # p_alpha_theta_new *=  bernoulli.pmf(k = alpha[examinee, attribute],p = tem_new)

                likelihood = (p_theta_new * norm.pdf(theta_new, loc=0, scale=1)) / \
                             (p_theta * norm.pdf(theta[examinee], loc=0, scale=1))

                if likelihood >= np.random.rand():
                    theta[examinee] = 1 / (1 + np.exp(-theta_new))

                theta_hat[rep, WWW, examinee] = theta[examinee]

        # draw alpha (latent skill vector), using Metropolis-Hastings (accept/reject)
        # TODO parallelize
        for examinee in range(n_examinees):
            strategy_membership = int(c[examinee])
            # using n=1, this is a bernoulli draw
            # alpha_new = np.random.binomial(n=1, p=mu[strategy_membership], size=n_attributes)
            alpha_new = np.random.binomial(n=1, p=0.5, size=n_attributes)

            if USE_EXTENSION_THETA:
                p_alpha_given_Theta = np.prod(bernoulli.pmf(
                    k=alpha[examinee, :],
                    p=1 / (1 + np.exp(-1.7 * lambda_1 * (theta[examinee] - lambda_0)))))

                p_NewAlpha_given_Theta = np.prod(bernoulli.pmf(
                    k=alpha_new,
                    p=1 / (1 + np.exp(-1.7 * lambda_1 * (theta[examinee] - lambda_0)))))

                # probability = theta_weight * mu[strategy_membership] + (1 - theta_weight) * theta[examinee]
                probability = mu_weight \
                    * binom.pmf(np.sum(alpha_new), n_attributes, mu[strategy_membership]) \
                    + (1 - mu_weight) \
                    * p_NewAlpha_given_Theta

                probability_old = mu_weight \
                    * binom.pmf(np.sum(alpha[examinee, :]), n_attributes, mu[strategy_membership]) \
                    + (1 - mu_weight) \
                    * p_alpha_given_Theta

                LLa = probability / \
                    probability_old
            else:
                LLa = binom.pmf(np.sum(alpha_new), n_attributes, mu[strategy_membership]) / \
                      binom.pmf(np.sum(alpha[examinee, :]), n_attributes, mu[strategy_membership])

            # Ratio of likelihoods for the new and prior alpha values
            # LLa = binom.pmf(np.sum(alpha_new), n_attributes, probability) / \
            #   binom.pmf(np.sum(alpha[examinee, :]), n_attributes, probability)
            # LLa = binom.pmf(np.sum(alpha[examinee, :]), n_attributes, mu[strategy_membership]) / \
            #       binom.pmf(np.sum(alpha_new), n_attributes, mu[strategy_membership])

            LLLa = 1
            # TODO parallelize / vectorize
            for item in range(n_items):
                eta = np.prod(alpha[examinee, :] ** q[item, :, strategy_membership])
                eta_new = np.prod(alpha_new ** q[item, :, strategy_membership])

                tem = (s_c[item, strategy_membership] ** eta) * (g[item, strategy_membership] ** (1 - eta))
                tem_new = (s_c[item, strategy_membership] ** eta_new) * (g[item, strategy_membership] ** (1 - eta_new))

                if score[examinee, item] == 1:
                    temp = tem_new / np.maximum(tem, TINY_CONST)
                    # temp = np.maximum(tem, TINY_CONST) / tem_new
                else:
                    temp = (1 - tem_new) / (1 - tem)
                    # temp = (1 - tem ) / (1 - tem_new)

                LLLa *= temp

            p3 = LLa * LLLa
            if p3 >= np.random.rand():
                alpha[examinee, :] = alpha_new

            alpha_hat[rep, WWW, examinee] = alpha[examinee]

        # draw s and g (item parameters), using Metropolis-Hastings (accept/reject)
        s_c_new = np.zeros((n_items, n_strategies))
        g_new = np.zeros((n_items, n_strategies))

        for item in range(n_items):
            for strategy in range(n_strategies):
                g_new[item, strategy] = np.random.uniform(0.0, 0.2)
                s_c_new[item, strategy] = np.random.uniform(0.6, 0.8)
                # g_new[item, strategy] = beta.rvs(1, 2) * 0.4 + 0.1
                # s_c_new[item, strategy] = beta.rvs(1, 2) * 0.4 + 0.1

            log_likelihood = np.zeros(n_strategies)

            # { PROD i=1 to N: [ p_ijm ^ u_ij * (1 - p_ijm) ^ (1 - u_ij) ] ^ c_j ] ^ I(c_i = m) } Beta(s_jm) x Beta(g_jm)
            # TODO parallelize / vectorize
            for examinee in range(n_examinees):
                strategy_membership = int(c[examinee])
                eta = np.prod(alpha[examinee, :] ** q[item, :, strategy_membership])

                tem = (s_c[item, strategy_membership] ** eta) * (g[item, strategy_membership] ** (1 - eta))
                tem_new = (s_c_new[item, strategy_membership] ** eta) * (g_new[item, strategy_membership] ** (1 - eta))

                if score[examinee, item] == 1:
                    p = np.maximum(tem, TINY_CONST)
                    p_new = tem_new
                else:
                    p = 1 - tem
                    p_new = 1 - tem_new

                log_likelihood[strategy_membership] += np.log(p_new / p)

            likelihood = np.exp(log_likelihood)

            for strategy in range(n_strategies):
                if likelihood[strategy] >= np.random.rand():
                    g[item, strategy] = g_new[item, strategy]
                    s_c[item, strategy] = s_c_new[item, strategy]

        slipping[rep, WWW] = s_c
        guessing[rep, WWW] = g

        # If we are past the burn-in period, the sum alpha, s and g to get an average value of them
        if WWW >= EM - BI:
            bi_counter += 1
            # TODO parallelize / vectorize
            alpha_sum += alpha
            s_c_sum += s_c
            g_sum += g

    alpha_avg = alpha_sum / BI
    s_c_avg = s_c_sum / BI
    s_avg = 1 - s_c_avg
    g_avg = g_sum / BI

    alpha_final = np.round(alpha_avg)

    slipping_trace = np.mean(1 - slipping[rep], axis=1)
    guessing_trace = np.mean(guessing[rep], axis=1)

    eta = np.zeros(shape=(n_examinees, n_items, n_strategies))
    for i in range(n_examinees):
        for j in range(n_items):
            for m in range(n_strategies):
                eta[i, j, m] = np.prod(alpha[i, :] ** q[j, :, m])

    p_MMS = np.zeros((n_examinees, n_items))
    score_pred = np.zeros((n_examinees, n_items))

    slipping_mean = np.mean(slipping[rep, BI:], axis=0)
    guessing_mean = np.mean(guessing[rep, BI:], axis=0)
    pi_mean = np.mean(pi_hat[rep, BI:], axis=0)

    # mixture multiple strategy model
    for examinee in range(n_examinees):
        for item in range(n_items):
            p_ijm = np.zeros(n_strategies)
            for strategy in range(n_strategies):
                tem = (slipping_mean[item, strategy] ** eta[examinee, item, strategy]) \
                      * (guessing_mean[item, strategy] ** (1 - eta[examinee, item, strategy]))
                p_ijm[strategy] = tem
            p_MMS[examinee, item] = np.sum(
                pi_mean * (p_ijm ** score[examinee, item]) * ((1 - p_ijm) ** (1 - score[examinee, item])))
            score_pred[examinee, item] = np.sum(pi_mean * p_ijm)
            # p_MMS[examinee, item] = np.sum(p_ijm)

    # -----------------------------------------------------------------------------------------------------------------
    # PLOTTING
    # -----------------------------------------------------------------------------------------------------------------

    # c - strategy membership
    plt.plot(np.arange(EM), np.mean(c_hat[rep], axis=1))
    plt.ylim([0, 1])
    plt.suptitle(f'c ({rep})') if USE_REAL_DATA else plt.suptitle(f'c ({rep}) - (pi_true: {pi_true})')
    plt.tight_layout()
    if SAVE_PLOTS:
        plt.savefig(f'{PLOTS_PATH}/{rep}_c.png')
    if SHOW_PLOTS:
        plt.show()
    plt.clf()
    plt.close()

    # mu_weight
    plt.plot(np.arange(EM), mu_weight_hat[rep])
    plt.suptitle(f'mu_weight ({rep})')
    plt.ylim([0, 1])
    plt.tight_layout()
    if SAVE_PLOTS:
        plt.savefig(f'{PLOTS_PATH}/{rep}_mu_weight.png')
    if SHOW_PLOTS:
        plt.show()
    plt.clf()
    plt.close()

    # s and g - slipping and guessing parameters
    plt.plot(np.arange(EM), slipping_trace[:, 0], label='slipping strategy 1')
    plt.plot(np.arange(EM), slipping_trace[:, 1], label='slipping strategy 2')
    plt.plot(np.arange(EM), guessing_trace[:, 0], label='guessing strategy 1')
    plt.plot(np.arange(EM), guessing_trace[:, 1], label='guessing strategy 2')
    plt.ylim([0, 0.5])
    plt.suptitle(f'slipping and guessing ({rep})')
    plt.legend()
    plt.tight_layout()
    if SAVE_PLOTS:
        plt.savefig(f'{PLOTS_PATH}/{rep}_s_g.png')
    if SHOW_PLOTS:
        plt.show()
    plt.clf()
    plt.close()

    # pi - strategy mixing parameter
    plt.plot(pi_hat[rep, :, 0], label='strategy 1')
    plt.plot(pi_hat[rep, :, 1], label='strategy 2')
    plt.ylim([0, 1])
    plt.suptitle(f'pi ({rep})') if USE_REAL_DATA else plt.suptitle(f'pi ({rep}) - (pi_true: {pi_true})')
    plt.legend()
    plt.tight_layout()
    if SAVE_PLOTS:
        plt.savefig(f'{PLOTS_PATH}/{rep}_pi.png')
    if SHOW_PLOTS:
        plt.show()
    plt.clf()
    plt.close()

    # mu - attribute mean
    plt.plot(mu_hat[rep, :, 0], label='strategy 1')
    plt.plot(mu_hat[rep, :, 1], label='strategy 2')
    plt.ylim([0, 1])
    plt.suptitle(f'mu ({rep})')
    plt.legend()
    plt.tight_layout()
    if SAVE_PLOTS:
        plt.savefig(f'{PLOTS_PATH}/{rep}_mu.png')
    if SHOW_PLOTS:
        plt.show()
    plt.clf()
    plt.close()

    # theta - general knowledge
    if USE_EXTENSION_THETA:
        plt.plot(np.arange(EM), np.mean(theta_hat[rep], axis=1))
        plt.ylim([0, 1])
        plt.suptitle(f'theta ({rep})')
        plt.tight_layout()
        if SAVE_PLOTS:
            plt.savefig(f'{PLOTS_PATH}/{rep}_theta.png')
        if SHOW_PLOTS:
            plt.show()
        plt.clf()
        plt.close()

    # lambda - attribute weight
    if USE_EXTENSION_THETA:
        for attribute in range(n_attributes):
            plt.plot(np.arange(EM), lambda_0_hat[rep, :, attribute], c='r', label=f'Lambda 0 [{attribute}]')
            plt.plot(np.arange(EM), lambda_1_hat[rep, :, attribute], c='b', label=f'Lambda 1 [{attribute}]')
        plt.suptitle(f'lambda ({rep})')
        plt.tight_layout()
        plt.legend()
        if SAVE_PLOTS:
            plt.savefig(f'{PLOTS_PATH}/{rep}_lambda.png')
        if SHOW_PLOTS:
            plt.show()
        plt.clf()
        plt.close()

    # plotting matrices
    cmap = 'plasma'
    colorbar_location = 'right'

    # alpha - latent skill vector
    if USE_REAL_DATA:
        plt.matshow(alpha_final.T, aspect='auto', cmap=cmap)
        plt.suptitle('alpha')
        plt.colorbar(location=colorbar_location)
        plt.tight_layout()
        if SAVE_PLOTS:
            plt.savefig(f'{PLOTS_PATH}/{rep}_alpha.png')
        if SHOW_PLOTS:
            plt.show()
        plt.clf()
        plt.close()

    # alpha (difference, simulated and sampled) - latent skill vector
    else:
        alpha_diff = np.abs(alpha_sim - alpha_final)
        print(f'weight for mu {mu_weight}: {np.sum(alpha_diff) / (n_attributes * n_examinees)}')

        fig, ax = plt.subplots(nrows=3, ncols=1)
        alpha_diff_ax = ax[0].matshow(1 - alpha_diff.T, aspect='auto', cmap='PiYG')
        ax[0].set_title('alpha_sim - alpha_final')
        alpha_sim_ax = ax[1].matshow(alpha_sim.T, aspect='auto', cmap=cmap)
        ax[1].set_title('alpha_sim')
        alpha_ax = ax[2].matshow(alpha_final.T, aspect='auto', cmap=cmap)
        ax[2].set_title('alpha_final')
        fig.colorbar(alpha_diff_ax, location=colorbar_location)
        fig.colorbar(alpha_sim_ax, location=colorbar_location)
        fig.colorbar(alpha_ax, location=colorbar_location)
        plt.suptitle(f'alpha_sim vs. alpha ({rep})')
        plt.tight_layout()
        if SAVE_PLOTS:
            plt.savefig(f'{PLOTS_PATH}/{rep}_alpha.png')
        if SHOW_PLOTS:
            plt.show()
        plt.clf()
        plt.close()

    # score and predicted probability
    score_diff = np.abs(score - np.round(score_pred))

    fig, ax = plt.subplots(nrows=4, ncols=1)
    score_ax = ax[0].matshow(score.T, aspect='auto', cmap=cmap)
    ax[0].set_title('score')
    score_pred_ax = ax[1].matshow(score_pred.T, aspect='auto', cmap=cmap)
    ax[1].set_title('score_pred')
    p_MMS_ax = ax[2].matshow(p_MMS.T, aspect='auto', cmap=cmap)
    ax[2].set_title('p_MMS')
    score_diff_ax = ax[3].matshow(1 - score_diff.T, aspect='auto', cmap='PiYG')
    ax[3].set_title('score - score_pred')
    fig.colorbar(score_ax, location=colorbar_location)
    fig.colorbar(score_pred_ax, location=colorbar_location)
    fig.colorbar(p_MMS_ax, location=colorbar_location)
    fig.colorbar(score_diff_ax, location=colorbar_location)
    plt.suptitle(f'score vs. p_MMS ({rep})')
    plt.tight_layout()
    if SAVE_PLOTS:
        plt.savefig(f'{PLOTS_PATH}/{rep}_p_MMS.png')
    if SHOW_PLOTS:
        plt.show()
    plt.clf()
    plt.close()

    # -----------------------------------------------------------------------------------------------------------------
    # SAVE RESULTS
    # -----------------------------------------------------------------------------------------------------------------

    if SAVE_RESULTS:
        np.savez(
            f'{RESULTS_PATH}/{rep}.npz',
            c_hat=c_hat[rep],
            pi_hat=pi_hat[rep],
            mu_hat=mu_hat[rep],
            slipping=slipping[rep],
            guessing=guessing[rep],
            alpha_hat=alpha_hat[rep],
            theta_hat=theta_hat[rep],
            lambda_0_hat=lambda_0_hat[rep],
            lambda_1_hat=lambda_1_hat[rep],
            score=score,
            score_pred=score_pred,
            score_diff=score_diff,
            p_MMS=p_MMS,
            alpha_sim=alpha_sim if not USE_REAL_DATA else None,
            alpha_final=alpha_final,
            alpha_diff=alpha_diff if not USE_REAL_DATA else None,
            slipping_trace=slipping_trace[rep],
            guessing_trace=guessing_trace[rep],
            mu_weight_hat=mu_weight_hat[rep],
            pi_true=pi_true if not USE_REAL_DATA else None,
        )

    # Keep track of avg values over repetitions
    pi_avg[rep] = pi_mean[0]
    slipping_traces[rep, :, :] = slipping_trace
    guessing_traces[rep, :, :] = guessing_trace
    slipping_trace_avg[rep] = np.mean(slipping_trace, axis=0)
    guessing_trace_avg[rep] = np.mean(guessing_trace, axis=0)

# -----------------------------------------------------------------------------------------------------------------
# ERROR MEASURES
# -----------------------------------------------------------------------------------------------------------------

# Plot Parameter Recovery
for rep in range(N_REPEATS):
    plt.plot(np.arange(EM), slipping_traces[rep, :, 0], label='slipping strategy 1')
    plt.plot(np.arange(EM), slipping_traces[rep, :, 1], label='slipping strategy 2')
    plt.plot(np.arange(EM), guessing_traces[rep, :, 0], label='guessing strategy 1')
    plt.plot(np.arange(EM), guessing_traces[rep, :, 1], label='guessing strategy 2')
plt.ylim([0, 0.5])
plt.suptitle(f'slipping and guessing over all reps)')
# plt.legend()
plt.tight_layout()
plt.show()

# Bias
Bias_slipping_1 = 1 / N_REPEATS * np.sum([slipping_trace_avg[rep, 0] - true_slipping for rep in range(N_REPEATS)])
Bias_guessing_1 = 1 / N_REPEATS * np.sum([guessing_trace_avg[rep, 0] - true_guessing for rep in range(N_REPEATS)])
Bias_slipping_2 = 1 / N_REPEATS * np.sum([slipping_trace_avg[rep, 1] - true_slipping for rep in range(N_REPEATS)])
Bias_guessing_2 = 1 / N_REPEATS * np.sum([guessing_trace_avg[rep, 1] - true_guessing for rep in range(N_REPEATS)])
if not USE_REAL_DATA:
    Bias_pi = 1 / N_REPEATS * np.sum([pi_avg[rep] - (1 - pi_true[0]) for rep in range(N_REPEATS)])
else:
    Bias_pi = None
# MSE
MSE_slipping_1 = 1 / N_REPEATS * np.sum([(slipping_trace_avg[rep, 0] - true_slipping) ** 2 for rep in range(N_REPEATS)])
MSE_guessing_1 = 1 / N_REPEATS * np.sum([(guessing_trace_avg[rep, 0] - true_guessing) ** 2 for rep in range(N_REPEATS)])
MSE_slipping_2 = 1 / N_REPEATS * np.sum([(slipping_trace_avg[rep, 1] - true_slipping) ** 2 for rep in range(N_REPEATS)])
MSE_guessing_2 = 1 / N_REPEATS * np.sum([(guessing_trace_avg[rep, 1] - true_guessing) ** 2 for rep in range(N_REPEATS)])
if not USE_REAL_DATA:
    MSE_pi = 1 / N_REPEATS * np.sum([(pi_avg[rep] - (1 - pi_true[0])) ** 2 for rep in range(N_REPEATS)])
else:
    MSE_pi = None

# SD
SD_slipping = 1 / N_REPEATS * np.sum([np.std(slipping[rep, BI:]) for rep in range(N_REPEATS)])
SD_guessing = 1 / N_REPEATS * np.sum([np.std(guessing[rep, BI:]) for rep in range(N_REPEATS)])
SD_pi = 1 / N_REPEATS * np.sum([np.std(pi_hat[rep, BI:, 0]) for rep in range(N_REPEATS)])

print(f'Bias slipping 1: {Bias_slipping_1} & Bias slipping 2: {Bias_slipping_2}; Bias guessing 1: {Bias_guessing_1} & Bias guessing 2: {Bias_guessing_2}; Bias pi: {Bias_pi}')
print(f'MSE slipping 1: {MSE_slipping_1} & MSE slipping 2: {MSE_slipping_2}; MSE guessing 1: {MSE_guessing_1} & MSE guessing 2: {MSE_guessing_2}; MSE pi: {MSE_pi}')
print(f'SD slipping: {SD_slipping}; SD guessing: {SD_guessing}; SD pi: {SD_pi}')

if not USE_REAL_DATA:
    # Attribute Recovery Error Measure
    print('Error for all attributes:', np.sum(alpha_diff) / alpha_diff.size)

    # Marginal correct classification rate (for each attribute)
    for i, row in enumerate(alpha_diff.T):
        error = (np.sum(row) / n_examinees)
        print(f'error for attribute {i}: {error}')

    # proportion of examinees classified correctly for all K attributes
    counter = 0
    for column in alpha_diff:
        if np.all(column == 0):
            counter += 1

    classification_rate = counter / n_examinees
    print(f'proportion of examinees classified correctly for all K attributes: {classification_rate}')

    # proportion of examinees classified correctly for at least K-1 attributes
    counter = 0
    for i, row in enumerate(alpha_diff):
        counter_rows = 0
        for j, attribute in enumerate(row):
            if alpha_diff[i, j] == 0:
                counter_rows += 1

        if counter_rows >= n_attributes - 1:
            counter += 1

    classification_rate = counter / n_examinees
    print(f'proportion of examinees classified correctly for at least K-1 attributes: {classification_rate}')

    # proportion of examinees classified incorrectly for K-1 or K attributes
    counter = 0
    for i, row in enumerate(alpha_diff):
        counter_rows = 0
        for j, column in enumerate(alpha_diff.T):
            if alpha_diff[i, j] == 1:
                counter_rows += 1

        if counter_rows >= n_attributes - 1:
            counter += 1

    classification_rate = counter / n_examinees
    print(f'proportion of examinees classified incorrectly for at least K-1 attributes: {classification_rate}')

# p_MMS_log[:, :, rep] = np.log(p_MMS)
# logLik_MMS[rep] = np.sum(np.sum(np.log(p_MMS)))

#     # DINA model
#     p_DINA_1 = p_MMS_temp[:, :, 0]
#     p_DINA_2 = p_MMS_temp[:, :, 1]
#     p_DINA_1_log[:, :, rep] = score * np.log(p_DINA_1) + (1 - score) * np.log(1 - p_DINA_1)
#     logLik_DINA_1[rep] = np.sum(np.sum(p_DINA_1_log[:, :, rep]))
#     p_DINA_2_log[:, :, rep] = score * np.log(p_DINA_2) + (1 - score) * np.log(1 - p_DINA_2)
#     logLik_DINA_2[rep] = np.sum(np.sum(p_DINA_2_log[:, :, rep]))
#
#     # multi strategy model
#     for examinee in range(examn):
#         for item in range(itemn):
#             if eta[examinee, item, 0] - eta[examinee, item, 1] >= 0:
#                 eta_temp[examinee, item] = eta[examinee, item, 0]
#             else:
#                 eta_temp[examinee, item] = eta[examinee, item, 1]
#
#     p_MS_1 = ((np.ones(examn, 1) * slipping_hat[:itemn].T) ** eta_temp) * ((np.ones(examn, 1) * guessing_hat[:itemn].T) ** (1 - eta_temp))
#     p_MS_2 = ((np.ones(examn, 1) * slipping_hat[itemn:2 * itemn].T) ** eta_temp) * ((np.ones(examn, 1) * guessing_hat[itemn:2*itemn].T) ** (1 - eta_temp))
#     p_MS_1_log[:, :, rep] = score * np.log(p_MS_1) + (1 - score) * np.log(1 - p_MS_1)
#     logLik_MS_1[rep] = np.sum(np.sum(p_MS_1_log[:, :, rep]))
#     p_MS_2_log[:, :, rep] = score * np.log(p_MS_2) + (1 - score) * np.log(1 - p_MS_2)
#     logLik_MS_2[rep] = np.sum(np.sum(p_MS_2_log[:, :, rep]))
#
# DIC_DINA_1 = 2 * (-2 * np.mean(logLik_DINA_1)) - (-2 * np.max(logLik_DINA_1))
# DIC_DINA_2 = 2 * (-2 * np.mean(logLik_DINA_2)) - (-2 * np.max(logLik_DINA_2))
# DIC_MS_1 = 2 * (-2 * np.mean(logLik_MS_1)) - (-2 * np.max(logLik_MS_1))
# DIC_MS_2 = 2 * (-2 * np.mean(logLik_MS_2)) - (-2 * np.max(logLik_MS_2))
# DIC_MMS = 2 * (-2 * np.mean(logLik_MMS)) - (-2 * np.max(logLik_MMS))
#
# logCPO_DINA_1 = -np.log(np.sum(np.exp(-p_DINA_1_log), axis=2) / repititions)
# logCPO_DINA_2 = -np.log(np.sum(np.exp(-p_DINA_2_log), axis=2) / repetitions)
# logCPO_MS_1 = -np.log(np.sum(np.exp(-p_MS_1_log), axis=2) / repetitions)
# logCPO_MS_2 = -np.log(np.sum(np.exp(-p_MS_2_log), axis=2) / repetitions)
# logCPO_MMS = -np.log(np.sum(np.exp(-p_MMS_log), axis=2) / repetitions)
# LPML_DINA_1 = np.sum(np.sum(logCPO_DINA_1))
# LPML_DINA_2 = np.sum(np.sum(logCPO_DINA_2))
# LPML_MS_1 = np.sum(np.sum(logCPO_MS_1))
# LPML_MS_2 = np.sum(np.sum(logCPO_MS_2))
# LPML_MMS = np.sum(np.sum(logCPO_MMS))
