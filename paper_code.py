import os
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from scipy.stats import bernoulli, binom, dirichlet, beta, multinomial, norm
from datetime import datetime


TINY_CONST = 1e-10
SHOW_PLOTS = False
SAVE_PLOTS = False
START_TIME = datetime.now().strftime("%Y%m%d-%H%M%S")
PLOTS_PATH = f'plots/{START_TIME}'
PLOT_SIZE = (22, 9)
PLOT_DPI = 50
SAVE_RESULTS = True
RESULTS_PATH = f'results/{START_TIME}'

# Script length variables
N_EXAMINEES = 500
N_ITERATIONS = 10000
N_REPEATS = 1

if int(os.environ.get('USE_TEST_MODE', 0)) == 1:
    print('Running in test mode...')
    SHOW_PLOTS = True
    SAVE_PLOTS = False
    SAVE_RESULTS = False
    PLOT_DPI = 100
    N_EXAMINEES = 100
    N_ITERATIONS = 1000
    N_REPEATS = 1


if SAVE_PLOTS:
    print(f'Creating directory {PLOTS_PATH}...')
    os.makedirs(PLOTS_PATH)

if SAVE_RESULTS:
    print(f'Creating directory {RESULTS_PATH}...')
    os.makedirs(RESULTS_PATH)


plt.rcParams["figure.figsize"] = PLOT_SIZE
plt.rcParams["figure.dpi"] = PLOT_DPI

# Simulation parameters
lambda_1 = lambda_2 = 0.5
beta_all = np.array([1, 1])
beta_sim = np.array([1, 1])

true_slipping = 0.3
true_guessing = 0.1

xi_0 = np.array([-0.95, -1.42, -0.66, 0.5, -0.05])
xi_1 = np.array([1.34, 1.22, 1.08, 1.11, 0.97])
theta_weight = 0.7

strategy_a_q_matrix = np.array([
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
    [0, 0, 1, 1, 1]
])

strategy_b_q_matrix = np.array([
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
    [1, 0, 0, 1, 0]
])

q = np.stack([strategy_a_q_matrix, strategy_b_q_matrix]).transpose(1, 2, 0)

n_attributes = strategy_a_q_matrix.shape[1]  # K attributes
n_strategies = len(np.unique(strategy_a_q_matrix))  # M strategies
n_items = strategy_a_q_matrix.shape[0]  # J items, j-th item
n_examinees = N_EXAMINEES

slipping_trace_avg = np.ones([N_REPEATS,n_strategies])
guessing_trace_avg = np.ones([N_REPEATS,n_strategies])
pi_avg = np.ones(N_REPEATS)

# ---------------------------------------------------------------------------------------------------------------------
# SIMULATION THE EXAMINEES TAKING A TEST
# ---------------------------------------------------------------------------------------------------------------------
alpha_sim = np.array(bernoulli.rvs(0.5, size=(n_examinees, n_attributes)))
pi = 1 - dirichlet.rvs(beta_sim, size=1).flatten()
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
        score[i, j] = bernoulli.rvs(np.sum([pi[m] * s_c[j, m] ** eta[i, j, m] * g[j, m] ** (1 - eta[i, j, m]) for m in range(n_strategies)]))

# ---------------------------------------------------------------------------------------------------------------------
# START WITH MCMC SAMPLING
# ---------------------------------------------------------------------------------------------------------------------
EM = N_ITERATIONS
BI = int(EM / 2)
repititions = N_REPEATS  # number of repitions

# track c, s, g
c_hat = np.zeros((repititions, EM, n_examinees))
pi_hat = np.zeros((repititions, EM, n_strategies))
mu_hat = np.zeros((repititions, EM, n_strategies))
slipping = np.zeros((repititions, EM, n_items, n_strategies))
guessing = np.zeros((repititions, EM, n_items, n_strategies))
alpha_hat = np.zeros((repititions, EM, n_examinees, n_attributes))

# after burn-in variables
bi_counter = 0
alpha_sum = np.zeros((n_examinees, n_attributes))
s_c_sum = np.zeros((n_items, n_strategies))
g_sum = np.zeros((n_items, n_strategies))


for rep in range(repititions):
    # Initial values
    alpha = np.array(bernoulli.rvs(0.5, size=(n_examinees, n_attributes)))
    mu = np.array(beta.rvs(lambda_1, lambda_2, size=n_strategies))
    pi = dirichlet.rvs(beta_all, size=1).flatten()
    s_c = 1 - np.array(beta.rvs(1, 2, size=(n_items, n_strategies)) * 0.4 + 0.1)
    g = np.array(beta.rvs(1, 2, size=(n_items, n_strategies)) * 0.4 + 0.1)
    c = np.argmax(multinomial.rvs(n=1, p=pi, size=n_examinees), axis=1)
    theta = np.array(beta.rvs(2, 2, size=n_examinees))


    # Start the MCMC
    print('\nRepition: ', rep)
    for WWW in tqdm(range(EM)):
        # -------------------------------------------------------------------------------------------------------------
        # GIBBS SAMPLING
        # -------------------------------------------------------------------------------------------------------------

        # draw pi using Gibbs Sampler (always accept, draw from conditional posterior)
        membership_counts = np.array([np.sum(c == m) for m in range(n_strategies)])
        # membership_counts = np.flip(membership_counts)
        pi = 1 - dirichlet.rvs(beta_all + membership_counts, size=1).flatten()
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
                    eta = np.prod([alpha[examinee, attribute] ** q[item, attribute, strategy] for attribute in range(n_attributes)])

                    # some kinda likelihood
                    tem = (s_c[item, strategy] ** eta) * (g[item, strategy] ** (1-eta))

                    #  p_ijm ^ u_ij * (1 - p_ijm) ^ (1 - u_ij)
                    p = tem if score[examinee, item] == 1 else 1 - tem
                    likelihood *= np.maximum(p, TINY_CONST) # likelihood

                # L[strategy] = binom.pmf(np.sum(alpha[examinee, :]), n_attributes, mu[strategy])  # prior
                prior = np.prod([bernoulli.pmf(alpha[examinee, attribute], mu[strategy]) for attribute in range(n_attributes)])
                Lc[strategy] = prior * likelihood * pi[strategy]  # posterior

            # c_hat[examinee, :] = 10 ** 5 * Lc
            # pp = Lc[1] / (Lc[0] + Lc[1])
            # print(WWW, examinee, pp, Lc, L, LL, p)
            c[examinee] = np.argmax(multinomial.rvs(1, Lc / np.sum(Lc)))  # np.random.binomial(1, pp)
            c_hat[rep, WWW, examinee] = c[examinee]

        # draw mu (strategy membership parameter), using Gibbs Sampler (always accept, draw from conditional posterior)
        num_attributes = np.sum(alpha, axis=1)
        attr_sum = [np.sum(num_attributes[c == m]) for m in range(n_strategies)]

        mu = [
            np.random.beta(attr_sum[m] + lambda_1, alpha[c == m].size - attr_sum[m] + lambda_2)
            # np.random.beta(attr_sum[m] + lambda_1, alpha.size - attr_sum[m] + lambda_2)
            for m in range(n_strategies)
        ]
        mu_hat[rep, WWW] = mu

        # -------------------------------------------------------------------------------------------------------------
        # METROPOLIS-HASTINGS SAMPLING
        # -------------------------------------------------------------------------------------------------------------

        # draw theta
        # TODO parallelize
        for examinee in range(n_examinees):
            alpha_param = 2
            beta_param = 2
            if theta[examinee] >= 0.5:
                alpha_param += theta[examinee]
            else:
                beta_param += theta[examinee]
            theta_new = beta.rvs(alpha_param, beta_param)

            # Ratio of likelihoods for the new and prior theta values
            tem = np.add(xi_0, xi_1 * theta[examinee])
            p_alpha_theta = np.exp(tem) / (1 + np.exp(tem))

            tem_new = np.add(xi_0, xi_1 * theta_new)
            p_alpha_theta_new = np.exp(tem_new) / (1 + np.exp(tem_new))
            likelihood = (np.prod(p_alpha_theta_new) * beta.pdf(theta_new,alpha_param,beta_param)) / (np.prod(p_alpha_theta) * beta.pdf(theta[examinee],alpha_param,beta_param))

            if likelihood >= np.random.rand():
                theta[examinee] = theta_new

        # draw alpha (latent skill vector), using Metropolis-Hastings (accept/reject)
        # TODO parallelize
        for examinee in range(n_examinees):
            strategy_membership = int(c[examinee])
            # using n=1, this is a bernoulli draw
            # alpha_new = np.random.binomial(n=1, p=mu[strategy_membership], size=n_attributes)
            alpha_new = np.random.binomial(n=1, p=0.5, size=n_attributes)

            probability = theta_weight * mu[strategy_membership] + (1 - theta_weight) * theta[examinee]
            # Ratio of likelihoods for the new and prior alpha values
            LLa = binom.pmf(np.sum(alpha_new), n_attributes, probability) / \
                  binom.pmf(np.sum(alpha[examinee, :]), n_attributes, probability)
            # LLa = binom.pmf(np.sum(alpha[examinee, :]), n_attributes, mu[strategy_membership]) / \
            #       binom.pmf(np.sum(alpha_new), n_attributes, mu[strategy_membership])

            LLLa = 1
            # TODO parallelize / vectorize
            for item in range(n_items):
                eta = np.prod([alpha[examinee, attribute] ** q[item, attribute, strategy_membership] for attribute in range(n_attributes)])
                eta_new = np.prod([alpha_new[attribute] ** q[item, attribute, strategy_membership] for attribute in range(n_attributes)])

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

            likelihood = np.ones(n_strategies)

            # { PROD i=1 to N: [ p_ijm ^ u_ij * (1 - p_ijm) ^ (1 - u_ij) ] ^ c_j ] ^ I(c_i = m) } Beta(s_jm) x Beta(g_jm)
            # TODO parallelize / vectorize
            for examinee in range(n_examinees):
                strategy_membership = int(c[examinee])
                eta = np.prod([alpha[examinee, attribute] ** q[item, attribute, strategy_membership] for attribute in range(n_attributes)])

                tem = (s_c[item, strategy_membership] ** eta) * (g[item, strategy_membership] ** (1 - eta))
                tem_new = (s_c_new[item, strategy_membership] ** eta) * (g_new[item, strategy_membership] ** (1 - eta))

                if score[examinee, item] == 1:
                    p = np.maximum(tem, TINY_CONST)
                    p_new = tem_new
                else:
                    p = 1 - tem
                    p_new = 1 - tem_new

                likelihood[strategy_membership] *= (p_new / p)

            for strategy in range(n_strategies):
                if likelihood[strategy] >= np.random.rand():
                    g[item, strategy] = g_new[item, strategy]
                    s_c[item, strategy] = s_c_new[item, strategy]

        slipping[rep, WWW] = s_c
        guessing[rep, WWW] = g

        # If were are past the burn-in period, the sum alpha, s and g to get an average value of them
        if WWW >= EM - BI:
            bi_counter += 1
            # TODO parallelize / vectorize
            for examinee in range(n_examinees):
                for attribute in range(n_attributes):
                    alpha_sum[examinee, attribute] += alpha[examinee, attribute]

            for item in range(n_items):
                for strategy in range(n_strategies):
                    s_c_sum[item, strategy] += s_c[item, strategy]
                    g_sum[item, strategy] += g[item, strategy]

    alpha_avg = alpha_sum / BI
    s_c_avg = s_c_sum / BI
    s_avg = 1 - s_c_avg
    g_avg = g_sum / BI

    alpha_final = np.round(alpha_avg)

    eta = np.zeros(shape=(n_examinees, n_items, n_strategies))
    for i in range(n_examinees):
        for j in range(n_items):
            for m in range(n_strategies):
                eta[i, j, m] = np.prod([alpha[i, k] ** q[j, k, m] for k in range(n_attributes)])

    p_MMS = np.zeros((n_examinees, n_items))
    score_pred = np.zeros((n_examinees, n_items))

    slipping_trace = np.mean(1 - slipping[rep], axis=1)
    guessing_trace = np.mean(guessing[rep], axis=1)

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
            p_MMS[examinee, item] = np.sum(pi_mean * (p_ijm ** score[examinee, item]) * ((1 - p_ijm) ** (1 - score[examinee, item])))
            score_pred[examinee, item] = np.sum(pi_mean * p_ijm)
            # p_MMS[examinee, item] = np.sum(p_ijm)

    # -----------------------------------------------------------------------------------------------------------------
    # PLOTTING
    # -----------------------------------------------------------------------------------------------------------------

    # c - strategy membership
    plt.plot(np.arange(EM), np.mean(c_hat[rep], axis=1))
    plt.suptitle(f'c ({rep})')
    plt.tight_layout()
    if SAVE_PLOTS:
        plt.savefig(f'{PLOTS_PATH}/{rep}_c.png')
    if SHOW_PLOTS:
        plt.show()
    plt.clf()
    plt.close()

    # s and g - slipping and guessing parameters
    plt.plot(np.arange(EM), slipping_trace[:, 0], label='slipping strategy 1')
    plt.plot(np.arange(EM), slipping_trace[:, 1], label='slipping strategy 2')
    plt.plot(np.arange(EM), guessing_trace[:, 0], label='guessing strategy 1')
    plt.plot(np.arange(EM), guessing_trace[:, 1], label='guessing strategy 2')
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
    plt.suptitle(f'pi ({rep})')
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
    plt.suptitle(f'mu ({rep})')
    plt.legend()
    plt.tight_layout()
    if SAVE_PLOTS:
        plt.savefig(f'{PLOTS_PATH}/{rep}_mu.png')
    if SHOW_PLOTS:
        plt.show()
    plt.clf()
    plt.close()

    # plotting matrices
    cmap = 'plasma'
    colorbar_location = 'right'

    # alpha (difference, simulated and sampled) - latent skill vector
    alpha_diff = np.abs(alpha_sim - alpha)
    print(f'weight for mu {theta_weight}: {np.sum(alpha_diff)/(n_attributes * n_examinees)}')

    fig, ax = plt.subplots(nrows=3, ncols=1)
    alpha_diff_ax = ax[0].matshow(1 - alpha_diff.T, aspect='auto', cmap='PiYG')
    alpha_sim_ax = ax[1].matshow(alpha_sim.T, aspect='auto', cmap=cmap)
    alpha_ax = ax[2].matshow(alpha.T, aspect='auto', cmap=cmap)
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
    fig, ax = plt.subplots(nrows=3, ncols=1)
    score_ax = ax[0].matshow(score.T, aspect='auto', cmap=cmap)
    score_pred_ax = ax[1].matshow(score_pred.T, aspect='auto', cmap=cmap)
    p_MMS_ax = ax[2].matshow(p_MMS.T, aspect='auto', cmap=cmap)
    fig.colorbar(score_ax, location=colorbar_location)
    fig.colorbar(score_pred_ax, location=colorbar_location)
    fig.colorbar(p_MMS_ax, location=colorbar_location)
    plt.suptitle(f'score vs. p_MMS ({rep})')
    plt.tight_layout()
    if SAVE_PLOTS:
        plt.savefig(f'{PLOTS_PATH}/{rep}_p_MMS.png')
    if SHOW_PLOTS:
        plt.show()
    plt.clf()
    plt.close()

    # Theta values
    print(theta)

    if SAVE_RESULTS:
        np.savez(f'{RESULTS_PATH}/{rep}.npz', alpha=alpha, c=c, pi=pi, mu=mu, s_c=s_c, g=g)


    # Keep track of avg values over repetitions 
    pi_avg[rep] = pi_mean[0]
    slipping_trace_avg[rep] = np.mean(slipping_trace, axis = 0)
    guessing_trace_avg[rep] = np.mean(guessing_trace, axis = 0)

# -----------------------------------------------------------------------------------------------------------------
# Error Measures 
# -----------------------------------------------------------------------------------------------------------------
# Bias 
Bias_slipping_1 = 1/N_REPEATS * np.sum([slipping_trace_avg[rep,0] - true_slipping for rep in range(N_REPEATS)])
Bias_guessing_1 = 1/N_REPEATS * np.sum([guessing_trace_avg[rep,0] - true_guessing for rep in range(N_REPEATS)])
Bias_slipping_2 = 1/N_REPEATS * np.sum([slipping_trace_avg[rep,1] - true_slipping for rep in range(N_REPEATS)])
Bias_guessing_2 = 1/N_REPEATS * np.sum([guessing_trace_avg[rep,1] - true_guessing for rep in range(N_REPEATS)])
Bias_pi = 1/N_REPEATS * np.sum([pi_avg[rep] - (1-pi_true[0]) for rep in range(N_REPEATS)])
# MSE
MSE_slipping_1 = 1/N_REPEATS * np.sum([(slipping_trace_avg[rep,0] - true_slipping)**2 for rep in range(N_REPEATS)])
MSE_guessing_1 = 1/N_REPEATS * np.sum([(guessing_trace_avg[rep,0] - true_guessing)**2 for rep in range(N_REPEATS)])
MSE_slipping_2 = 1/N_REPEATS * np.sum([(slipping_trace_avg[rep,1] - true_slipping)**2 for rep in range(N_REPEATS)])
MSE_guessing_2 = 1/N_REPEATS * np.sum([(guessing_trace_avg[rep,1] - true_guessing)**2 for rep in range(N_REPEATS)])
MSE_pi = 1/N_REPEATS * np.sum([(pi_avg[rep] - (1 -pi_true[0]))**2 for rep in range(N_REPEATS)])
# SD 
SD_slipping = 1/N_REPEATS * np.sum(np.std(slipping[rep,BI:]) for rep in range(N_REPEATS))
SD_guessing = 1/N_REPEATS * np.sum(np.std(guessing[rep,BI:]) for rep in range(N_REPEATS))
SD_pi = 1/N_REPEATS * np.sum(np.std(pi_hat[rep,BI:,0]) for rep in range(N_REPEATS))

print(f'Bias slipping 1: {Bias_slipping_1} & Bias slipping 2: {Bias_slipping_2}; Bias guessing 1: {Bias_guessing_1} & Bias guessing 2: {Bias_guessing_2}; Bias pi: {Bias_pi}')
print(f'MSE slipping 1: {MSE_slipping_1} & MSE slipping 2: {MSE_slipping_2}; MSE guessing 1: {MSE_guessing_1} & MSE guessing 2: {MSE_guessing_2}; MSE pi: {MSE_pi}')
print(f'SD slipping: {SD_slipping}; SD guessing: {SD_guessing}; SD pi: {SD_pi}')

# Attribute Recovery Error Measure 
# Marginal correct classification rate (for each attribute)
for i, row in enumerate(alpha_diff.T):
    error = (np.sum(row)/n_examinees)
    print(f'error for attribute {i}: {error}')

# proportion of examinees classified correctly for all K attributes
counter = 0
for column in alpha_diff:
    if np.all(column == 0):
        counter += 1
    else: 
        pass
classification_rate = counter / N_EXAMINEES
print(f'proportion of examinees classified correctly for all K attributes: {classification_rate}')

# proportion of examinees classified correctly for at least K-1 attributes
counter = 0
for i,row in enumerate(alpha_diff):
    counter_rows = 0
    for j,attribute in enumerate(row):
        if alpha_diff[i, j] == 0:
            counter_rows += 1
        else: 
            pass
    if counter_rows >= n_attributes-1:
        counter += 1 
    else: 
        pass
classification_rate = counter / N_EXAMINEES
print(f'proportion of examinees classified correctly for at least K-1 attributes: {classification_rate}')

# proportion of examinees classified incorrectly for K-1 or K attributes
counter = 0
for i, row in enumerate(alpha_diff):
    counter_rows = 0
    for j,column in enumerate(alpha_diff.T):
        if alpha_diff[i, j] == 1:
            counter_rows += 1
        else: 
            pass
    if counter_rows >= n_attributes-1:
        counter += 1 
    else: 
        pass
classification_rate = counter / N_EXAMINEES
print(f'proportion of examinees classified incorrectly for at least K-1 attributes: {classification_rate}')

    # score and alpha together in one plot
    # fig, ax = plt.subplots(nrows=2, ncols=2)
    # alpha_sim_ax = ax[0, 0].matshow(alpha_sim.T, aspect='auto', cmap=cmap)
    # alpha_ax = ax[0, 1].matshow(alpha.T, aspect='auto', cmap=cmap)
    # score_ax = ax[1, 0].matshow(score.T, aspect='auto', cmap=cmap)
    # p_MMS_ax = ax[1, 1].matshow(p_MMS.T, aspect='auto', cmap=cmap)
    # fig.colorbar(alpha_sim_ax, location='bottom')
    # fig.colorbar(alpha_ax, location='bottom')
    # fig.colorbar(score_ax, location='bottom')
    # fig.colorbar(p_MMS_ax, location='bottom')
    # plt.suptitle(f'alpha vs. score ({rep})')
    # plt.tight_layout()
    # if SAVE_PLOTS:
    #     plt.savefig(f'{PLOTS_PATH}/{rep}_alpha.png')
    #     plt.clf()
    # if SHOW_PLOTS:
    #     plt.show()
    # plt.close()

    # SAVING THE RESULTS

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
