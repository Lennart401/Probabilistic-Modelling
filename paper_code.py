import os
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from scipy.stats import bernoulli, dirichlet, binom, beta, multinomial
from datetime import datetime


TINY_CONST = 1e-10
SHOW_PLOTS = False
SAVE_PLOTS = True
START_TIME = datetime.now().strftime("%Y%m%d-%H%M%S")
PLOTS_PATH = f'plots/{START_TIME}'
PLOT_SIZE = (22, 9)
PLOT_DPI = 300
SAVE_RESULTS = True
RESULTS_PATH = f'results/{START_TIME}'


if SAVE_PLOTS:
    os.makedirs(PLOTS_PATH)

if SAVE_RESULTS:
    os.makedirs(RESULTS_PATH)


plt.rcParams["figure.figsize"] = PLOT_SIZE
plt.rcParams["figure.dpi"] = PLOT_DPI


# Simulation parameters
lambda_1 = lambda_2 = 0.5
beta_1 = beta_2 = 0.01
beta_all = np.array([beta_1, beta_2])

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
n_examinees = 500

EM = 10000  # ???
BI = int(EM / 2)
repititions = 5  # number of repitions

# i : Examinee
# w : Strategy
# j : Item
# k : Attribute

# parameters
true_slipping = 0.3
true_guessing = 0.1

alpha = np.array(bernoulli.rvs(0.5, size=(n_examinees, n_attributes)))
mu = np.array(beta.rvs(lambda_1, lambda_2, size=n_strategies))
pi = dirichlet.rvs(beta_all, size=1).flatten()
s_c = np.ones((n_items, n_strategies)) * (1 - true_slipping)
g = np.ones((n_items, n_strategies)) * true_guessing
c = np.argmax(multinomial.rvs(n=1, p=pi, size=n_examinees), axis=1)

# build score from item response function
eta = np.zeros(shape=(n_examinees, n_items, n_strategies))
for i in range(n_examinees):
    for j in range(n_items):
        for m in range(n_strategies):
            eta[i, j, m] = np.sum([alpha[i, k] * q[j, k, m] for k in range(n_attributes)])

score = np.zeros(shape=(n_examinees, n_items))
for i in range(n_examinees):
    for j in range(n_items):
        score[i, j] = np.round(np.sum([pi[m] * (1 - true_slipping) ** eta[i, j, m] * true_guessing ** eta[i, j, m] for m in range(n_strategies)]))

# score = np.array(bernoulli.rvs(0.5, size=(n_examinees, n_items)))

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
    print('\nRepition: ', rep)
    for WWW in tqdm(range(EM)):
        # draw pi
        membership_counts = np.array([np.sum(c == m) for m in range(n_strategies)])
        # membership_counts = np.flip(membership_counts)
        pi = 1 - dirichlet.rvs(beta_all + membership_counts, size=1).flatten()
        pi_hat[rep, WWW] = pi

        # draw c (strategy membership parameter)
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

        # draw mu
        num_attributes = np.sum(alpha, axis=1)
        attr_sum = [np.sum(num_attributes[c == m]) for m in range(n_strategies)]

        mu = [
            np.random.beta(attr_sum[m] + lambda_1, alpha.size - attr_sum[m] + lambda_2)
            for m in range(n_strategies)
        ]
        mu_hat[rep, WWW] = mu

        # draw alpha (latent skill vector)
        # TODO parallelize
        for examinee in range(n_examinees):
            strategy_membership = int(c[examinee])
            # using n=1, this is a bernoulli draw
            # alpha_new = np.random.binomial(n=1, p=mu[strategy_membership], size=n_attributes)
            alpha_new = np.random.binomial(n=1, p=0.5, size=n_attributes)

            # Ratio of likelihoods for the new and prior alpha values
            LLa = binom.pmf(np.sum(alpha_new), n_attributes, mu[strategy_membership]) / \
                  binom.pmf(np.sum(alpha[examinee, :]), n_attributes, mu[strategy_membership])

            LLLa = 1
            # TODO parallelize / vectorize
            for item in range(n_items):
                eta = np.prod([alpha[examinee, attribute] ** q[item, attribute, strategy_membership] for attribute in range(n_attributes)])
                eta_new = np.prod([alpha_new[attribute] ** q[item, attribute, strategy_membership] for attribute in range(n_attributes)])

                tem = (s_c[item, strategy_membership] ** eta) * (g[item, strategy_membership] ** (1 - eta))
                tem_new = (s_c[item, strategy_membership] ** eta_new) * (g[item, strategy_membership] ** (1 - eta_new))

                if score[examinee, item] == 1:
                    temp = tem_new / np.maximum(tem, TINY_CONST)
                else:
                    temp = (1 - tem_new) / (1 - tem)

                LLLa *= temp

            p3 = LLa * LLLa
            if p3 >= np.random.rand():
                alpha[examinee, :] = alpha_new

            alpha_hat[rep, WWW, examinee] = alpha[examinee]

        # draw s and g
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

    slipping_hat = np.mean(slipping[rep, BI:])
    guessing_hat = np.mean(guessing[rep, BI:])
    # np.mean(np.mean(guessing[0, BI:], axis=0), axis=0)

    # some plotting
    plt.plot(np.arange(EM), np.mean(c_hat[rep], axis=1))
    plt.suptitle(f'c ({rep})')
    plt.tight_layout()
    if SAVE_PLOTS:
        plt.savefig(f'{PLOTS_PATH}/{rep}_c.png')
        plt.clf()
    if SHOW_PLOTS:
        plt.show()
    plt.close()

    slipping_trace = np.mean(1 - slipping[rep], axis=1)
    guessing_trace = np.mean(guessing[rep], axis=1)
    plt.plot(np.arange(EM), slipping_trace[:, 0], label='slipping strategy 1')
    plt.plot(np.arange(EM), slipping_trace[:, 1], label='slipping strategy 2')
    plt.plot(np.arange(EM), guessing_trace[:, 0], label='guessing strategy 1')
    plt.plot(np.arange(EM), guessing_trace[:, 1], label='guessing strategy 2')
    plt.suptitle(f'slipping and guessing ({rep})')
    plt.legend()
    plt.tight_layout()
    if SAVE_PLOTS:
        plt.savefig(f'{PLOTS_PATH}/{rep}_s_g.png')
        plt.clf()
    if SHOW_PLOTS:
        plt.show()
    plt.close()

    plt.plot(pi_hat[rep, :, 0], label='strategy 1')
    plt.plot(pi_hat[rep, :, 1], label='strategy 2')
    plt.suptitle(f'pi ({rep})')
    plt.legend()
    plt.tight_layout()
    if SAVE_PLOTS:
        plt.savefig(f'{PLOTS_PATH}/{rep}_pi.png')
        plt.clf()
    if SHOW_PLOTS:
        plt.show()
    plt.close()

    plt.plot(mu_hat[rep, :, 0], label='strategy 1')
    plt.plot(mu_hat[rep, :, 1], label='strategy 2')
    plt.suptitle(f'mu ({rep})')
    plt.legend()
    plt.tight_layout()
    if SAVE_PLOTS:
        plt.savefig(f'{PLOTS_PATH}/{rep}_mu.png')
        plt.clf()
    if SHOW_PLOTS:
        plt.show()
    plt.close()

    plt.matshow(alpha.T)
    plt.suptitle(f'alpha ({rep})')
    if SAVE_PLOTS:
        plt.savefig(f'{PLOTS_PATH}/{rep}_alpha.png')
        plt.clf()
    if SHOW_PLOTS:
        plt.show()
    plt.close()

    if SAVE_RESULTS:
        np.savez(f'{RESULTS_PATH}/{rep}.npz', alpha=alpha, c=c, pi=pi, mu=mu, s_c=s_c, g=g)

#     eta = np.zeros((examn, itemn, M))
#     eta_temp = zeros((examn, itemn))
#
#     p_MMS = zeros((examn, itemn))
#     p_MMS_temp = zeros((examn, itemn, M))
#
#     # mixture multiple strategy model
#     for strategy in range(k_total):
#         eta[:, :, strategy] = eta_temp[alpha, q[:, :, strategy]]
#         p_MMS_temp[:, :, strategy] = ((np.ones(examn, 1) * slipping_hat[itemn * (strategy-1) + 1:itemn * strategy].T) ** eta[:, :, strategy]) \
#             * ((np.ones(examn, 1) * guessing_hat[itemn * (strategy - 1) + 1:itemn * strategy].T ** eta[:, :, strategy]))
#         p_MMS = p_MMS + pai[strategy] * (p_MMS_temp[:, :, strategy] ** score * (1 - p_MMS_temp[:, :, strategy]) ** (1 - score))
#
#     p_MMS_log[:, :, rep] = np.log(p_MMS)
#     logLik_MMS[rep] = np.sum(np.sum(np.log(p_MMS)))
#
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
