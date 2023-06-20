import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from scipy.stats import bernoulli, dirichlet, binom, beta


TINY_CONST = 1e-100


# Simulation parameters
lambda_1 = lambda_2 = 0.5
beta_1 = beta_2 = 0.01

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
q_ext = q[np.newaxis, :, :, :]

n_attributes = strategy_a_q_matrix.shape[1]  # K attributes
n_strategies = len(np.unique(strategy_a_q_matrix))  # M strategies
n_items = strategy_a_q_matrix.shape[0]  # J items, j-th item
n_examinees = 10

EM = 30  # ???
BI = int(EM / 2)
repititions = 1  # number of repitions

# i : Examinee
# w : Strategy
# j : Item
# k : Attribute

# TODO !!!! THIS NEEDS TO BE UPDATED!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
score = np.array(bernoulli.rvs(0.5, size=(n_examinees, n_items)))

# parameters
alpha = np.array(bernoulli.rvs(0.5, size=(n_examinees, n_attributes)))
mu = np.array(beta.rvs(lambda_1, lambda_2, size=n_strategies))
pi = dirichlet.rvs([beta_1, beta_2], size=1).flatten()
s_c = np.ones((n_items, n_strategies)) * (1 - 0.5)
g = np.ones((n_items, n_strategies)) * 0.5

# track c, s, g
c_hat = np.zeros((repititions, EM, n_examinees))
pi_hat = np.zeros((repititions, EM, n_strategies))
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
        # draw c (strategy membership parameter)
        c = np.zeros(n_examinees)
        for examinee in range(n_examinees):
            LL = np.ones(n_strategies)
            Lc = np.ones(n_strategies)
            L = np.ones(n_strategies)
            eta = np.ones(n_strategies)
            p = np.ones(n_strategies)

            for strategy in range(n_strategies):
                for item in range(n_items):
                    eta[strategy] = np.prod([alpha[examinee, attribute] ** q[item, attribute, strategy] for attribute in range(n_attributes)])

                    # some kinda likelihood
                    tem = (s_c[item, strategy] ** eta[strategy]) * (g[item, strategy] ** (1-eta[strategy]))

                    #  p_ijm ^ u_ij * (1 - p_ijm) ^ (1 - u_ij)
                    p[strategy] = tem if score[examinee, item] == 1 else 1 - tem

                    LL[strategy] = LL[strategy] * (p[strategy] + TINY_CONST) # likelihood

                L[strategy] = binom.pmf(np.sum(alpha[examinee, :]), n_attributes, mu[strategy])  # prior
                Lc[strategy] = L[strategy] * LL[strategy] * pi[strategy]  # posterior

            # c_hat[examinee, :] = 10 ** 5 * Lc
            pp = Lc[1] / (Lc[0] + Lc[1])
            # print(WWW, examinee, pp, Lc, L, LL, p)
            c[examinee] = np.random.binomial(1, pp)
            c_hat[rep, WWW, examinee] = c[examinee]

        # draw alpha (latent skill vector)
        for examinee in range(n_examinees):
            alpha_new = np.random.binomial(1, 0.5, n_attributes)
            strategy_membership = int(c[examinee])

            # Ratio of likelihoods for the new and prior alpha values
            LLa = binom.pmf(np.sum(alpha_new), n_attributes, mu[strategy_membership]) / \
                  binom.pmf(np.sum(alpha[examinee, :]), n_attributes, mu[strategy_membership])

            LLLa = 1
            for item in range(n_items):
                yitt = np.prod([alpha[examinee, attribute] ** q[item, attribute, strategy_membership] for attribute in range(n_attributes)])
                yita_new = np.prod([alpha_new[attribute] ** q[item, attribute, strategy_membership] for attribute in range(n_attributes)])

                temm = (s_c[item, strategy_membership] ** yitt) * (g[item, strategy_membership] ** (1 - yitt))
                tem_new = (s_c[item, strategy_membership] ** yita_new) * (g[item, strategy_membership] ** (1 - yita_new))

                if score[examinee, item] == 1:
                    temp = tem_new / (temm + TINY_CONST)
                else:
                    temp = (1 - tem_new) / (1 - temm)

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
                # TODO replace this with a 4 beta distribution
                temp1_g = 0.2
                temp2_g = 0.0
                temp1_s_c = 0.8
                temp2_s_c = 0.6
                g_new[item, strategy] = np.random.randint(np.floor(temp2_g * 1000), np.floor(temp1_g * 1000)) / 1000
                s_c_new[item, strategy] = np.random.randint(np.floor(temp2_s_c * 1000), np.floor(temp1_s_c * 1000)) / 1000

            LLb1 = 1
            LLb2 = 1

            for examinee in range(n_examinees):
                strategy_membership = int(c[examinee])
                if strategy_membership == 0:
                    yitat = 1

                    for attribute in range(n_attributes):
                        yit = alpha[examinee, attribute] ** q[item, attribute, strategy_membership]
                        yitat = yitat * yit

                    tem = (s_c[item, strategy_membership] ** yitat) * (g[item, strategy_membership] ** (1 - yitat))
                    tem_new = (s_c_new[item, strategy_membership] ** yitat) * (g_new[item, strategy_membership] ** (1 - yitat))

                    if score[examinee, item] == 1:
                        p = (tem + TINY_CONST)
                        p_new = tem_new
                    else:
                        p = 1 - tem
                        p_new = 1 - tem_new

                    temp1 = p_new / p
                    LLb1 = LLb1 * temp1

                else:  # strategy_membership == 1
                    yitw = 1

                    for attribute in range(n_attributes):
                        yiw = alpha[examinee, attribute] * q[item, attribute, strategy_membership]
                        yitw = yitw * yiw

                    tem = (s_c[item, strategy_membership] ** yitw) * (g[item, strategy_membership] ** (1 - yitw))
                    tem_new = (s_c_new[item, strategy_membership] ** yitw) * (g[item, strategy_membership] ** (1 - yitw))

                    if score[examinee, item] == 1:
                        p = (tem + TINY_CONST)
                        p_new = tem_new
                    else:
                        p = 1 - tem
                        p_new = 1 - tem_new

                    temp2 = p_new / p
                    LLb2 = LLb2 * temp2

            t = np.random.rand()
            if LLb1 >= t:
                g[item, 0] = g_new[item, 0]
                s_c[item, 0] = s_c_new[item, 0]

            t = np.random.rand()
            if LLb2 >= t:
                g[item, 1] = g_new[item, 1]
                s_c[item, 1] = s_c_new[item, 1]

        slipping[rep, WWW] = s_c
        guessing[rep, WWW] = g

        # draw pi
        ss = np.sum(c)
        rr1 = ss + 0.01
        rr2 = n_examinees - ss + 0.01

        dd1 = np.random.gamma(shape=rr1, scale=1.0)
        dd2 = np.random.gamma(shape=rr2, scale=1.0)

        pi1 = dd1 / (dd1 + dd2)
        pi2 = dd2 / (dd1 + dd2)
        pi = np.array([pi1, pi2])
        pi_hat[rep, WWW] = pi

        # draw mu
        aa = np.sum(alpha, axis=1)
        rrt1 = 0
        rrt2 = 0

        for examinee in range(n_examinees):
            wt = int(c[examinee])
            if wt == 0:
                rrt1 += aa[examinee]
            else:
                rrt2 += aa[examinee]

        ddt1 = rrt1 + lambda_1
        ddt2 = n_examinees * n_attributes + lambda_2 - rrt1
        dd3 = rrt2 + lambda_1
        dd4 = n_examinees * n_attributes + lambda_2 - rrt2

        mu1 = np.random.beta(ddt1, ddt2)
        mu2 = np.random.beta(dd3, dd4)
        mu = np.array([mu1, mu2])

        # If were are past the burn-in period, the sum alpha, s and g to get an average value of them
        if WWW >= EM - BI:
            bi_counter += 1
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

    guessing_hat = np.mean(guessing[rep, BI:])
    slipping_hat = np.mean(slipping[rep, BI:])
    # np.mean(np.mean(guessing[0, BI:], axis=0), axis=0)

    # some plotting
    slipping_trace = np.mean(1 - slipping[rep], axis=1)
    guessing_trace = np.mean(guessing[rep], axis=1)
    plt.plot(np.arange(EM), slipping_trace[:, 0], label='slipping strategy 0')
    plt.plot(np.arange(EM), slipping_trace[:, 1], label='slipping strategy 1')
    plt.plot(np.arange(EM), guessing_trace[:, 0], label='guessing strategy 0')
    plt.plot(np.arange(EM), guessing_trace[:, 1], label='guessing strategy 1')
    plt.legend()
    plt.show()

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