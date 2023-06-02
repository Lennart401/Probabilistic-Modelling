import numpy as np
import scipy.stats as stats

examn = pass  # number of examinees
M = pass  # number of strategies
itemn = pass  # number of items
K = pass  # number of attributes
k_total = pass  # total number of attributes
EM = pass  # ???
repititions = pass  # number of repitions

# i : Examinee
# w : Strategy
# j : Item
# k : Attribute

# to fill out:
LL[strategy]
L[strategy]
yita[strategy]
c_hat[examinee, ???]
c[examinee]  # strategy membership parameter, c[i] is either 0 or 1

# given
alpha[examinee, attribute]
q[item, attribute, strategy]
mu[strategy]
score[examinee, item]
pai[strategy]

# functions needed
binomialPDF  # used in: draw c
binomialRandom  # used in: draw c, draw alpha

for rep in range(repititions):
    for WWW in range(EM):
        # draw c (strategy membership parameter)
        for examinee in range(examn):
            for strategy in range(M):
                LL[strategy] = 1
                
                for item in range(itemn):
                    yita[strategy] = 1
                    
                    for attribute in range(K):
                        yit = alpha[examinee, attribute] ** q[item, attribute, strategy]
                        yita[strategy] = yita[strategy] * yit
                        
                    # some kinda likelihood
                    tem = (s_c[item, strategy] ** yita[strategy]) * (g[item, strategy] ** 1-yita[strategy])
                    
                    if score[examinee, item] == 1:
                        p[strategy] = tem
                    else:
                        p[strategy] = 1 - tem
                    
                    LL[strategy] = LL[strategy] * p[strategy]  # likelihood
                
                L[strategy] = stats.binom.pmf(np.sum(alpha[examinee, :], axis=1), k_total, mu[strategy])  # prior
                Lc[strategy] = L[strategy] * LL[strategy] * pai[strategy]  # posterior
            
            # c_hat[examinee, :] = 10 ** 5 * Lc
            pp = Lc[1] / (Lc[0] + Lc[1])
            c[examinee] = np.random.binomial(1, pp)
            
        # draw alpha (latent skill vector)
        for examinee in range(examn):
            for attribute in range(k_total):
                alpha_new[examinee, attribute] = np.random.binomial(1, 0.5)
            
            strategy_membership = c[examinee]
            
            # Ratio of likelihoods for the new and prior alpha values
            LLa = stats.binom.pmf(np.sum(alpha_new[examinee, :], axis=1), k_total, mu[strategy_membership]) / binomialPDF(np.sum(alpha[examinee, :], axis=1), k_total, mu[strategy_membership])
            LLLa = 1
            
            for item in range(itemn):
                yitt = 1
                yita_new = 1
                
                for attribute in range(K):
                    yit = alpha[examinee, attribute] ** q[item, attribute, strategy_membership]
                    yitt = yitt * yit
                    yit_new = alpha_new[examinee, attribute] ** q[item, attribute, strategy_membership]
                    yita_new = yita_new * yit_new
                    
                temm = (s_c[item, strategy_membership] ** yitt) * (g[item, strategy_membership] ** (1 - yitt))
                tem_new = (s_c[item, strategy_membership] ** yita_new) * (g[item, strategy_membership] ** (1 - yita_new))
                
                if score[examinee, item] == 1:
                    p = temm
                    p_new = tem_new
                else:
                    p = 1 - temm
                    p_new = 1 - tem_new
                    
                temp = p_new / p
                LLLa = LLLa * temp
            
            p3 = LLa * LLLa
            t = np.random.rand()
            
            if p3 >= t:
                alpha[examinee, :] = alpha_new[examinee, :]
        
        # draw s and g
        for item in range(itemn):
            for strategy in range(M):
                temp1_g = 0.2
                temp2_g = 0.0
                temp1_s_c = 0.8
                temp1_s_c = 0.6
                g_new[item, strategy] = np.random.randint(np.floor(temp2_g * 1000), np.floor(temp1_g * 1000)) / 1000
                s_c_new[item, strategy] = np.random.randint(np.floor(temp2_s_c * 1000), np.floor(temp1_s_c * 1000)) / 1000
                
            LLb1 = 1
            LLb2 = 1
            
            for examinee in range(examn):
                strategy_membership = c[examinee]
                if strategy_membership == 0:
                    yitat = 1
                    
                    for attribute in range(K):
                        yit = alpha[examinee, attribute] ** q[item, attribute, strategy_membership]
                        yitat = yitat * yit
                        
                    tem = (s_c[item, strategy_membership] ** yitat) * (g[item, strategy_membership] ** (1 - yitat))
                    tem_new = (s_c_new[item, strategy_membership] ** yitat) * (g_new[item, strategy_membership] ** (1 - yitat))
                    
                    if score[examinee, item] == 1:
                        p = tem
                        p_new = tem_new
                    else:
                        p = 1 - tem
                        p_new = 1 -  tem_new
                    
                    temp1 = p_new / p
                    LLb1 = LLb1 * temp1
                    
                else:  # strategy_membership == 1
                    yitw = 1
                    
                    for attribute in range(K):
                        yiw = alpha[examinee, attribute] * q[item, attribute, strategy_membership]
                        yitw = yitw * yiw
                        
                    tem = (s_c[item, strategy_membership] ** yitw) * (g[item, strategy_membership] ** (1 - yitw))
                    tem_new = (s_c_new[item, strategy_membership] ** yitw) * (g[item, strategy_membership] ** (1 - yitw))
                    
                    if score[examinee, item] == 1:
                        p = tem
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
        
        slipping[:, WWW, rep] = s_c[:]
        guessing[:, WWW, rep] = g[:]
        
        # draw pai (pi)
        ss = np.sum(c)
        rr1 = ss + 0.01
        rr2 = examn - ss + 0.01
        
        dd1 = np.random.gamma(shape=rr1, scale=1.0)
        dd2 = np.random.gamma(shape=rr2, scale=1.0)
        
        pai1 = dd1 / (dd1 + dd2)
        pai2 = dd2 / (dd1 + dd2)
        pai = np.array([pai1, pai2])
        # pai_hat[:, WWW, rep] = pai.T
        
        # draw mu
        aa = np.sum(alpha, axis=1)
        rrt1 = 0
        rrt2 = 0
        
        for examinee in range(examn):
            wt = c[examinee]
            if wt == 0:
                rrt1 += aa[examinee]
            else
                rrt2 += aa[examinee]
        
        ddt1 = rrt1 + aw
        ddt2 = N * K + bw - rrt1
        dd3 = rrt2 + aw
        dd4 = N * K + bw - rrt2
        
        mu1 = np.random.beta(ddt1, ddt2)
        mu2 = np.random.beta(dd3, dd4)
        mu = np.array([mu1, mu2])
        
        # something else:
        if WWW >= EM - BI + 1:
            vvv = vvv + 1
            for examinee in range(examn):
                for attribute in range(k_total):
                    alpha_alpha[examinee, attribute] = alpha_alpha[examinee, attribute] + alpha[examinee, attribute]
                
            for item in range(itemn):
                for strategy in range(M):
                    s_c_s_c[item, strategy] = s_c_s_c[item, strategy] + s_c[item, strategy]
                    g_g[item, strategy] = g_g[item, strategy] + g[item, strategy]
                    
    Alpha = alpha_alpha / BI
    S_C = s_c_s_c / BI
    S = 1 - S_C
    G = g_g / BI
    
    for examinee in range(examn):
        for attribute in range(k_total):
            if Alpha(examinee, attribute) >= 0.5:
                Alpha2(examinee, attribute) = 1
            else:
                Alpha2(examinee, attribute) = 0
            
    guessing_hat = np.mean(guessing(:, (BI+1):, rep), axis=1)
    slipping_hat = np.mean(slipping(:, (BI+1):, rep), axis=1)
    
    eta = np.zeros((examn, itemn, M))
    eta_temp = zeros((examn, itemn))
    
    p_MMS = zeros((examn, itemn))
    p_MMS_temp = zeros((examn, itemn, M))
    
    # mixture multiple strategy model
    for strategy in range(k_total):
        eta[:, :, strategy] = eta_temp[alpha, q[:, :, strategy]]
        p_MMS_temp[:, :, strategy] = ((np.ones(examn, 1) * slipping_hat[itemn * (strategy-1) + 1:itemn * strategy].T) ** eta[:, :, strategy]) \
            * ((np.ones(examn, 1) * guessing_hat[itemn * (strategy - 1) + 1:itemn * strategy].T ** eta[:, :, strategy]))
        p_MMS = p_MMS + pai[strategy] * (p_MMS_temp[:, :, strategy] ** score * (1 - p_MMS_temp[:, :, strategy]) ** (1 - score))
        
    p_MMS_log[:, :, rep] = np.log(p_MMS)
    logLik_MMS[rep] = np.sum(np.sum(np.log(p_MMS)))
    
    # DINA model
    p_DINA_1 = p_MMS_temp[:, :, 0]
    p_DINA_2 = p_MMS_temp[:, :, 1]
    p_DINA_1_log[:, :, rep] = score * np.log(p_DINA_1) + (1 - score) * np.log(1 - p_DINA_1)
    logLik_DINA_1[rep] = np.sum(np.sum(p_DINA_1_log[:, :, rep]))
    p_DINA_2_log[:, :, rep] = score * np.log(p_DINA_2) + (1 - score) * np.log(1 - p_DINA_2)
    logLik_DINA_2[rep] = np.sum(np.sum(p_DINA_2_log[:, :, rep]))
    
    # multi strategy model
    for examinee in range(examn):
        for item in range(itemn):
            if eta[examinee, item, 0] - eta[examinee, item, 1] >= 0:
                eta_temp[examinee, item] = eta[examinee, item, 0]
            else:
                eta_temp[examinee, item] = eta[examinee, item, 1]
    
    p_MS_1 = ((np.ones(examn, 1) * slipping_hat[:itemn].T) ** eta_temp) * ((np.ones(examn, 1) * guessing_hat[:itemn].T) ** (1 - eta_temp))
    p_MS_2 = ((np.ones(examn, 1) * slipping_hat[itemn:2 * itemn].T) ** eta_temp) * ((np.ones(examn, 1) * guessing_hat[itemn:2*itemn].T) ** (1 - eta_temp))
    p_MS_1_log[:, :, rep] = score * np.log(p_MS_1) + (1 - score) * np.log(1 - p_MS_1)
    logLik_MS_1[rep] = np.sum(np.sum(p_MS_1_log[:, :, rep]))
    p_MS_2_log[:, :, rep] = score * np.log(p_MS_2) + (1 - score) * np.log(1 - p_MS_2)
    logLik_MS_2[rep] = np.sum(np.sum(p_MS_2_log[:, :, rep]))

DIC_DINA_1 = 2 * (-2 * np.mean(logLik_DINA_1)) - (-2 * np.max(logLik_DINA_1))
DIC_DINA_2 = 2 * (-2 * np.mean(logLik_DINA_2)) - (-2 * np.max(logLik_DINA_2))
DIC_MS_1 = 2 * (-2 * np.mean(logLik_MS_1)) - (-2 * np.max(logLik_MS_1))
DIC_MS_2 = 2 * (-2 * np.mean(logLik_MS_2)) - (-2 * np.max(logLik_MS_2))
DIC_MMS = 2 * (-2 * np.mean(logLik_MMS)) - (-2 * np.max(logLik_MMS))

logCPO_DINA_1 = -np.log(np.sum(np.exp(-p_DINA_1_log), axis=2) / repititions)
logCPO_DINA_2 = -np.log(np.sum(np.exp(-p_DINA_2_log), axis=2) / repetitions)
logCPO_MS_1 = -np.log(np.sum(np.exp(-p_MS_1_log), axis=2) / repetitions)
logCPO_MS_2 = -np.log(np.sum(np.exp(-p_MS_2_log), axis=2) / repetitions)
logCPO_MMS = -np.log(np.sum(np.exp(-p_MMS_log), axis=2) / repetitions)
LPML_DINA_1 = np.sum(np.sum(logCPO_DINA_1))
LPML_DINA_2 = np.sum(np.sum(logCPO_DINA_2))
LPML_MS_1 = np.sum(np.sum(logCPO_MS_1))
LPML_MS_2 = np.sum(np.sum(logCPO_MS_2))
LPML_MMS = np.sum(np.sum(logCPO_MMS))
