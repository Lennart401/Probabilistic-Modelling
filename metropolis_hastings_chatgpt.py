def metropolis_hastings(n_examinees, k_total, K, c, mu, q, s_c, g, score, alpha, itemn):
    for examinee in range(n_examinees):
        alpha_new = np.random.binomial(1, 0.5, k_total)
        strategy_membership = c[examinee]

        # Ratio of likelihoods for the new and prior alpha values
        LLa = stats.binom.pmf(np.sum(alpha_new), k_total, mu[strategy_membership]) / \
              stats.binom.pmf(np.sum(alpha[examinee, :]), k_total, mu[strategy_membership])

        LLLa = 1
        for item in range(itemn):
            # calculate eta_ijm for the current and new alphas
            eta = np.prod([alpha[examinee, attribute] ** q[item, attribute, strategy_membership] for attribute in range(K)])
            eta_new = np.prod([alpha_new[attribute] ** q[item, attribute, strategy_membership] for attribute in range(K)])
            
            # calculate the item response, (1-s_jm)^eta_ijm * g_jm^eta_ijm for the current and new eta
            temm = (s_c[item, strategy_membership] ** eta) * (g[item, strategy_membership] ** (1 - eta))
            tem_new = (s_c[item, strategy_membership] ** eta_new) * (g[item, strategy_membership] ** (1 - eta_new))

            if score[examinee, item] == 1:
                temp = tem_new / temm
            else:
                temp = (1 - tem_new) / (1 - temm)
                
            LLLa *= temp
        
        p3 = LLa * LLLa
        if p3 >= np.random.rand():
            alpha[examinee, :] = alpha_new

