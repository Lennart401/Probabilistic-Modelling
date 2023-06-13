import pymc as pm
import arviz as az
import pytensor.tensor as pt


lambda_1 = lambda_2 = 0.5
beta_1 = beta_2 = 0.01

sg_alpha = 1
sg_beta = 2
sg_lower = 0.1
sg_upper = 0.5
sg_scale = sg_lower - sg_upper

n_attributes = 5  # K attributes
n_strategies = 2  # M strategies
n_examinees = 10  # N examinees, i-th examinee
n_items = 30  # J items, j-th item

with pm.Model() as model:
    # Priors
    # pi = (pi_1, pi_2, ..., pi_M) ~ Dirichlet(beta_1, beta_2, ..., beta_M)
    pi = pm.Dirichlet('pi', a=[beta_1, beta_2])
    
    # c_i ~ Multinomial(1 | pi_1, pi_2, ..., pi_M)
    c = pm.Multinomial('c', n=1, p=pi, shape=(n_examinees, 2))
    c_category = pt.argmax(c, axis=-1)
    
    # mu_m ~ Beta(lambda_1, lambda_2)
    mu = pm.Beta('mu', alpha=lambda_1, beta=lambda_2, shape=n_strategies)
    
    # [ alpha_ik | c_i = m ] ~ Bernoulli(mu_m)
    alpha = pm.math.stack([pm.Bernoulli('alpha', p=mu[c_category]) for _ in range(n_attributes)]).T  # T?

    # s_jm ~ 4-Beta(v_s, t_s, a_s, b_s)
    s_raw = pm.Beta('s_raw', alpha=sg_alpha, beta=sg_beta, shape=(n_items, n_strategies))
    s = pm.Deterministic('s', sg_scale * s_raw + sg_lower)

    # g_jm ~ 4-Beta(v_g, t_g, a_g, b_g)
    g_raw = pm.Beta('g_raw', alpha=sg_alpha, beta=sg_beta, shape=(n_items, n_strategies))
    g = pm.Deterministic('g', sg_scale * g_raw + sg_lower)

    # likelihood
    

    # start the sampling
    trace = pm.sample(1000, tune=1000, random_seed=42)

az.summary(trace)
