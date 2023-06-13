import pymc as pm
import arviz as az

# Example data: heights of 10 individuals in centimeters
data = [160, 170, 162, 172, 168, 174, 176, 165, 163, 171]

# Building the model
with pm.Model() as height_model:

    # Priors
    mean = pm.Normal('mean', mu=170, sigma=10)
    std_dev = pm.HalfNormal('std_dev', sigma=10)

    # Likelihood
    heights = pm.Normal('heights', mu=mean, sigma=std_dev, observed=data)

    # Sampling from the posterior distributions
    trace = pm.sample(1000, tune=1000)

# Visualizing the results
az.plot_trace(trace)
az.plot_posterior(trace)

# Summary of the results
print(az.summary(trace))
