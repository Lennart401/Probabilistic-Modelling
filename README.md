# Exploring and Extending Bayesian MCMC Techniques in Multiple Strategy Problem Solving

This is the code corresponding to the paper "Exploring and Extending Bayesian MCMC Techniques in Multiple Strategy Problem Solving", written within the course Probabilistic Modelling, held by Prof. Dr. Burkhardt Funk, at Leuphana University Lüneburg. The paper can be accessed <span style="text-decoration: line-through">[here]()</span> [TO BE ADDED].

## Abstract
This paper extends Bayesian Markov Chain Monte Carlo (MCMC) methods for cognitive diagnosis models (CDMs) in educational psychology. It refines the original MMS-DINA model by Zhang et al. (2021) and proposes an extension using general knowledge parameters from De La Torre and Douglas (2004). Through simulation studies and real data, both revised models were found to outperform the original in accuracy, parameter recovery, and convergence. Concerns are raised about the models’ robustness and adaptability to real-world scenarios, the extended model’s marginal improvement not justifying its increased computational demands and need for further research is highlighted.

## Code Usage
The parts of sampling code are a python adaption of the original MATLAB code by Zhang et al. (2021). It is structured into two main files:

- `simulation_and_sampling.py`: This code optionally runs the simulation and performs the sampling procedure. For this, the configuration variables in the beginning of the file need to be adjusted. During testing, the environment variable `USE_TEST_MODE` should be set to `1`, which sets shorter sampling durations etc. Simulation data can be saved/loaded by setting the `SIMULATION_DATA` to a valid path. If no file exists for that path, data is simulated, otherwise it is loaded. Enabling the real data flag loads the fraction subtraction dataset.
- `post_sampling.py`: This code has been written solely for this project, and it generated performance stats and trace/density plots for the results generated from the `simulation_and_sampling.py` file.

## References

De La Torre, Jimmy and Jeffrey A. Douglas (2004). “Higher- order latent trait models for cognitive diagnosis”. In: Psy- chometrika 69, pp. 333–353.

Zhang, Jiwei et al. (2021). “Exploring Multiple Strategic Problem Solving Behaviors in Educational Psychology Re- search by Using Mixture Cognitive Diagnosis Model”. In: Frontiers in psychology 12, pp. 1–12.