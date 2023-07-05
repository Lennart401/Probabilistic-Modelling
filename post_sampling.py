import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def gelman_rubin(chains, burn_in=0):
    """
    Compute the Gelman-Rubin statistic R-hat for convergence of MCMC chains.

    Args:
        chains (list of 1D numpy arrays): A list containing MCMC chains

    Returns:
        float: The R-hat statistic
    """
    # Number of chains
    m = len(chains)

    # Number of iterations in each chain (assuming all chains have the same length)
    n = len(chains[0])

    # Cut the chains at the burn-in point
    chains = [chain[burn_in:] for chain in chains]

    # Calculate the within-chain variance
    W = np.mean([np.var(chain, ddof=1) for chain in chains])

    # Calculate the mean of each chain
    chain_means = [np.mean(chain) for chain in chains]

    # Calculate the variance of the chain means
    B = n * np.var(chain_means, ddof=1)

    # Estimate the variance as a weighted sum of within and between chain variance
    var_estimate = (1 - 1/n) * W + (1/n) * B

    # Compute the potential scale reduction factor
    R_hat = np.sqrt(var_estimate / W)

    return R_hat


def plot_chains(chains, color=None, ylim=None, title=None):
    """
    Plot trace and density plot of MCMC chains.

    Args:
        chains (list of 1D numpy arrays): A list containing MCMC chains
        ylim (tuple of floats): The y-axis limits for the trace plots
        title (str): The title of the plot

    """
    # Number of chains
    m = len(chains)

    # Create a figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Plot the trace plots on the left subplot
    for i, chain in enumerate(chains):
        axes[0].plot(chain, color=color, alpha=1/m)
    axes[0].set_title('Trace plots')
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Value')
    axes[0].legend()

    if ylim is not None:
        axes[0].set_ylim(ylim)

    # Plot the density plots on the right subplot
    for chain in chains:
        sns.kdeplot(chain, ax=axes[1], color=color, fill=True, alpha=2/m)
    axes[1].set_title('Density plots')
    axes[1].set_xlabel('Value')
    axes[1].set_ylabel('Density')

    if ylim is not None:
        axes[1].set_xlim(ylim)

    # Show plot
    if title is not None:
        plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_multiple_chains(chains_dict, colors=None, ylim=None, figure_title=None):
    """
    Plot trace and density plot of MCMC chains.

    Args:
        chains_dict (list of 1D numpy arrays): A list containing MCMC chains
        colors (dict): A dictionary mapping the title of each chain to a color
        ylim (tuple of floats): The y-axis limits for the trace plots
        figure_title (str): The title of the plot

    """
    # Number of chains
    m = len(next(iter(chains_dict.values())))

    # Create a figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Plot the trace plots on the left subplot
    for title, chains in chains_dict.items():
        color = colors[title] if colors is not None else None
        for chain in chains:
            axes[0].plot(chain, color=color, alpha=2/m)

    axes[0].set_title('Trace plots')
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Value')
    axes[0].legend()

    if ylim is not None:
        axes[0].set_ylim(ylim)

    # Plot the density plots on the right subplot
    for title, chains in chains_dict.items():
        color = colors[title] if colors is not None else None
        for chain in chains:
            sns.kdeplot(chain, ax=axes[1], color=color, fill=True, alpha=1/m)

    axes[1].set_title('Density plots')
    axes[1].set_xlabel('Value')
    axes[1].set_ylabel('Density')

    if ylim is not None:
        axes[1].set_xlim(ylim)

    # Show plot
    if figure_title is not None:
        plt.suptitle(figure_title)
    plt.tight_layout()
    plt.show()

# Load chains from folder
FOLDER = 'results/s1_500_examinees-20230704-000145'
N_CHAINS = 10
BURN_IN = 500

# Load from FOLDER/{chain}.npz
chains = []
for i in range(N_CHAINS):
    chains.append(np.load(f'{FOLDER}/{i}.npz'))

# For s, g, and pi, extract the chains
s_0_chains = [np.mean(chain['slipping'], axis=1)[:, 0] for chain in chains]
s_1_chains = [np.mean(chain['slipping'], axis=1)[:, 1] for chain in chains]
g_0_chains = [np.mean(chain['guessing'], axis=1)[:, 0] for chain in chains]
g_1_chains = [np.mean(chain['guessing'], axis=1)[:, 1] for chain in chains]
pi_0_chains = [chain['pi_hat'][:, 0] for chain in chains]
pi_1_chains = [chain['pi_hat'][:, 1] for chain in chains]
c_chains = [np.mean(chain['c_hat'], axis=1) for chain in chains]

# Compute R-hat for all the chains
print('R-hat for s_0:', gelman_rubin(s_0_chains, burn_in=BURN_IN))
print('R-hat for s_1:', gelman_rubin(s_1_chains, burn_in=BURN_IN))
print('R-hat for g_0:', gelman_rubin(g_0_chains, burn_in=BURN_IN))
print('R-hat for g_1:', gelman_rubin(g_1_chains, burn_in=BURN_IN))
print('R-hat for pi_0:', gelman_rubin(pi_0_chains, burn_in=BURN_IN))
print('R-hat for pi_1:', gelman_rubin(pi_1_chains, burn_in=BURN_IN))
print('R-hat for c:', gelman_rubin(c_chains, burn_in=BURN_IN))

# Plot all the chains
s_ylim = (0.6, 0.8)
g_ylim = (0.0, 0.3)

# s_color = 'deepskyblue'
s_color = 'yellowgreen'
g_color = 'cornflowerblue'

plot_multiple_chains(
    chains_dict={'Strategy A': s_0_chains, 'Strategy B': s_1_chains},
    colors={'Strategy A': s_color, 'Strategy B': g_color},
    ylim=s_ylim,
    figure_title='Slipping parameter')

plot_multiple_chains(
    chains_dict={'Strategy A': g_0_chains, 'Strategy B': g_1_chains},
    colors={'Strategy A': s_color, 'Strategy B': g_color},
    ylim=g_ylim, figure_title='Guessing parameter')

plot_multiple_chains(
    chains_dict={'Strategy A': pi_0_chains, 'Strategy B': pi_1_chains},
    colors={'Strategy A': s_color, 'Strategy B': g_color},
    figure_title='Mixing parameter')

plot_chains(c_chains, color='mediumorchid', title='Strategy membership parameter mean')



