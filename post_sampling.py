import os
import pathlib

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


RESULTS_FOLDER = pathlib.Path('results')
RESULT_PLOTS_FOLDER = pathlib.Path('result_plots')


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


def plot_chains(chains_dict, colors=None, ylim=None, figure_title=None, save_path=None, show_plots=True,
                split_burn_in=True):
    """
    Plot trace and density plot of MCMC chains.

    Args:
        chains_dict (list of 1D numpy arrays): A list containing MCMC chains
        colors (dict): A dictionary mapping the title of each chain to a color
        ylim (tuple of floats): The y-axis limits for the trace plots
        figure_title (str): The title of the plot
        save_path: The path to save the plot
        show_plots (bool): Whether to show the plot
        split_burn_in (bool): If True, then split the chains at the burn-in point and plot the two parts separately

    """
    # Print what is being plotted
    print(f'Plotting {figure_title}')

    # Number of chains
    m = len(next(iter(chains_dict.values())))

    def plot_chains_on_axes(axes, start, end, show_burn_in_label=False):
        # Plot the trace plots on the left subplot
        for title, chains in chains_dict.items():
            color = colors[title] if colors is not None else None
            for i, chain in enumerate(chains):
                label = title if i == 0 else None
                axes[0].plot(np.arange(start, end), chain[start:end], color=color, alpha=2/m, label=label)

        axes[0].set_title(f'Trace plots{" (burn-in)" if show_burn_in_label else ""}')
        axes[0].set_xlabel('Iteration')
        axes[0].set_ylabel('Value')
        axes[0].legend()

        if ylim is not None:
            axes[0].set_ylim(ylim)

        # Plot the density plots on the right subplot
        for title, chains in chains_dict.items():
            color = colors[title] if colors is not None else None
            for chain in chains:
                sns.kdeplot(chain[start:end], ax=axes[1], color=color, fill=True, alpha=1/m)

        axes[1].set_title(f'Density plots{" (burn-in)" if show_burn_in_label else ""}')
        axes[1].set_xlabel('Value')
        axes[1].set_ylabel('Density')

        if ylim is not None:
            axes[1].set_xlim(ylim)

    # Create a figure with two subplots
    fig, axs = plt.subplots(nrows=2 if split_burn_in else 1, ncols=2, figsize=(18, 10))
    start = 0
    end = len(next(iter(chains_dict.values()))[0])

    if split_burn_in:
        half = int((end - start) / 2)
        plot_chains_on_axes(axs[0, :], start, half, show_burn_in_label=True)
        plot_chains_on_axes(axs[1, :], half, end, show_burn_in_label=False)

    else:
        plot_chains_on_axes(axs, start, end, show_burn_in_label=False)

    # Show plot
    if figure_title is not None:
        plt.suptitle(figure_title)
    plt.tight_layout()

    if save_path is not None:
        os.makedirs(save_path.resolve().parent, exist_ok=True)
        plt.savefig(save_path.resolve())

    if show_plots:
        plt.show()


def process_study(study_folder, save_plots=False, skip_plots=False, n_chains='auto', burn_in=None, real_data=False,
                  show_plots=True):
    # Load chains
    chains = []

    # Load all files names in the study folder, then sort by value
    if n_chains == 'auto':
        # Load all files names in the study folder, then sort by value
        files = sorted((RESULTS_FOLDER / study_folder).glob('*.npz'), key=lambda x: int(x.name.split('.')[0]))
        for file in files:
            chains.append(np.load(file))

    # Load a specific number of chains
    elif isinstance(n_chains, int):
        for i in range(n_chains):
            chains.append(np.load(RESULTS_FOLDER / study_folder / f'{i}.npz'))

    else:
        raise ValueError('n_chains must be an integer or "auto"')

    # For s, g, and pi, extract the chains
    s_0_chains = [np.mean(chain['slipping'], axis=1)[:, 0] for chain in chains]
    s_1_chains = [np.mean(chain['slipping'], axis=1)[:, 1] for chain in chains]
    g_0_chains = [np.mean(chain['guessing'], axis=1)[:, 0] for chain in chains]
    g_1_chains = [np.mean(chain['guessing'], axis=1)[:, 1] for chain in chains]
    pi_0_chains = [chain['pi_hat'][:, 0] for chain in chains]
    pi_1_chains = [chain['pi_hat'][:, 1] for chain in chains]
    c_chains = [np.mean(chain['c_hat'], axis=1) for chain in chains]
    mu_weight_chains = [chain['mu_weight_hat'] for chain in chains] if 'mu_weight_hat' in chains[0] else None
    score_diff_chains = [chain['score_diff'] for chain in chains]

    c_values = np.mean(np.stack([np.unique(chain['c_hat'], return_counts=True)[1] / chain['c_hat'].shape[0] for chain in chains]), axis=0)
    print('Average membership:', c_values)

    # Compute R-hat for all the chains
    if burn_in is None:
        burn_in = s_0_chains[0].shape[0] // 2

    r_hat_s_0 = gelman_rubin(s_0_chains, burn_in=burn_in)
    r_hat_s_1 = gelman_rubin(s_1_chains, burn_in=burn_in)
    r_hat_g_0 = gelman_rubin(g_0_chains, burn_in=burn_in)
    r_hat_g_1 = gelman_rubin(g_1_chains, burn_in=burn_in)
    r_hat_pi = gelman_rubin(pi_0_chains, burn_in=burn_in)
    r_hat_c = gelman_rubin(c_chains, burn_in=burn_in)

    # Create a pandas Series with the results
    r_hat_series = pd.Series({'R-hat for s_0': r_hat_s_0,
                              'R-hat for s_1': r_hat_s_1,
                              'R-hat for g_0': r_hat_g_0,
                              'R-hat for g_1': r_hat_g_1,
                              'R-hat for pi': r_hat_pi,
                              'R-hat for c': r_hat_c})

    # Print the Series
    print(r_hat_series)

    # Print the results for copy and paste into google sheets
    print('\nCopy and paste into google sheets:')
    for _, value in r_hat_series.items():
        print(value)

    # Score accuracy
    score_avg_acc = np.mean([np.mean(1 - score_diff) for score_diff in score_diff_chains])
    print(f'\nScore accuracy: {score_avg_acc}')

    # Compute alpha diff error measures
    if not real_data:
        print('\nAlpha diff error measures:')

        alpha_pred_chains = [np.mean(chain['alpha_hat'][burn_in:], axis=0) for chain in chains]
        alpha_final = np.round(np.mean(np.stack(alpha_pred_chains), axis=0))
        alpha_sim = chains[0]['alpha_sim']

        alpha_diff = np.abs(alpha_final - alpha_sim)

        # Error for all attributes
        total_error = np.mean(alpha_diff)

        # Marginal correct classification rate (for each attribute)
        attribute_errors = np.mean(alpha_diff, axis=0)

        # Proportion of examinees classified correctly for all K attributes
        correctly_classified_all = np.all(alpha_diff == 0, axis=1).mean()

        # Proportion of examinees classified correctly for at least K-1 attributes
        correctly_classified_k_minus_1 = np.sum(alpha_diff == 0, axis=1) >= alpha_diff.shape[1] - 1
        correctly_classified_k_minus_1_rate = correctly_classified_k_minus_1.mean()

        # Proportion of examinees classified incorrectly for K-1 or K attributes
        incorrectly_classified_k_minus_1 = np.sum(alpha_diff == 1, axis=1) >= alpha_diff.shape[1] - 1
        incorrectly_classified_k_minus_1_rate = incorrectly_classified_k_minus_1.mean()

        # Storing error measures in a Pandas Series
        error_measures = pd.Series({
            'Total Error': total_error,
            **{f'Error for Attribute {i}': error for i, error in enumerate(attribute_errors)},
            'Proportion Correctly Classified (All Attributes)': correctly_classified_all,
            'Proportion Correctly Classified (At Least K-1 Attributes)': correctly_classified_k_minus_1_rate,
            'Proportion Incorrectly Classified (K-1 or K Attributes)': incorrectly_classified_k_minus_1_rate
        })

        print(error_measures)

        # Print the results for copy and paste into google sheets
        print('\nCopy and paste into google sheets:')
        for _, value in error_measures.items():
            print(value)

        # Plot alpha diff
        if not skip_plots:
            print('\nPlotting alpha diff...')
            fig, ax = plt.subplots(figsize=(18, 6))
            mat_ax = ax.matshow(1 - alpha_diff.T, aspect='auto', cmap='PiYG')
            fig.colorbar(mat_ax, location='right')
            ax.set_title('Alpha Divergence')
            plt.tight_layout()
            if save_plots:
                (RESULT_PLOTS_FOLDER / study_folder).mkdir(parents=True, exist_ok=True)
                plt.savefig(RESULT_PLOTS_FOLDER / study_folder / 'alpha_diff.png')

            if show_plots:
                plt.show()

    if not skip_plots:
        # Plot all the chains
        s_ylim = (0.6, 0.8)
        g_ylim = (0.0, 0.3)

        # s_color = 'deepskyblue'
        s0_color = 'yellowgreen'
        s1_color = 'cornflowerblue'
        g0_color = 'lightcoral'
        g1_color = 'khaki'

        if real_data:
            print()  # empty line only when real data, since there is no newline line with the alpha diff plot

        plot_chains(
            chains_dict={'Strategy A': s_0_chains, 'Strategy B': s_1_chains},
            colors={'Strategy A': s0_color, 'Strategy B': s1_color},
            ylim=s_ylim,
            figure_title='Slipping parameter',
            save_path=RESULT_PLOTS_FOLDER / study_folder / 'slipping.png' if save_plots else None,
            show_plots=show_plots,
        )

        plot_chains(
            chains_dict={'Strategy A': g_0_chains, 'Strategy B': g_1_chains},
            colors={'Strategy A': g0_color, 'Strategy B': g1_color},
            ylim=g_ylim, figure_title='Guessing parameter',
            save_path=RESULT_PLOTS_FOLDER / study_folder / 'guessing.png' if save_plots else None,
            show_plots=show_plots,
        )

        plot_chains(
            chains_dict={
                'Slipping Strategy A': s_0_chains, 'Slipping Strategy B': s_1_chains,
                'Guessing Strategy A': g_0_chains, 'Guessing Strategy B': g_1_chains,
            },
            colors={
                'Slipping Strategy A': s0_color, 'Slipping Strategy B': s1_color,
                'Guessing Strategy A': g0_color, 'Guessing Strategy B': g1_color,
            },
            figure_title='Slipping & Guessing Parameters',
            save_path=RESULT_PLOTS_FOLDER / study_folder / 'slipping_guessing.png' if save_plots else None,
            show_plots=show_plots,
        )

        plot_chains(
            chains_dict={'Strategy A': pi_0_chains, 'Strategy B': pi_1_chains},
            colors={'Strategy A': s0_color, 'Strategy B': s1_color},
            figure_title='Mixing parameter',
            save_path=RESULT_PLOTS_FOLDER / study_folder / 'mixing.png' if save_plots else None,
            show_plots=show_plots,
        )

        plot_chains(
            chains_dict={'C': c_chains},
            colors={'C': 'mediumorchid'},
            figure_title='Strategy membership parameter mean',
            save_path=RESULT_PLOTS_FOLDER / study_folder / 'c.png' if save_plots else None,
            show_plots=show_plots,
        )

        if mu_weight_chains is not None:
            plot_chains(
                chains_dict={'Mu Weight': mu_weight_chains},
                colors={'Mu Weight': 'mediumorchid'},
                figure_title='Mu Weight',
                save_path=RESULT_PLOTS_FOLDER / study_folder / 'mu_weight.png' if save_plots else None,
                show_plots=show_plots,
            )


if __name__ == '__main__':
    studies = ['s1_500', 's1_1000', 's1_2000', 's2_20', 's2_30', 's3'][:1]
    # get the full folder names from their closest matches in results/
    directory = pathlib.Path('results')
    study_directories = [next(directory.glob(f'{study}*')) for study in studies]

    for study_folder in study_directories:
        print(f'\nProcessing {study_folder}')
        process_study(study_folder.name, save_plots=False, show_plots=False)
