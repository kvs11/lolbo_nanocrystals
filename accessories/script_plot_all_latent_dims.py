"""
Latent space: 32 dims
> Find the min and maximum of each latent dimension
> divide the latent space into n_bins bins
> count the number of data samples that belong to each bin in each dimension
> Plot a histogram with count of samples in a bin on y-axis and bins on x-axis
"""
import numpy as np
import matplotlib.pyplot as plt


def plot_latent_hist(latent_data, latent_space_dims, n_bins, hist_plot_name):

    # Find the min and max of each latent dimension
    min_values = np.min(latent_data, axis=0)
    max_values = np.max(latent_data, axis=0)

    # Divide the latent space into n_bins bins
    bin_ranges = [np.linspace(min_val, max_val, n_bins + 1) for min_val, max_val in zip(min_values, max_values)]

    # Count the number of data samples that belong to each bin in each dimension
    bin_counts = [np.histogramdd(latent_data[:, i], bins=[bin_range])[0] for i, bin_range in enumerate(bin_ranges)]
    max_count = np.array(bin_counts).max()

    # Plot histograms
    fig, axes = plt.subplots(nrows=4, ncols=8, figsize=(40, 24))

    for i in range(latent_space_dims):
        row = i // 8
        col = i % 8
        ax = axes[row, col]
    
        ax.hist(bin_ranges[i][:-1], bins=bin_ranges[i], weights=bin_counts[i], edgecolor='k')
        ax.set_ylim(0, max_count)
        ax.set_xlabel(f'Latent Dim {i + 1}', fontsize=22)

        # Set y-axis label on the first column only
        if col == 0:
            ax.set_ylabel('Count', fontsize=22)

        # Set y-tick labels to only first column
        if col != 0:
            ax.set_yticklabels([])
    
        # Set xtick_labels on the last row only
        #if row != 3:
        #    ax.set_xticklabels([])

        # Set xtick and ytick labels to fontsize 18
        ax.tick_params(axis='both', which='both', labelsize=18)
        
    # Remove empty subplots if the number of latent dimensions isn't a multiple of 4x8
    if latent_space_dims % (4 * 8) != 0:
        for i in range(latent_space_dims, 4 * 8):
            row = i // 8
            col = i % 8
            fig.delaxes(axes[row, col])

    plt.tight_layout()
    plt.savefig(hist_plot_name)


if __name__ == '__main__':
    # Define the parameters
    latent_space_dims = 32
    n_bins = 16
    hist_plot_name = "train_latent_hist.png"
    latent_Zs_path = "latent_Zs.npy"
    latent_Zs = np.load(latent_Zs_path, allow_pickle=True).astype('float32')

    plot_latent_hist(latent_Zs, latent_space_dims, n_bins, hist_plot_name)

