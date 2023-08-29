import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap

############################################################
# t-SNE representation for fingerprints/representations
############################################################

def get_UMAP_2D_embedding(fingerprints, n_neighbors=15, n_epochs=None, min_dist=0.1):
    features = np.array([f.flatten() for f in fingerprints]) 
    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, n_epochs=n_epochs, min_dist=min_dist)
    umap_embedding = reducer.fit_transform(features)

    def scale_to_01_range(x):
        value_range = (np.max(x) - np.min(x))
        starts_from_zero = x - np.min(x)
        return starts_from_zero / value_range

    tx = umap_embedding[:, 0]
    ty = umap_embedding[:, 1]

    tx = scale_to_01_range(tx)
    ty = scale_to_01_range(ty)
    return tx,ty


############ UMAP Plot

latent_Zs_path = ""
Y_array_path = ""

a, b, c = 100, 400, 0.02
plot_title = ""
plot_type = "intervals"            # or "continuous"
plot_name = "umap_intervals.png"    # or "umap_continuous.png"

latent_Zs = np.load(latent_Zs_path, allow_pickle=True).astype('float32')
Y = np.load(Y_array_path, allow_pickle=True).astype('float32')

# Custom y_intervals are being used to have > 1.6 models as outliers group. 
y_intervals = [min(Y)-0.0001, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, max(Y)+0.0001]

interval_inds = []
for i in range(len(y_intervals)):
    if i+1 < len(y_intervals):
        inds_1 = np.where(y_intervals[i]<Y)[0]
        inds_2 = np.where(Y<y_intervals[i+1])[0]
        inds = np.intersect1d(inds_1, inds_2)
        interval_inds.append(inds)
test_lens = [len(i) for i in interval_inds]


legends = [' < 0.2', '0.2 - 0.3', '0.3 - 0.4', '0.4 - 0.6', 
            '0.6 - 0.8', '0.8 - 1.0', '1.0 - 1.2', '1.2 - 1.4', ' > 1.4']
#colors = [plt.cm.tab10(i) for i in range(10)]
#del colors[7] # remove dark gray color
colors = [plt.cm.Set1(i) for i in range(9)]
colors.reverse()


tx, ty = get_UMAP_2D_embedding(latent_Zs, n_neighbors=a, n_epochs=b, min_dist=c)


if plot_type == "intervals":
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    plt.tick_params(labelsize=18)
    plotting_seq = [i for i in range(9)]
    plotting_seq.reverse()
    alphas = [0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 1, 1]
    handles = []
    for i, j in enumerate(plotting_seq):
        group_inds = interval_inds[j]
        color = colors[i]
        if i==7:
            color='red'
        if i==8:
            color='black'
        alpha = alphas[i]
        current_tx = np.take(tx, group_inds)
        current_ty = np.take(ty, group_inds)
        pp = ax.scatter(current_tx, current_ty, color=color, alpha=1, s=20)
        handles.append(pp)
    handles.reverse()
    plt.legend(handles, legends, prop={'size': 14})
    plt.title(plot_title, fontsize=20)
    plt.tight_layout()
    plt.savefig(fname=plot_name)
    plt.close()

if plot_type == "continuous":
    # Plot continuously
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111)
    plt.tick_params(labelsize=18)

    # Set all Y>1.4 equal to 1.41
    mask_Y = Y.copy()
    mask_Y[mask_Y>1.4] = 1.41
    sorted_tx = [xx for _, xx in sorted(zip(mask_Y, tx), reverse=True)]
    sorted_ty = [yy for _, yy in sorted(zip(mask_Y, ty), reverse=True)]
    mask_Y[::-1].sort()

    plt.scatter(sorted_tx, sorted_ty, c=mask_Y, cmap='brg', s=15, alpha=1)
    cb = plt.colorbar()
    #cb = plt.colorbar(im, orientation="horizontal", pad=0.15)
    cb.set_label(label=r'E$_f$ (meV/atom)', size=18)
    cb.ax.tick_params(labelsize=18)
    plt.title(plot_title, fontsize=20)
    plt.tight_layout()
    plt.savefig(fname=plot_name)
    plt.close()