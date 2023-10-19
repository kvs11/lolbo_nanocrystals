"""
This is what the script does:

load the last checkpoint
get the latent Zs of the training set
get the latent Zs of the test set
save them as .npy files in the same directory

use the training latent Zs and get UMAP plot
use the testing latent Zs and get the UMAP plot
combine the training and testing latent Zs and get the UMAP plot


get the latent dim histogram with training data. (No need of testing data as 
we only want to see if it is following unit gaussian in all latent dims)

generate 10 samples each from low density and high density region & decode them. 
(We should see poor decoded structures from low-den & "decent" ones from high-den regions)
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from lolbo_nanocrystal.accessories import script_Dataset_to_LatentZs, \
                            script_Zs_to_UMAPEmbeds, script_plot_all_latent_dims

model_dir_path = os.getcwd()

path_prefix = '/sandbox/vkolluru/Gen_models_for_FANTASTX/test_cases/CdTe_case/FX_5000/Datasets'
train_PC_array_path = path_prefix + '/first_1000/train_set_arrays/PC_array.npy'
train_Y_array_path = path_prefix + '/first_1000/train_set_arrays/Y_array.npy'
train_embds_path = None

test_PC_array_path = path_prefix + '/last_500_test_200/train_set_arrays/PC_array.npy'
test_Y_array_path = path_prefix + '/last_500_test_200/train_set_arrays/Y_array.npy'
test_Nsites_path = path_prefix + '/last_500_test_200/train_set_arrays/Nsites.npy'
test_embds_path = None

bsz = 64
path_to_vae_ckpt = model_dir_path + "/lightning_logs/NanoCrystalVAE/version_0/checkpoints/last.ckpt"
path_to_vae_statedict = None
save_to_npy_file = True


nc_vae_params = dict(
        input_dim = 188,
        channel_dim = 3,
        regression_dim = 1,
        graph_embds_dim = 0,
        latent_dim = 32,
        max_filters = 128,
        num_filters = [32, 64, 128],
        filter_size = [5, 3, 3],
        strides = [2, 2, 1],
        coeffs = (1, 2, 15),
)



# ------------- UMAP inputs
train_latent_Zs_path = "train_latent_Zs.npy"
test_latent_Zs_path = "test_latent_Zs.npy"

n_neighbors, n_epochs, min_dist = 80, 400, 0.03

plot_title = "Latent space 2D-UMAP"
plot_type = "continuous"            # or "continuous"
plot_name = "umap_continuous.png"    # or "umap_intervals.png"

# Do not change this
n_intervals = 10    

# ------------- Latent space hist plot params

# Define the parameters
latent_space_dims = 32
n_bins = 16
hist_plot_name = "train_latent_hist.png"


print ("Input parameters read. Proceeding with post-analysis..")
######### 1. script_Dataset_to_LatentZs


input_x_path = train_PC_array_path
input_graph_embds_path = train_embds_path 
train_latent_Zs = script_Dataset_to_LatentZs.get_latent_Zs(
        nc_vae_params=nc_vae_params,
        input_x_path=input_x_path, 
        input_graph_embds_path=input_graph_embds_path, 
        bsz=bsz, 
        path_to_vae_ckpt=path_to_vae_ckpt, 
        path_to_vae_statedict=path_to_vae_statedict, 
        save_to_npy_file=save_to_npy_file
)
os.rename('latent_Zs.npy', train_latent_Zs_path)
print ("Train set latent Zs created and saved..")

input_x_path = test_PC_array_path
input_graph_embds_path = test_embds_path 
test_latent_Zs = script_Dataset_to_LatentZs.get_latent_Zs(
        nc_vae_params=nc_vae_params,
        input_x_path=input_x_path, 
        input_graph_embds_path=input_graph_embds_path, 
        bsz=bsz, 
        path_to_vae_ckpt=path_to_vae_ckpt, 
        path_to_vae_statedict=path_to_vae_statedict, 
        save_to_npy_file=save_to_npy_file
)
os.rename('latent_Zs.npy', test_latent_Zs_path)
print ("Test set latent Zs created and saved..")

######### 2. Get the UMPA embeddings ; use script_Zs_to_UMAPEmbeds
print ("Generating 2D-UMAP plot of Train set..")
Y_array_path = train_Y_array_path
latent_Zs_path = train_latent_Zs_path
plot_title = "Training set latent space"
plot_name = "umap_train.png"
script_Zs_to_UMAPEmbeds.get_UMAP_plot(Y_array_path, latent_Zs_path, 
                                      n_neighbors, n_epochs, min_dist, 
                                      plot_title, plot_type, plot_name, n_intervals)

print ("Generating 2D-UMAP plot of Test set..")
Y_array_path = test_Y_array_path
latent_Zs_path = test_latent_Zs_path
plot_title = "Test set latent space"
plot_name = "umap_test.png"
script_Zs_to_UMAPEmbeds.get_UMAP_plot(Y_array_path, latent_Zs_path, 
                                      n_neighbors, n_epochs, min_dist, 
                                      plot_title, plot_type, plot_name, n_intervals)

print ("Generating 2D-UMAP plot of joint Train/Test set..")
# joint training and test set umap plot
joint_latent_Zs = np.concatenate((train_latent_Zs, test_latent_Zs), axis=0)
train_Ys = np.load(train_Y_array_path, allow_pickle=True).astype('float32')
test_Ys = np.load(test_Y_array_path, allow_pickle=True).astype('float32')
joint_Ys = np.concatenate((train_Ys, test_Ys), axis=0)

tx, ty = script_Zs_to_UMAPEmbeds.get_UMAP_2D_embedding(joint_latent_Zs,  
                        n_neighbors=n_neighbors, n_epochs=n_epochs, min_dist=min_dist)
train_tx, test_tx = tx[:train_Ys.shape[0]], tx[train_Ys.shape[0]:]
train_ty, test_ty = ty[:train_Ys.shape[0]], ty[train_Ys.shape[0]:]

# Plot continuously
fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(111)
plt.tick_params(labelsize=18)

plt.scatter(train_tx, train_ty, c='gray', s=20, alpha=0.7)
plt.scatter(test_tx, test_ty, c='blue', s=20, alpha=0.8, marker='D')

plt.title("Train/test latent space", fontsize=20)
plt.tight_layout()
plt.savefig(fname="umap_joint_train_test.png")
plt.close()

print ("Generating 2D-UMAP dual plot of Train/Test sets..")
# plot dual plot train and test subplots
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))

for i, data in enumerate(zip([train_tx, test_tx], [train_ty, test_ty], [train_Ys, test_Ys])):
    ttx, tty, tYs = data
    sorted_tx = [xx for _, xx in sorted(zip(tYs, ttx), reverse=True)]
    sorted_ty = [yy for _, yy in sorted(zip(tYs, tty), reverse=True)]

    # Set all Y>1.4 equal to 1.41   
    mask_Y = tYs.copy()
    mask_Y[::-1].sort()
    mask_Y[mask_Y>0] = 0

    sc = axes[i].scatter(sorted_tx, sorted_ty, c=mask_Y, cmap='brg', s=15, alpha=1)
    axes[i].tick_params(labelsize=15)

cb = fig.colorbar(sc, ax=axes, orientation='vertical')
#cb = plt.colorbar(im, orientation="horizontal", pad=0.15)
cb.set_label(label=r'E$_f$ (eV/atom)', size=18)
cb.ax.tick_params(labelsize=18)

axes[0].set_title("Training set", fontsize=18)
axes[1].set_title("Test set", fontsize=18)
#plt.tight_layout()
plt.savefig(fname="umap_dual_subplots.png")
plt.close()

######### 3. Train latent space hist ; script_plot_all_latent_dims

print ("Creating all latent dims distribution plot..")
script_plot_all_latent_dims.plot_latent_hist(train_latent_Zs.numpy(), latent_space_dims, n_bins, hist_plot_name)

print ("Done!!!")
