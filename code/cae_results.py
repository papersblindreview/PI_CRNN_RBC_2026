import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
import sys 
sys.path.insert(1, os.path.dirname(os.getcwd()))
from functions import *

def ssim(preds, data):
  c1, c2 = 1e-5, 1e-5
  
  mu = data.mean(axis=(1,2))
  mu_hat = preds.mean(axis=(1,2))
  sigma = data.std(axis=(1,2))
  sigma_hat = preds.std(axis=(1,2))
  
  data_centered = data - mu[:, None, None, :]
  preds_centered = preds - mu_hat[:, None, None, :]
  sigma_cross = np.mean(data_centered * preds_centered, axis=(1, 2))  # shape (n, 4)
  
  # Compute SSIM components
  luminance = (2 * mu * mu_hat + c1) / (mu**2 + mu_hat**2 + c1)
  contrast = (2 * sigma * sigma_hat + c2) / (sigma**2 + sigma_hat**2 + c2)
  structure = (sigma_cross + c2/2) / (sigma * sigma_hat + c2/2)
  
  # Combine components
  ssim_map = luminance * contrast * structure
  
  return ssim_map

# load model and data
model_path = 'cae.keras'
autoencoder = tf.keras.models.load_model(model_path, safe_mode=False)

const_dict = load_constants() 
Uf, P, T_h, T_0, Pr, Ra = get_model_constants(const_dict) 

train_size, val_size = 2000, 500
data_train, data_val, x, z, _ = load_data(train_size, val_size, Uf, P, T_h, T_0)

preds_val = autoencoder.predict(data_val, batch_size=10, verbose=0) 
 
mse_val = ((preds_val-data_val)**2)
iqr = np.quantile(mse_val, 0.75) - np.quantile(mse_val, 0.25)
val_losses = mse_val.mean(axis=(0,1,2))

ssim_map = ssim(preds_val, data_val)

iqr = np.quantile(ssim_map, 0.75, axis=0) - np.quantile(ssim_map, 0.25, axis=0)

ssim_val = ssim_map.mean(axis=0)

print(f'Validation MSE: (u) {val_losses[0]:.2e}, (w) {val_losses[1]:.2e}, (p) {val_losses[2]:.2e}, (T) {val_losses[3]:.2e}\n')

print(f'SSIM: (u) {ssim_val[0]:.2e}, (w) {ssim_val[1]:.2e}, (p) {ssim_val[2]:.2e}, (T) {ssim_val[3]:.2e}')
print(f'IQR:  (u) {iqr[0]:.2e}, (w) {iqr[1]:.2e}, (p) {iqr[2]:.2e}, (T) {iqr[3]:.2e}')

from sklearn.decomposition import PCA, IncrementalPCA, FactorAnalysis, FastICA
import scipy.linalg as la
from pykrige.ok import OrdinaryKriging
from joblib import Parallel, delayed

X, Z = np.meshgrid(x, z)
x_kr = X[::4,::4].flatten()
z_kr = Z[::4,::4].flatten()

def kriging(var_krig, x_kr=x_kr, z_kr=z_kr, x=x, z=z):
    OK_u = OrdinaryKriging(x_kr, z_kr, var_krig[...,0].flatten(), variogram_model='exponential', verbose=False, enable_plotting=False)
    OK_w = OrdinaryKriging(x_kr, z_kr, var_krig[...,1].flatten(), variogram_model='exponential', verbose=False, enable_plotting=False)
    OK_p = OrdinaryKriging(x_kr, z_kr, var_krig[...,2].flatten(), variogram_model='exponential', verbose=False, enable_plotting=False)
    OK_T = OrdinaryKriging(x_kr, z_kr, var_krig[...,3].flatten(), variogram_model='exponential', verbose=False, enable_plotting=False)
    u_temp, _ = OK_u.execute('grid', x, z)
    w_temp, _ = OK_w.execute('grid', x, z)
    p_temp, _ = OK_p.execute('grid', x, z)
    T_temp, _ = OK_T.execute('grid', x, z)
    return np.concatenate((u_temp[...,None],w_temp[...,None],p_temp[...,None],T_temp[...,None]), axis=-1)

results_kr = Parallel(n_jobs=-1)(delayed(kriging)(data_val[i,::4,::4,:]) for i in ids)
results_kr = np.asarray(results_kr)

mse_kr = (results_kr-data_val)**2
print(f'FRK MSE: {mse_kr.mean():.2e}')
print(f'FRK IQR: {np.quantile(mse_kr, 0.75)-np.quantile(mse_kr, 0.25):.2e}')

ssim_map = ssim(results_kr, data_val)
iqr = np.quantile(ssim_map, 0.75, axis=0) - np.quantile(ssim_map, 0.25, axis=0)
ssim_val = ssim_map.mean(axis=0)

print(f'SSIM: (u) {ssim_val[0]:.2e}, (w) {ssim_val[1]:.2e}, (p) {ssim_val[2]:.2e}, (T) {ssim_val[3]:.2e}')
print(f'IQR:  (u) {iqr[0]:.2e}, (w) {iqr[1]:.2e}, (p) {iqr[2]:.2e}, (T) {iqr[3]:.2e}')

def perform_pod(snapshots, num_modes=None):
    N, M = snapshots.shape  # N = snapshots, M = spatial DOFs
      
    # 1. Subtract mean field
    mean_field = np.mean(snapshots, axis=0, keepdims=True)
    X = snapshots - mean_field   # (N, M)

    # 2. Correlation matrix in snapshot space (N, N)
    C = X @ X.T

    # Step 3: eigen-decomposition
    eigvals, eigvecs = la.eigh(C)  # symmetric matrix
    idx = np.argsort(eigvals)[::-1]  # descending order
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # Step 4: truncate
    eigvals = eigvals[:num_modes]
    eigvecs = eigvecs[:, :num_modes]

    # Step 5: spatial modes
    denom = np.sqrt(np.maximum(eigvals, 1e-12))[np.newaxis, :]
    modes = (X.T @ eigvecs) / denom

    # Step 6: temporal coefficients
    temporal_coefs = np.diag(np.sqrt(eigvals)) @ eigvecs.T

    return modes, eigvals, temporal_coefs, mean_field

def reduce_new_data(new_data, modes, mean_field):
    X_new = new_data - mean_field
    reduced_coefficients = X_new @ modes
    return reduced_coefficients
    
def reconstruct_from_coefficients(coefs, modes, mean_field): 
    return coefs @ modes.T + mean_field

preds_pod_val = []
for i, v in enumerate(['u','w','p','T']):
    snapshots_train = data_train[...,i].reshape(train_size, -1) 
    snapshots_val = data_val[...,i].reshape(val_size, -1) 

    print(f'Variable: {v}')
    modes, eigvals, temporal_coefs, mean_field = perform_pod(snapshots_train, num_modes=train_size)
    coefs_val = reduce_new_data(snapshots_val, modes, mean_field)
    pred_pod = reconstruct_from_coefficients(coefs_val, modes, mean_field)

    preds_pod_val.append(pred_pod.reshape((val_size,256,256,1)))
    mse = ((pred_pod - snapshots_val)**2).mean()
    print(f'MSE: {mse:.2e}')

preds_pod_val = np.concatenate(preds_pod_val, axis=-1)
ssim_map = ssim(preds_pod_val, data_val)

iqr = np.quantile(ssim_map, 0.75, axis=0) - np.quantile(ssim_map, 0.25, axis=0)
ssim_val = ssim_map.mean(axis=0)

print(f'SSIM: (u) {ssim_val[0]:.2e}, (w) {ssim_val[1]:.2e}, (p) {ssim_val[2]:.2e}, (T) {ssim_val[3]:.2e}')
print(f'IQR:  (u) {iqr[0]:.2e}, (w) {iqr[1]:.2e}, (p) {iqr[2]:.2e}, (T) {iqr[3]:.2e}')

data = np.concatenate((data_train, data_val), axis=0)
data_flat = data.reshape((train_size+val_size, 256*256*4))

data_flat_train = data_flat[:train_size]
data_flat_val = data_flat[train_size:]

n_components = train_size

ipca = IncrementalPCA(n_components=n_components)
ipca.partial_fit(data_flat_train)

data_pca_val_tr = ipca.transform(data_flat_val)
data_pca_val_itr = ipca.inverse_transform(data_pca_val_tr)

ipca_mse = (data_pca_val_itr-data_flat_val)**2
ipca_iqr = np.quantile(ipca_mse, 0.75) - np.quantile(ipca_mse, 0.25)

preds_pca_val = data_pca_val_itr.reshape(data_val.shape)

print('IPCA Validtion MSE: {:.4e}'.format(ipca_mse.mean()))
print('IPCA Validtion IQR: {:.4e}'.format(ipca_iqr))

ssim_map = ssim(preds_pca_val, data_val)

iqr = np.quantile(ssim_map, 0.75, axis=0) - np.quantile(ssim_map, 0.25, axis=0)
ssim_val = ssim_map.mean(axis=0)

print(f'SSIM: (u) {ssim_val[0]:.2e}, (w) {ssim_val[1]:.2e}, (p) {ssim_val[2]:.2e}, (T) {ssim_val[3]:.2e}')
print(f'IQR:  (u) {iqr[0]:.2e}, (w) {iqr[1]:.2e}, (p) {iqr[2]:.2e}, (T) {iqr[3]:.2e}')

fastica = FastICA(n_components=n_components, random_state=0, max_iter=1000)
fastica.fit(data_flat_train)

data_ica_val_tr = fastica.transform(data_flat_val)
data_ica_val_itr = fastica.inverse_transform(data_ica_val_tr)

ica_mse = (data_ica_val_itr-data_flat_val)**2
ica_iqr = np.quantile(ica_mse, 0.75) - np.quantile(ipca_mse, 0.25)

print('ICA Validtion MSE: {:.4e}'.format(ica_mse.mean()))
print('ICA Validtion IQR: {:.4e}'.format(ica_iqr))

preds_ica_val = data_ica_val_itr.reshape(data_val.shape)
ssim_map = ssim(preds_ica_val, data_val)

iqr = np.quantile(ssim_map, 0.75, axis=0) - np.quantile(ssim_map, 0.25, axis=0)
ssim_val = ssim_map.mean(axis=0)

print(f'SSIM: (u) {ssim_val[0]:.2e}, (w) {ssim_val[1]:.2e}, (p) {ssim_val[2]:.2e}, (T) {ssim_val[3]:.2e}')
print(f'IQR:  (u) {iqr[0]:.2e}, (w) {iqr[1]:.2e}, (p) {iqr[2]:.2e}, (T) {iqr[3]:.2e}')

import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.ticker import FuncFormatter
formatter = FuncFormatter(lambda x, pos: f'{x:.1e}')

X, Z = np.meshgrid(x, z)
mins = data_val.min(axis=(0,1,2))
maxs = data_val.max(axis=(0,1,2))

fig, ax = plt.subplots(3, 4, figsize=(20,7), sharex=True, sharey=True, layout='constrained')

ax[0,0].set_ylabel('Data', fontsize=15)
ax[1,0].set_ylabel('Predictions', fontsize=15)

for j, v in enumerate([r'$u$',r'$w$',r'$p$',r'$\theta$']):        
  im1 = ax[0,j].contourf(X, Z, data_val[-150,:,:,j].T, cmap='jet', vmin=mins[j], vmax=maxs[j], levels=np.linspace(mins[j],maxs[j],40))
  im1 = ax[1,j].contourf(X, Z, preds_val[-150,:,:,j].T, cmap='jet', vmin=mins[j], vmax=maxs[j], levels=np.linspace(mins[j],maxs[j],40))
  
  cbar1 = plt.colorbar(im1, ax=ax[1,j], orientation='horizontal', extend='max', shrink=0.7, aspect=20, pad=0.05)
  cbar1.locator = ticker.MaxNLocator(nbins=5)
  cbar1.update_ticks()
  
  ax[0,j].set_title(v, fontsize=15)

plt.show()
