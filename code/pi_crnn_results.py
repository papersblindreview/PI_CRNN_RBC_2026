import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
import sys 
sys.path.insert(1, os.path.dirname(os.getcwd()))
from functions import *
import numpy as np 
import h5py
import tensorflow as tf
from keras import optimizers
from tensorflow.keras.saving import register_keras_serializable
from tensorflow.keras.utils import get_custom_objects
import time

# HYPERPARAMETERS
train_size, val_size = 80, 40
look_back, look_fwd = 10, 10
batch_size = 8
nodes, kernel_size = 128, 1
activation = 'tanh'

autoencoder = tf.keras.saving.load_model('cae.keras') 

# Physical constants
const_dict = load_constants()
Uf, P, T_h, T_0, Pr, Ra = get_model_constants(const_dict)

############################################

# CREATE SEQUENCE GENERATOR. SAVED WEIGHTS WILL BE LOADED
@register_keras_serializable()
class SequenceGenerator(tf.keras.Model):
  def __init__(self, hidden_size, kernel_size, out_size):
    super(SequenceGenerator, self).__init__(name='SequenceGenerator_Model')
    self.hidden_size = hidden_size
    self.kernel_size = kernel_size
    self.out_size = out_size
    self.rnn = ConvLSTM2D(hidden_size, kernel_size, return_state=True, padding='same', name='ConvLSTM_SG')
    self.conv = Conv2D(64, kernel_size=3, padding='same', name='Conv2D') 
    self.norm = LayerNormalization(name='Norm')
    self.act = LeakyReLU(0.2, name='ReLU')
    self.expand_dim = Lambda(lambda x: tf.expand_dims(x, axis=1))

  def build(self, input_shape):
    initial_input_shape = input_shape[0]
    h_shape = c_shape = (1, 16, 16, self.hidden_size)

    dummy_input = tf.zeros(initial_input_shape)
    dummy_h = tf.zeros(h_shape)
    dummy_c = tf.zeros(c_shape)

    dec_o, _, _ = self.rnn(dummy_input, initial_state=[dummy_h, dummy_c])
    _ = self.act(self.norm(self.conv(dec_o)))
    super().build(input_shape)
    
  def call(self, inputs):
    initial_input, h, c, targets, autoreg_prob = inputs
    T = tf.shape(targets)[1]
    t_switch = tf.cast(T, tf.float32) * autoreg_prob
    outputs = tf.TensorArray(dtype=tf.float32, size=T)
    input_at_t = initial_input
      
    def cond_autoreg(t, input_at_t, h, c, outputs):
      return tf.cast(t, tf.float32) < t_switch
      
    def body_autoreg(t, input_at_t, h, c, outputs):
      dec_o, h, c = self.rnn(input_at_t, initial_state=[h, c])
      output = self.act(self.norm(self.conv(dec_o)))
      outputs = outputs.write(t, output)
      input_at_t = self.expand_dim(output)
      return t + 1, input_at_t, h, c, outputs
      
    def cond_teacher(t, input_at_t, h, c, outputs):
      return tf.cast(t, tf.float32) < tf.cast(T, tf.float32)

    def body_teacher(t, input_at_t, h, c, outputs):
      dec_o, h, c = self.rnn(input_at_t, initial_state=[h, c])
      output = self.act(self.norm(self.conv(dec_o)))
      outputs = outputs.write(t, output)
      input_at_t = targets[:, t:t+1]
      return t + 1, input_at_t, h, c, outputs

    t = tf.constant(0)
    shape_invs = [t.get_shape(), tf.TensorShape([None, None, 16, 16, self.out_size]), tf.TensorShape([None, 16, 16, 128]), tf.TensorShape([None, 16, 16, 128]), tf.TensorShape(None)]
    t, input_at_t, h, c, outputs = tf.while_loop(cond_autoreg, body_autoreg, loop_vars=[t, input_at_t, h, c, outputs], shape_invariants=shape_invs)
    t, input_at_t, h, c, outputs = tf.while_loop(cond_teacher, body_teacher, loop_vars=[t, input_at_t, h, c, outputs], shape_invariants=shape_invs)
    return tf.transpose(outputs.stack(), perm=[1,0,2,3,4])  

  def get_config(self):    
    config = super().get_config()
    config.update({"hidden_size": self.hidden_size, "kernel_size": self.kernel_size, 'out_size':self.out_size})
    return config
    
  @classmethod
  def from_config(cls, config):
    return cls(**config)  

# load the 4 pieces of the spatiotemporal model and data
ae_encoder = build_ae_encoder(autoencoder)
context_builder = tf.keras.saving.load_model('context_builder.keras')
sequence_generator = SequenceGenerator(hidden_size=nodes, kernel_size=kernel_size, out_size=64)
sequence_generator.load_weights('sequence_generator.weights.h5', overwrite=True)
ae_decoder = build_ae_decoder(autoencoder)

data_train, data_val, x, z, _ = load_data(2000, 500, Uf, P, T_h, T_0)
uwpT = np.concatenate((data_train, data_val), axis=0)

# helper function to forecast with PI-CRNN given an input sequence
def run_forecast(input_rbc, horizon):
    input_encoder = ae_encoder.predict(tf.expand_dims(input_rbc, axis=0), verbose=0)
    h, c = context_builder(input_encoder, training=False)
    x = sequence_generator(input_encoder[:,-1:], h, c, horizon, training=False)
    forecast = ae_decoder.predict(x, verbose=0, batch_size=8) 
    return forecast[0]

ensembles = 30
horizon = 20

forecast_starts = np.linspace(look_back, val_size-1, ensembles).astype('int')

forecast_ensemble_pi_crnn = []
for s in forecast_starts:
    input_rbc_temp = data_val[(s-look_back):s]
    forecast_temp = run_forecast(input_rbc_temp, horizon)
    forecast_ensemble_pi_crnn.append(forecast_temp)

forecast_ensemble_crnn = np.load('forecast_ensemble_crnn.npy')
forecast_ensemble_pi_esn = np.load('forecast_ensemble_pi_esn.npy')
forecast_ensemble_arima = np.load('forecast_ensemble_arima.npy')

dx, dz, dt = get_grads(x, z, const_dict, Uf)

def compute_pde_loss(forecast_ensemble, model_name):
    pde_losses = []
    for i in range(ensembles):
        forecast_temp = forecast_ensemble[i]
        pde_residuals_temp = ns_loss(forecast_temp, Pr, Ra, dx, dz, dt) # shape (horizon, 256, 256, 4)
        pde_loss_temp = (pde_residuals_temp**2).mean()
        pde_losses.append(pde_loss_temp)

    print(model_name)
    print(f'PDE Loss: {np.median(pde_losses):.2e} ({np.quantile(pde_losses, 0.75) - np.quantile(pde_losses, 0.25):.2e})\n')

compute_pde_loss(forecast_ensemble_pi_crnn, model_name='PI-CRNN')
compute_pde_loss(forecast_ensemble_crnn, model_name='CRNN')
compute_pde_loss(forecast_ensemble_pi_esn, model_name='PI-ESN')
compute_pde_loss(forecast_ensemble_arima, model_name='ARIMA')

# true Nu number
j_conv = uwpT[...,127,1:2] * (uwpT[...,127,-1:] - tf.reduce_mean(uwpT[...,127,-1:], axis=(0,1,2), keepdims=True) ) 
Nu_true = 1 + tf.math.sqrt(Pr*Ra) * tf.reduce_mean(j_conv)

def compute_Nu(forecast_ensemble, model_name):
    Nus = []
    for i in range(ensembles):
        forecast_temp = forecast_ensemble[i]
        j_conv = forecast_temp[...,127,1:2] * (forecast_temp[...,127,-1:] - tf.reduce_mean(forecast_temp[...,127,-1:], axis=(0,1,2), keepdims=True) ) 
        Nus.append(1 + tf.math.sqrt(Pr*Ra) * tf.reduce_mean(j_conv))

    print(model_name)
    print(f'Nu true: {Nu_true:.2f}. Pred: {np.median(Nus):.4f} ({np.quantile(Nus, 0.75) - np.quantile(Nus, 0.25):.4f})')

compute_Nu(forecast_ensemble_pi_crnn, model_name='PI-CRNN')
compute_Nu(forecast_ensemble_crnn, model_name='CRNN')
compute_Nu(forecast_ensemble_pi_esn, model_name='PI-ESN')
compute_Nu(forecast_ensemble_arima, model_name='ARIMA')

from scipy.stats import gaussian_kde

def compute_pdfs(forecast_ensemble, model_name):
    var_list = ['u','w','p','T']
    x_grid_size = 500

    kls = {}
    for i, v in enumerate(['u','w','T']):
        j = var_list.index(v)
        forecast_ensemble_var = forecast_ensemble[...,j]
        x_range = np.linspace(uwpT[...,j].min(), uwpT[...,j].max(), x_grid_size)

        kde_true = []
        for s in np.linspace(0, uwpT.shape[0]-horizon-1, ensembles):
          kde_true.append( gaussian_kde(uwpT[int(s):int(s+horizon),...,j].flatten())(x_range) )
        kde_true = np.asarray(kde_true)
        pdf_true = np.clip(np.asarray(kde_true).mean(axis=0), 1e-10, None)

        kl_temp = []
        for k in range(ensembles):
            kde_temp = gaussian_kde(forecast_ensemble_var[k].flatten())
            pdf_temp = np.clip(kde_temp(x_range), 1e-10, None)
            kl_temp.append( (pdf_true * np.log(pdf_true / pdf_temp)).sum() ) 

        print(model_name)
        print(f'{v}. Median: {np.median(kl_temp):.2e}, iqr: {np.quantile(kl_temp, 0.75) - np.quantile(kl_temp, 0.25):.2e}\n')

compute_pdfs(forecast_ensemble_pi_crnn, model_name='PI-CRNN')
compute_pdfs(forecast_ensemble_crnn, model_name='CRNN')
compute_pdfs(forecast_ensemble_pi_esn, model_name='PI-ESN')
compute_pdfs(forecast_ensemble_arima, model_name='ARIMA')

# true dissipation
z_beg = 25
z_end = 230 
epsilon_T = np.array( DX(uwpT[...,-1:], dx)**2 + DZ(uwpT[...,-1:], dz)**2 )[...,z_beg:z_end,-1].mean(axis=0)
epsilon_T = epsilon_T * const_dict['kappa'][0,0] * (const_dict['T_bot'][0,0]-const_dict['T_top'][0,0])**2

def compute_diss(forecast_ensemble, model_name):
    epsilon_T_pred = []
    for i in range(ensembles):
        epsilon_T_temp = np.array( DX(forecast_ensemble[i,...,-1:], dx)**2 + DZ(forecast_ensemble[i,...,-1:], dz[z_beg:z_end])**2 ).mean(axis=0)
        epsilon_T_pred.append(epsilon_T_temp * const_dict['kappa'][0,0] * (const_dict['T_bot'][0,0]-const_dict['T_top'][0,0])**2 )

    diff = np.asarray(epsilon_T_pred)[...,0] - np.expand_dims(epsilon_T, axis=0)
    error = diff**2 
    med = np.median(error.mean(axis=(1,2)))
    q1 = np.quantile(error.mean(axis=(1,2)), 0.25)
    q2 = np.quantile(error.mean(axis=(1,2)), 0.75)

    print(model_name)
    print(f'Dissipation error: {med:.2e} ({q2-q1:.2e})\n')

compute_diss(forecast_ensemble_pi_crnn, model_name='PI-CRNN')
compute_diss(forecast_ensemble_crnn, model_name='CRNN')
compute_diss(forecast_ensemble_pi_esn, model_name='PI-ESN')
compute_diss(forecast_ensemble_arima, model_name='ARIMA')

import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.ticker import FuncFormatter
formatter = FuncFormatter(lambda x, pos: f'{x:.1e}')

horizon = 20

forecast = run_forecast(data_val[:look_back], horizon)

mins_data = uwpT.min(axis=(0,1,2)).reshape(1,-1)
maxs_data = uwpT.max(axis=(0,1,2)).reshape(1,-1)
mins_f = forecast.min(axis=(0,1,2)).reshape(1,-1)
maxs_f = forecast.max(axis=(0,1,2)).reshape(1,-1)

mins = np.concatenate((mins_data, mins_f), axis=0).min(axis=0)
maxs = np.concatenate((maxs_data, maxs_f), axis=0).max(axis=0)

X, Z = np.meshgrid(x, z)

t_ = np.arange(1, forecast.shape[0]+1) * 0.05
t_list = np.linspace(0, horizon-1, 5).astype('int')

fig, ax = plt.subplots(2, len(t_list), figsize=(20,7), layout='constrained')

for j, t in enumerate(t_list):        
    im1 = ax[0,j%5].contourf(X, Z, forecast[t,...,3].T, cmap='jet', vmin=mins[-1], vmax=maxs[-1], levels=np.linspace(mins[-1],maxs[-1],40))
    im1 = ax[1,j%5].contourf(X, Z, data_val[t,...,3].T, cmap='jet', vmin=mins[-1], vmax=maxs[-1], levels=np.linspace(mins[-1],maxs[-1],40))

ax[0,j%5].set_title(r'$t$ =' + f' {t_[t]:.2f}s', fontsize=15)

cbar1 = plt.colorbar(im1, ax=ax[1,:], orientation='horizontal', shrink=0.4, aspect=20, pad=0.05)
cbar1.locator = ticker.MaxNLocator(nbins=5)
cbar1.update_ticks()

for i in range(5): ax[0,i].xaxis.set_visible(False)
ax[0,1].yaxis.set_visible(False)
ax[0,2].yaxis.set_visible(False)
ax[0,3].yaxis.set_visible(False)
ax[0,4].yaxis.tick_right() 
ax[0,4].yaxis.set_ticks_position('right')

ax[1,1].yaxis.set_visible(False)
ax[1,2].yaxis.set_visible(False)
ax[1,3].yaxis.set_visible(False)
ax[1,4].yaxis.tick_right() 
ax[1,4].yaxis.set_ticks_position('right')

plt.show()
