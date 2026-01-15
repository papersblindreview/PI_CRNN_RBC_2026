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

epochs = 1000
train_size, val_size = 80, 40
look_back, look_fwd = 10, 10
batch_size = 8
nodes, kernel_size = 128, 1
activation = 'tanh'

autoencoder = tf.keras.saving.load_model('cae.keras') 

const_dict = load_constants()
Uf, P, T_h, T_0, Pr, Ra = get_model_constants(const_dict)

train_size, val_size = 2000, 500
data_train, data_val, x, z, _ = load_data(train_size, val_size, Uf, P, T_h, T_0)
dx_np, dz_np, dt_np = get_grads(x, z, const_dict, Uf)

dx = tf.constant(dx_np, tf.float32)
dz = tf.constant(dz_np, tf.float32)
dt = tf.constant(np.array(dt_np).reshape(1,), tf.float32)

# CREATE CONTEXT BUILDER
def get_context_builder(hidden_size, kernel_size):
  inputs = tf.keras.layers.Input(shape=(None, 16, 16, 64), name='Inputs')
  _, h1, c1 = ConvLSTM2D(hidden_size, kernel_size, return_sequences=True, return_state=True, padding='same', name='ConvLSTM_CB')(inputs)
  return tf.keras.Model(inputs, [h1, c1], name='ContextBuilder_Model')

# CREATE SEQUENCE GENERATOR
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
  
  
optimizer = tf.keras.optimizers.Adam(1e-3)

ae_encoder = build_ae_encoder(autoencoder)
context_builder = get_context_builder(hidden_size=nodes, kernel_size=kernel_size)
sequence_generator = SequenceGenerator(hidden_size=nodes, kernel_size=kernel_size, out_size=64)
ae_decoder = build_ae_decoder(autoencoder)

# LOAD DATA
data_train, data_val, x, z = load_lstm_data(train_size, val_size, look_back, look_fwd, Uf, P, T_h, T_0, seqs_train=16, seqs_val=4, autoencoder=autoencoder)               

@tf.function(input_signature=[tf.TensorSpec(shape=[1,look_fwd,256,256,4], dtype=tf.float32),
                              tf.TensorSpec(shape=[1,look_fwd,256,256,4], dtype=tf.float32)])
def loss_fn(U_true, U_pred): 
    loss_data = tf.reduce_mean(tf.math.square(U_pred-U_true), axis=[0,1,2,3])  
    Ld_u = loss_data[0]
    Ld_w = loss_data[1]
    Ld_p = loss_data[2]
    Ld_T = loss_data[3]
    return Ld_u, Ld_w, Ld_p, Ld_T


# VALIDATION STEP HELPER FUNCTION TO KEEP TRACK OF VALIDATION LOSS
low_dims = 256 // (2**4)  
@tf.function(input_signature=[tf.TensorSpec(shape=[1, look_back, low_dims, low_dims, 64], dtype=tf.float32),
                            tf.TensorSpec(shape=[1, look_fwd, low_dims, low_dims, 64], dtype=tf.float32),
                            tf.TensorSpec(shape=[1, look_fwd, 256, 256, 4], dtype=tf.float32),
                            tf.TensorSpec(shape=[], dtype=tf.float32)])
def val_step(x_batch, x_dec, U_batch, autoreg_prob):
    h, c = context_builder(x_batch, training=False)
    x = sequence_generator((x_batch[:,-1:], h, c, x_dec, autoreg_prob), training=False)
    U_pred = ae_decoder(x, training=False)
    Ld_u, Ld_w, Ld_p, Ld_T = loss_fn(U_batch, U_pred) 
    return tf.stack([Ld_u, Ld_w, Ld_p, Ld_T], axis=0)

# HELPER FUNCTION FOR TRAIN STEP FOR EACH BATCH
## Batch size of 1, gradients are accumulated to simulate larger batch size due to memory constraints
input_signature_train = [tf.TensorSpec(shape=[1, look_back, low_dims, low_dims, 64], dtype=tf.float32),
                       tf.TensorSpec(shape=[1, look_fwd, low_dims, low_dims, 64], dtype=tf.float32),
                       tf.TensorSpec(shape=[1, look_fwd, 256, 256, 4], dtype=tf.float32),
                       tf.TensorSpec(shape=[4], dtype=tf.float32),
                       tf.TensorSpec(shape=[], dtype=tf.float32)]

 
@tf.function(input_signature=input_signature_train)
def train_step(x_batch, x_dec, U_batch, l_dwa, autoreg_prob):
  with tf.GradientTape(persistent=True) as tape:
    h, c = context_builder(x_batch, training=True)
    x = sequence_generator((x_batch[:,-1:], h, c, x_dec, autoreg_prob), training=True)
    U_pred = ae_decoder(x, training=False)
    
    Ld_u, Ld_w, Ld_p, Ld_T = loss_fn(U_batch, U_pred)
    loss_data = tf.stack([Ld_u, Ld_w, Ld_p, Ld_T], axis=0)
    loss = (l_dwa[0]*Ld_u + l_dwa[1]*Ld_w + l_dwa[2]*Ld_p + l_dwa[3]*Ld_T)
    
  grad = tape.gradient(loss, context_builder.trainable_variables + sequence_generator.trainable_variables)
  
  return loss_data, grad

# HELPER VARIABLES FOR TRAINING
loss_history = []
w = tf.constant(0.9, dtype=tf.float32)
l_dwa = tf.Variable(tf.ones([4], tf.float32) / 4, trainable=False)
 
best_loss = float('inf')
patience = 20
decay_rate = 0.8

dwa_steps = tf.Variable(16, dtype=tf.int32, trainable=False)
acc_steps = tf.Variable(4, dtype=tf.int32, trainable=False)

grad_acc = [tf.Variable(tf.zeros_like(v), trainable=False) for v in context_builder.trainable_variables]
grad_acc += [tf.Variable(tf.zeros_like(v), trainable=False) for v in sequence_generator.trainable_variables]

# WE USE TEACHER FORCING WITH SCHEDULED SAMPLING
## The training starts with the model predicting one step ahead. 
## Gradually we expose the model to longer output sequences up to the full 60
rec_prob = tf.Variable(0., trainable=False)
cur_step = tf.Variable(0, dtype=tf.int32, trainable=False)
rec_prob_val = tf.constant(1.)

stop_teach = look_fwd*20
warmup = 250
learning_rate = 1-3
min_lr = 1e-4
wait = 0

# RUN TRAINING
for epoch in tf.range(1, epochs, dtype=tf.float32):
  
  rec_prob.assign( tf.minimum(1., epoch / stop_teach) )
  
  Ldata = tf.zeros([4], tf.float32)
  val_loss = tf.zeros([4], tf.float32)
    
  for step, (x_train, x_dec_train, U_train) in enumerate(data_train):
    cur_step.assign(step)
    
    Ldata_b, grad_data_b = train_step(x_train, x_dec_train, U_train, l_dwa, rec_prob)
    Ldata += Ldata_b
    
    for i in tf.range(len(grad_acc)):
      grad_acc[i].assign_add(grad_data_b[i] / tf.cast(acc_steps, tf.float32) )
      
    loss_history.append(Ldata_b)

    # Loss balancing & gradient accumulation
    if (cur_step+1) % acc_steps == 0: 
    
      dwa_rollmean = tf.reduce_mean(loss_history[-dwa_steps:-1], axis=0)
      l_dwa.assign( tf.nn.softmax(tf.cast(loss_history[-1] / dwa_rollmean, tf.float32)) )
      
      optimizer.apply_gradients(zip(grad_acc, context_builder.trainable_variables + sequence_generator.trainable_variables))
      
      for i in tf.range(len(grad_acc)):
        grad_acc[i].assign(tf.zeros_like(grad_data_b[i]))

  Ldata /= (step+1)
  for step_val, (x_val, x_dec_val, U_val) in enumerate(data_val):      
    val_loss_b = val_step(x_val, x_dec_val, U_val, rec_prob_val)
    val_loss = tf.math.add_n([val_loss, val_loss_b])
    
  val_loss /= (step_val+1)
  val_data_loss = tf.reduce_mean(val_loss[:4])
  val_loss = val_loss[-4:]

  # Learning rate scheduler
  if epoch >= warmup:
    if (best_loss - tf.reduce_mean(val_loss)) > 1e-3: 
      best_loss = tf.reduce_mean(val_loss)
      wait = 0
    else:
      wait += 1
      
    if wait >= patience:
      new_lr = max(learning_rate * decay_rate, min_lr)
      if new_lr < learning_rate:
        learning_rate = new_lr
        optimizer.learning_rate.assign(learning_rate)
        
      wait = 0
   
  log1 = f"{tf.cast(epoch, tf.int32)}. Data:{tf.reduce_mean(Ldata[:4]):.2e}, "
  log2 = f"MC:{Ldata[4]:.2e}, u:{Ldata[5]:.2e}, w:{Ldata[6]:.2e}, T:{Ldata[7]:.2e}, V Data:{val_data_loss:.2e}"
  
  print(log1+log2)

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

forecast_ensemble_crnn = []
for s in forecast_starts:
    input_rbc_temp = data_val[(s-look_back):s]
    forecast_temp = run_forecast(input_rbc_temp, horizon)
    forecast_ensemble_crnn.append(forecast_temp)

np.save('forecast_ensemble_crnn.npy', forecast_ensemble_crnn)

batch_size = 20

# HELPER FUNCTIONS TO COMPUTE DERIVATIVES
@tf.function(input_signature=[tf.TensorSpec(shape=[batch_size,256,256,4], dtype=tf.float32),
                              tf.TensorSpec(shape=[1], dtype=tf.float32)]) 
def DT_tf(var, dt):
    ddt1 = (var[...,1:2,:,:,:] - var[...,:1,:,:,:]) / dt
    ddt = (var[...,2:,:,:,:] - var[...,:-2,:,:,:]) / (2*dt)
    ddt2 = (var[...,-2:-1,:,:,:] - var[...,-1:,:,:,:]) / (-dt)
    ddt = tf.concat([ddt1,ddt,ddt2], axis=-4)
    return ddt

@tf.function(input_signature=[tf.TensorSpec(shape=[batch_size,256,256,4], dtype=tf.float32),
                              tf.TensorSpec(shape=[256], dtype=tf.float32)]) 
def DX_tf(var, dx):
    dx = tf.reshape(dx, [1,1,tf.shape(dx)[0],1,1])
    ddx1 = var[...,1:2,:,:] - var[...,:1,:,:]
    ddx = var[...,2:,:,:] - var[...,:-2,:,:]
    ddx2 = var[...,-2:-1,:,:] - var[...,-1:,:,:]
    ddx = tf.concat([ddx1, ddx, ddx2], axis=-3)
    return ddx / dx 

@tf.function(input_signature=[tf.TensorSpec(shape=[batch_size,256,256,4], dtype=tf.float32),
                              tf.TensorSpec(shape=[256], dtype=tf.float32)])  
def DZ_tf(var, dz):
    dz = tf.reshape(dz, [1,1,1,tf.shape(dz)[0],1])
    ddz1 = var[...,:,1:2,:] - var[...,:,:1,:]
    ddz = var[...,:,2:,:] - var[...,:,:-2,:]
    ddz2 = var[...,:,-2:-1,:] - var[...,:,-1:,:]
    ddz = tf.concat([ddz1,ddz,ddz2], axis=-2)
    return ddz / dz 

#PHYSICS LOSS WRT MASS, MOMENTUM, ENERGY CONSERVATION
@tf.function(input_signature=[tf.TensorSpec(shape=[batch_size,256,256,4], dtype=tf.float32),
                              tf.TensorSpec(shape=[batch_size,256,256,4], dtype=tf.float32)])
def loss_ns(U_true, U_pred): 
    loss_data = tf.reduce_mean(tf.math.square(U_pred-U_true), axis=[0,1,2])  
    Ld_u = loss_data[0]
    Ld_w = loss_data[1]
    Ld_p = loss_data[2]
    Ld_T = loss_data[3]

    U_pred_x = DX_tf(U_pred, dx)
    U_pred_z = DZ_tf(U_pred, dz)
    U_pred_t  = DT_tf(U_pred, dt)
    U_pred_xx = DX_tf(U_pred_x, dx)
    U_pred_zz = DZ_tf(U_pred_z, dz)  

    u, w, p, T = tf.split(U_pred, 4, axis=-1)
    u_x, w_x, p_x, T_x = tf.split(U_pred_x, 4, axis=-1)
    u_z, w_z, p_z, T_z = tf.split(U_pred_z, 4, axis=-1)
    u_t, w_t, _, T_t = tf.split(U_pred_t, 4, axis=-1)
    u_xx, w_xx, _, T_xx = tf.split(U_pred_xx, 4, axis=-1)
    u_zz, w_zz, _, T_zz = tf.split(U_pred_zz, 4, axis=-1)

    f_mc = u_x + w_z     
    f_u = u_t + u*u_x + w*u_z + p_x - tf.math.sqrt(Pr/Ra)*(u_xx + u_zz)
    f_w = w_t + u*w_x + w*w_z + p_z - tf.math.sqrt(Pr/Ra)*(w_xx + w_zz) - T
    f_T = T_t + u*T_x + w*T_z - (T_xx + T_zz)/tf.math.sqrt(Pr*Ra)
      
    L_mc = tf.reduce_mean(tf.math.square(f_mc))
    L_u = tf.reduce_mean(tf.math.square(f_u))
    L_w = tf.reduce_mean(tf.math.square(f_w))
    L_T = tf.reduce_mean(tf.math.square(f_T))

    return Ld_u, Ld_w, Ld_p, Ld_T, L_mc, L_u, L_w, L_T


optimizer = tf.keras.optimizers.Adam(1e-3)
ae_path_model = 'cae.keras'

ae_convESN = build_ae_ESN(nodes=128, kernel_size=3, ae_path_model=ae_path_model, look_back=80)

data_train_esn, data_val_esn, x, z = load_esn_data(train_size, val_size, look_back, batch_size, Uf, P, T_h, T_0, ae_path_model)

low_dims = 16
input_signature_train = [tf.TensorSpec(shape=[batch_size, look_back, low_dims, low_dims, 64], dtype=tf.float32),
                         tf.TensorSpec(shape=[batch_size, 256, 256, 4], dtype=tf.float32),
                         tf.TensorSpec(shape=[8], dtype=tf.float32),
                         tf.TensorSpec(shape=[2], dtype=tf.float32)]

@tf.function(input_signature=input_signature_train)
def train_step(x_batch, U_batch, l_dwa, l_g):
    with tf.GradientTape(persistent=True) as tape:
        U_pred = ae_convESN(x_batch, training=True)
        Ld_u, Ld_w, Ld_p, Ld_T, L_mc, L_u, L_w, L_T = loss_ns(U_batch, U_pred)
        ldata = (l_dwa[0]*Ld_u + l_dwa[1]*Ld_w + l_dwa[2]*Ld_p + l_dwa[3]*Ld_T)
        lpde = (l_dwa[4]*L_mc + l_dwa[5]*L_u + l_dwa[6]*L_w + l_dwa[7]*L_T)
        loss_data_pde = tf.stack([Ld_u, Ld_w, Ld_p, Ld_T, L_mc, L_u, L_w, L_T], axis=0)
    
    grad_data = tape.gradient(ldata, ae_convESN.trainable_variables)
    grad_data_flat = tf.concat([tf.reshape(g, [-1]) for g in grad_data if g is not None], axis=0)
    grad_data_norm = tf.math.sqrt(tf.reduce_mean(tf.math.square(grad_data_flat)))
    
    grad_pde = tape.gradient(lpde, ae_convESN.trainable_variables)
    grad_pde_flat = tf.concat([tf.reshape(g, [-1]) for g in grad_pde if g is not None], axis=0) 
    grad_pde_norm = tf.math.sqrt(tf.reduce_mean(tf.math.square(grad_pde_flat)))
    
    grad = [l_g[0]*g_data + l_g[1]*g_pde for g_data, g_pde in zip(grad_data, grad_pde)]
    grad_data_pde_norms = tf.stack([grad_data_norm, grad_pde_norm], axis=0)
  
    optimizer.apply_gradients(zip(grad, ae_convESN.trainable_variables))
    return loss_data_pde, grad_data_pde_norms

@tf.function(input_signature=[tf.TensorSpec(shape=[batch_size, look_back, low_dims, low_dims, 64], dtype=tf.float32),
                              tf.TensorSpec(shape=[batch_size, 256, 256, 4], dtype=tf.float32)])
def val_step(x_batch, U_batch):
    U_pred = ae_convESN(x_batch, training=False)
    Ld_u, Ld_w, Ld_p, Ld_T, L_mc, L_u, L_w, L_T = loss_ns(U_batch, U_pred)
    return tf.stack([Ld_u, Ld_w, Ld_p, Ld_T, L_mc, L_u, L_w, L_T], axis=0)  

start_time = time.time()
learning_rate = 1e-3
best_loss = float('inf')

l_g = tf.convert_to_tensor(np.array([1.,1.], dtype=np.float32)) 
l_dwa = tf.ones([8], tf.float32) / 8
grad_norms = tf.zeros([2], tf.float32)
loss_history = []
dwa_steps = 16

for epoch in range(epochs):
    Ldata_pde = tf.zeros([8], tf.float32)
    val_loss = tf.zeros([8], tf.float32)
  
    for step, (x_batch, U_batch) in enumerate(data_train_esn):
        Ldata_pde_b, grad_norms_b = train_step(x_batch, U_batch, l_dwa, l_g)
        Ldata_pde += Ldata_pde_b
        grad_norms = tf.math.add_n([grad_norms, grad_norms_b])
        loss_history.append(Ldata_pde)
    
    if (step+1) % dwa_steps == 0: 
        dwa_rollmean = tf.reduce_mean(loss_history[-dwa_steps:-1], axis=0)
        l_dwa = tf.nn.softmax(tf.cast(loss_history[-1] / dwa_rollmean, tf.float32))
        
        l_g_data_hat = tf.reduce_sum(grad_norms)/grad_norms[0]
        l_g_pde_hat = tf.reduce_sum(grad_norms)/grad_norms[1] 
        l_g_hat = tf.stack([l_g_data_hat, l_g_pde_hat])
        l_g = 0.9*l_g + 0.1*l_g_hat
        grad_norms = tf.zeros([2], tf.float32)     
    Ldata_pde /= (step+1)  
  
    for step, (x_batch, U_batch) in enumerate(data_val_esn):
        val_loss += val_step(x_batch, U_batch)
    val_loss /= (step+1)
  
  
    val_data_loss = tf.reduce_mean(val_loss[:4])
    val_pde_loss = val_loss[-4:] 
    if (best_loss - tf.reduce_mean(val_pde_loss)) > 1e-3: 
        best_loss = tf.reduce_mean(val_pde_loss)
        wait = 0
    else:
        wait += 1
    
    if wait >= 20:
        new_lr = max(learning_rate * 0.8, 1e-4)
        if new_lr < learning_rate:
            learning_rate = new_lr
            optimizer.learning_rate.assign(learning_rate)
        wait = 0
  
  
    log1 = f"{epoch+1}. Data: {tf.reduce_mean(Ldata_pde[:4]):.2e}, MC: {Ldata_pde[4]:.2e}, u: {Ldata_pde[5]:.2e}, "
    log2 = f"w: {Ldata_pde[6]:.2e}, T: {Ldata_pde[7]:.2e}, Val Data: {val_data_loss:.2e}, "
    log3 = f"Val MC: {val_pde_loss[0]:.2e}, Val u: {val_pde_loss[1]:.2e}, Val w: {val_pde_loss[2]:.2e}, Val T: {val_pde_loss[3]:.2e}"
    print(log1+log2+log3)

data = np.concatenate((data_train, data_val), axis=0)

@tf.function
def predict_step(input_seq):
    input_temp = ae_encoder(input_seq, training=False)
    return ae_convESN(input_temp, training=False)

forecast_ensemble_pi_esn = []
forecast_starts = np.linspace(look_back, val_size-1, ensembles).astype('int')
for s in forecast_starts:

    ensemble_member = data[s-look_back:s]
    for i in range(look_back, look_back+horizon):
        output_temp = predict_step(tf.expand_dims(ensemble_member[-look_back:], axis=0))
        ensemble_member = tf.concat([ensemble_member, output_temp], axis=0)

    forecast_ensemble_pi_esn.append( np.array(ensemble_member[look_back:]) )

np.save('forecast_ensemble_pi_esn.npy', forecast_ensemble_pi_esn)

import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from joblib import Parallel, delayed

n_ens = 30
def fit_arima(time_series, val_size=378, n_ens=n_ens):

    forecast = np.zeros((n_ens, val_size, 4))
    for k in range(4):
        ts = time_series[:,k]

        # Near constant fallback, ARIMA will fail 
        if np.std(ts) < 1e-6:
            forecast[:,:,k] = ts[-1]
            continue

        try:
            model = ARIMA(ts, order=(10, 0, 0), trend="t")
            model_fit = model.fit()
            sims = model_fit.simulate(nsimulations=val_size, repetitions=n_ens, anchor="end")
            forecast[:,:,k] = sims.T

        except (ValueError, np.linalg.LinAlgError):
            # Fallback on failure
            forecast[:,:,k] = ts[-1]

    return forecast

data_arima_train = data_train.reshape((train_size, 256*256, 4))
results = Parallel(n_jobs=-1)(delayed(fit_arima)(data_arima_train[:,i,:], val_size) for i in range(256*256))

results = np.asarray(results) # (n_cells, n_ens, val_size, 4)
results_arima = results.reshape(256, 256, n_ens, val_size, 4)

results_arima = np.moveaxis(results_arima, [2, 3], [0, 1]) # (n_ens, val_size, 256, 256, 4)

np.save('forecast_ensemble_arima.npy', results_arima)
