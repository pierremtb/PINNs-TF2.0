#%% IMPORTING/SETTING UP PATHS

import sys
import json
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp

# Manually making sure the numpy random seeds are "the same" on all devices
np.random.seed(1234)
tf.random.set_seed(1234)

#%% LOCAL IMPORTS

eqnPath = "1dcomplex-schrodinger"
sys.path.append(eqnPath)
sys.path.append("utils")
from custom_lbfgs import lbfgs, Struct
from schrodingerutil import prep_data, Logger, plot_inf_cont_results

#%% HYPER PARAMETERS

if len(sys.argv) > 1:
  with open(sys.argv[1]) as hpFile:
    hp = json.load(hpFile)
else:
  hp = {}
  # Data size on the initial condition solution
  hp["N_0"] = 50
  # Collocation points on the boundaries
  hp["N_b"] = 50
  # Collocation points on the domain
  hp["N_f"] = 20000
  # DeepNN topology (2-sized input [x t], 4 hidden layer of 100-width, 2-sized output [u, v])
  hp["layers"] = [2, 100, 100, 100, 100, 2]
  # Setting up the TF SGD-based optimizer (set tf_epochs=0 to cancel it)
  hp["tf_epochs"] = 2000
  hp["tf_lr"] = 0.03
  hp["tf_b1"] = 0.99
  hp["tf_eps"] = 1e-1 
  # Setting up the quasi-newton LBGFS optimizer (set nt_epochs=0 to cancel it)
  hp["nt_epochs"] = 0
  hp["nt_lr"] = 0.8
  hp["nt_ncorr"] = 50

#%% DEFINING THE MODEL

class PhysicsInformedNN(object):
  def __init__(self, layers, optimizer, logger, X_f, ub, lb):
    # Descriptive Keras model
    self.H_model = tf.keras.Sequential()
    self.H_model.add(tf.keras.layers.InputLayer(input_shape=(layers[0],)))
    self.H_model.add(tf.keras.layers.Lambda(
      lambda X: 2.0*(X - lb)/(ub - lb) - 1.0))
    for width in layers[1:]:
        self.H_model.add(tf.keras.layers.Dense(
          width, activation=tf.nn.tanh,
          kernel_initializer='glorot_normal'))

    # Computing the sizes of weights/biases for future decomposition
    self.sizes_w = []
    self.sizes_b = []
    for i, width in enumerate(layers):
      if i != 1:
        self.sizes_w.append(int(width * layers[1]))
        self.sizes_b.append(int(width if i != 0 else layers[1]))

    self.optimizer = optimizer
    self.logger = logger

    self.dtype = tf.float32

    self.lb = lb
    self.ub = ub

    X_lb = np.concatenate((0*tb + lb[0], tb), 1) # (lb[0], tb)
    X_ub = np.concatenate((0*tb + ub[0], tb), 1) # (ub[0], tb)

    self.x_lb = X_lb[:,0:1]
    self.t_lb = X_lb[:,1:2]
    self.x_ub = X_ub[:,0:1]
    self.t_ub = X_ub[:,1:2]
    self.X_lb = tf.convert_to_tensor(np.hstack((self.x_lb, self.t_lb)), dtype=self.dtype)
    self.X_ub = tf.convert_to_tensor(np.hstack((self.x_ub, self.t_ub)), dtype=self.dtype)

    # Separating the collocation coordinates
    self.x_f = tf.convert_to_tensor(X_f[:, 0:1], dtype=self.dtype)
    self.t_f = tf.convert_to_tensor(X_f[:, 1:2], dtype=self.dtype)
    
  # Defining custom loss
  def __loss(self, h, h_pred):
    u0 = h[:,0:1]
    v0 = h[:,1:2]
    u0_pred = h_pred[:,0:1]
    v0_pred = h_pred[:,1:2]
    # return tf.reduce_mean(tf.square(u0 - u0_pred)) + \
    #        tf.reduce_mean(tf.square(v0 - v0_pred))

    u_lb_pred, v_lb_pred, u_x_lb_pred, v_x_lb_pred = self.uv_model(self.X_lb)
    u_ub_pred, v_ub_pred, u_x_ub_pred, v_x_ub_pred = self.uv_model(self.X_ub)
    f_u_pred, f_v_pred = self.f_model()
    
    return tf.reduce_mean(tf.square(u0 - u0_pred)) + \
           tf.reduce_mean(tf.square(v0 - v0_pred)) + \
           tf.reduce_mean(tf.square(u_lb_pred - u_ub_pred)) + \
           tf.reduce_mean(tf.square(v_lb_pred - v_ub_pred)) + \
           tf.reduce_mean(tf.square(u_x_lb_pred - u_x_ub_pred)) + \
           tf.reduce_mean(tf.square(v_x_lb_pred - v_x_ub_pred)) + \
           tf.reduce_mean(tf.square(f_u_pred)) + \
           tf.reduce_mean(tf.square(f_v_pred))

  def __grad(self, X, h):
    with tf.GradientTape() as tape:
      loss_value = self.__loss(h, self.H_model(X))
    return loss_value, tape.gradient(loss_value, self.__wrap_training_variables())

  def __wrap_training_variables(self):
    var = self.H_model.trainable_variables
    return var

  # Decomposes the multi-output into the complex values and spatial derivatives
  def uv_model(self, X):
    x = X[:,0:1]
    t = X[:,0:2]
    with tf.GradientTape(persistent=True) as tape:
      tape.watch(x)
      tape.watch(t)
      # Packing together the inputs
      Xtemp = tf.stack([x[:,0], t[:,0]], axis=1)

      h = self.H_model(Xtemp)
      u = h[:,0:1]
      v = h[:,1:2]
      
    u_x = tape.gradient(u, x)
    v_x = tape.gradient(v, x)
    del tape

    return u, v, u_x, v_x

  # The actual PINN
  def f_model(self):
    # Using the new GradientTape paradigm of TF2.0,
    # which keeps track of operations to get the gradient at runtime
    with tf.GradientTape(persistent=True) as tape:
      # Watching the two inputs we’ll need later, x and t
      tape.watch(self.x_f)
      tape.watch(self.t_f)
      # Packing together the inputs
      X_f = tf.stack([self.x_f[:,0], self.t_f[:,0]], axis=1)

      # Getting the prediction
      u, v, u_x, v_x = self.uv_model(X_f)
    
    # Getting the other derivatives
    u_xx = tape.gradient(u_x, self.x_f)
    v_xx = tape.gradient(v_x, self.x_f)
    u_t = tape.gradient(u, self.t_f)
    v_t = tape.gradient(v, self.t_f)

    # Letting the tape go
    del tape

    # Buidling the PINNs
    f_u = u_t + 0.5*v_xx + (u**2 + v**2)*v
    f_v = v_t - 0.5*u_xx - (u**2 + v**2)*u   
    
    return f_u, f_v

  def get_params(self, numpy=False):
    return []

  def get_weights(self):
    w = []
    for layer in self.H_model.layers[1:]:
      weights_biases = layer.get_weights()
      weights = weights_biases[0].flatten()
      biases = weights_biases[1]
      w.extend(weights)
      w.extend(biases)
    return tf.convert_to_tensor(w, dtype=self.dtype)

  def set_weights(self, w):
    for i, layer in enumerate(self.H_model.layers[1:]):
      start_weights = sum(self.sizes_w[:i]) + sum(self.sizes_b[:i])
      end_weights = sum(self.sizes_w[:i+1]) + sum(self.sizes_b[:i])
      weights = w[start_weights:end_weights]
      w_div = int(self.sizes_w[i] / self.sizes_b[i])
      weights = tf.reshape(weights, [w_div, self.sizes_b[i]])
      biases = w[end_weights:end_weights + self.sizes_b[i]]
      weights_biases = [weights, biases]
      layer.set_weights(weights_biases)

  def summary(self):
    return self.H_model.summary()

  # The training function
  def fit(self, X_h, h, tf_epochs=5000, nt_config=Struct()):
    self.logger.log_train_start(self)

    # Creating the tensors
    X_h = tf.convert_to_tensor(X_h, dtype=self.dtype)
    h = tf.convert_to_tensor(h, dtype=self.dtype)

    self.logger.log_train_opt("Adam")
    for epoch in range(tf_epochs):
      # Optimization step
      loss_value, grads = self.__grad(X_h, h)
      self.optimizer.apply_gradients(zip(grads, self.__wrap_training_variables()))
      self.logger.log_train_epoch(epoch, loss_value)
    
    self.logger.log_train_opt("LBFGS")
    def loss_and_flat_grad(w):
      with tf.GradientTape() as tape:
        self.set_weights(w)
        loss_value = self.__loss(h, self.H_model(X_h))
      grad = tape.gradient(loss_value, self.H_model.trainable_variables)
      grad_flat = []
      for g in grad:
        grad_flat.append(tf.reshape(g, [-1]))
      grad_flat =  tf.concat(grad_flat, 0)
      return loss_value, grad_flat
    # tfp.optimizer.lbfgs_minimize(
    #   loss_and_flat_grad,
    #   initial_position=self.get_weights(),
    #   num_correction_pairs=nt_config.nCorrection,
    #   max_iterations=nt_config.maxIter,
    #   f_relative_tolerance=nt_config.tolFun,
    #   tolerance=nt_config.tolFun,
    #   parallel_iterations=6)
    lbfgs(loss_and_flat_grad,
      self.get_weights(),
      nt_config, Struct(), True,
      lambda epoch, loss, is_iter:
        self.logger.log_train_epoch(epoch, loss, "", is_iter))

    self.logger.log_train_end(tf_epochs + nt_config.maxIter)

  def predict(self, X_star):
    h_pred = self.H_model(X_star)
    u_pred = h_pred[:,0:1]
    v_pred = h_pred[:,1:2]
    return u_pred.numpy(), v_pred.numpy()

#%% TRAINING THE MODEL

# Getting the data
path = os.path.join(eqnPath, "data", "NLS.mat")
x, t, X, T, Exact_u, Exact_v, Exact_h, \
  X_star, u_star, v_star, h_star, X_f, ub, lb, tb, x0, u0, v0, X0, H0 = prep_data(path, hp["N_0"], hp["N_b"], hp["N_f"], noise=0.05)

#%%

# Creating the model and training
logger = Logger(frequency=100, hp=hp)


# Setting up the optimizers with the previously defined hyper-parameters
nt_config = Struct()
nt_config.learningRate = hp["nt_lr"]
nt_config.maxIter = hp["nt_epochs"]
nt_config.nCorrection = hp["nt_ncorr"]
nt_config.tolFun = 1.0 * np.finfo(float).eps
tf_optimizer = tf.keras.optimizers.Adam(
  learning_rate=hp["tf_lr"],
  beta_1=hp["tf_b1"],
  epsilon=hp["tf_eps"])

pinn = PhysicsInformedNN(hp["layers"], tf_optimizer, logger, X_f, ub, lb)

# Defining the error function for the logger
def error():
  u_pred, v_pred = pinn.predict(X_star)
  h_pred = np.sqrt(u_pred**2 + v_pred**2)
  return np.linalg.norm(h_star - h_pred, 2) / np.linalg.norm(h_star, 2)
logger.set_error_fn(error)

# Training the PINN
pinn.fit(X0, H0, hp["tf_epochs"], nt_config)

# Getting the model predictions, from the same (x,t) that the predictions were previously gotten from
u_pred, v_pred = pinn.predict(X_star)
h_pred = np.sqrt(u_pred**2 + v_pred**2)

#%% PLOTTING
plot_inf_cont_results(X_star, u_pred, v_pred, h_pred, Exact_h, X, T, x, t, ub, lb, x0, tb,
  save_path=eqnPath, save_hp=hp) 