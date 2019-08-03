#%% IMPORTING/SETTING UP PATHS

import sys
import os
import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp

# Manually making sure the numpy random seeds are "the same" on all devices
np.random.seed(1234)
tf.random.set_seed(1234)

#%% LOCAL IMPORTS

from custom_lbfgs import lbfgs, Struct
from burgersutil import prep_data, Logger, plot_inf_disc_results, appDataPath

#%% HYPER PARAMETERS

# Data size on initial condition on u
N_n = 250
# Number of RK stages
q = 500
# DeepNN topology (1-sized input [x], 3 hidden layer of 50-width, q+1-sized output [u_1^n(x), ..., u_{q+1}^n(x)]
layers = [1, 50, 50, 50, q + 1]
# Setting up the TF SGD-based optimizer (set tf_epochs=0 to cancel it)
tf_epochs = 200
tf_optimizer = tf.keras.optimizers.Adam(
  lr=0.001,
  beta_1=0.9,
  beta_2=0.999,
  epsilon=1e-08)
# Setting up the quasi-newton LBGFS optimizer (set nt_epochs=0 to cancel it)
nt_epochs = 1000
nt_config = Struct()
nt_config.learningRate = 0.8
nt_config.maxIter = nt_epochs
nt_config.nCorrection = 50
nt_config.tolFun = 1.0 * np.finfo(float).eps

#%% DEFINING THE MODEL

class PhysicsInformedNN(object):
  def __init__(self, layers, optimizer, logger, dt, x_1, lb, ub, nu, q, IRK_weights, IRK_times):
    self.lb = lb
    self.ub = ub
    self.nu = nu

    self.dt = dt

    self.q = max(q,1)
    self.IRK_weights = IRK_weights
    self.IRK_times = IRK_times

    # Descriptive Keras model [2, 50, …, 50, q+1]
    self.U_1_model = tf.keras.Sequential()
    self.U_1_model.add(tf.keras.layers.InputLayer(input_shape=(layers[0],)))
    self.U_1_model.add(tf.keras.layers.Lambda(
      lambda X: 2.0*(X - lb)/(ub - lb) - 1.0))
    for width in layers[1:]:
        self.U_1_model.add(tf.keras.layers.Dense(
          width, activation=tf.nn.tanh,
          kernel_initializer='glorot_normal'))

    # Computing the sizes of weights/biases for future decomposition
    self.sizes_w = []
    self.sizes_b = []
    for i, width in enumerate(layers):
      if i != 1:
        self.sizes_w.append(int(width * layers[1]))
        self.sizes_b.append(int(width if i != 0 else layers[1]))

    self.dtype = tf.float32

    self.x_1 = tf.convert_to_tensor(x_1, dtype=self.dtype)

    self.optimizer = optimizer
    self.logger = logger

  def U_0_model(self, x):
    # Using the new GradientTape paradigm of TF2.0,
    # which keeps track of operations to get the gradient at runtime
    with tf.GradientTape(persistent=True) as tape:
      # Watching the two inputs we’ll need later, x and t
      tape.watch(x)
      tape.watch(self.dummy_x0_tf)

      # Getting the prediction, and removing the last item (q+1)
      U_1 = self.U_1_model(x) # shape=(len(x), q+1)
      U = U_1[:, :-1] # shape=(len(x), q)

      # Deriving INSIDE the tape (2-step-dummy grad technique because U is a mat)
      g_U = tape.gradient(U, x, output_gradients=self.dummy_x0_tf)
      U_x = tape.gradient(g_U, self.dummy_x0_tf)
      g_U_x = tape.gradient(U_x, x, output_gradients=self.dummy_x0_tf)
    
    # Doing the last one outside the with, to optimize performance
    # Impossible to do for the earlier grad, because they’re needed after
    U_xx = tape.gradient(g_U_x, self.dummy_x0_tf)

    # Letting the tape go
    del tape

    # Buidling the PINNs, shape = (len(x), q+1), IRK shape = (q, q+1)
    nu = self.get_params(numpy=True)
    N = U*U_x - nu*U_xx # shape=(len(x), q)
    return U_1 + self.dt*tf.matmul(N, self.IRK_weights.T)

  # Defining custom loss
  def __loss(self, u_0, u_0_pred):
    u_1_pred = self.U_1_model(self.x_1)
    return tf.reduce_sum(tf.square(u_0_pred - u_0)) + \
      tf.reduce_sum(tf.square(u_1_pred))

  def __grad(self, x_0, u_0):
    with tf.GradientTape() as tape:
      loss_value = self.__loss(u_0, self.U_0_model(x_0))
    return loss_value, tape.gradient(loss_value, self.__wrap_training_variables())

  def __wrap_training_variables(self):
    var = self.U_1_model.trainable_variables
    return var

  def get_weights(self):
    w = []
    for layer in self.U_1_model.layers[1:]:
      weights_biases = layer.get_weights()
      weights = weights_biases[0].flatten()
      biases = weights_biases[1]
      w.extend(weights)
      w.extend(biases)
    return tf.convert_to_tensor(w, dtype=self.dtype)

  def set_weights(self, w):
    for i, layer in enumerate(self.U_1_model.layers[1:]):
      start_weights = sum(self.sizes_w[:i]) + sum(self.sizes_b[:i])
      end_weights = sum(self.sizes_w[:i+1]) + sum(self.sizes_b[:i])
      weights = w[start_weights:end_weights]
      w_div = int(self.sizes_w[i] / self.sizes_b[i])
      weights = tf.reshape(weights, [w_div, self.sizes_b[i]])
      biases = w[end_weights:end_weights + self.sizes_b[i]]
      weights_biases = [weights, biases]
      layer.set_weights(weights_biases)

  def get_params(self, numpy=False):
    return self.nu

  def summary(self):
    return self.U_1_model.summary()

  # The training function
  def fit(self, x_0, u_0, tf_epochs, nt_config):
    self.logger.log_train_start(self)

    # Creating the tensors
    x_0 = tf.convert_to_tensor(x_0, dtype=self.dtype)
    u_0 = tf.convert_to_tensor(u_0, dtype=self.dtype)

    # Creating dummy tensors for the gradients
    self.dummy_x0_tf = tf.ones([x_0.shape[0], self.q], dtype=self.dtype)

    self.logger.log_train_opt("Adam")
    for epoch in range(tf_epochs):
      # Optimization step
      loss_value, grads = self.__grad(x_0, u_0)
      self.optimizer.apply_gradients(
        zip(grads, self.__wrap_training_variables()))
      self.logger.log_train_epoch(epoch, loss_value)

    self.logger.log_train_opt("LBFGS")
    def loss_and_flat_grad(w):
      with tf.GradientTape() as tape:
        self.set_weights(w)
        loss_value = self.__loss(u_0, self.U_0_model(x_0))
      grad = tape.gradient(loss_value, self.U_1_model.trainable_variables)
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
    
    self.logger.log_train_end(tf_epochs)

  def predict(self, x_star):
    u_star = self.U_1_model(x_star)[:, -1]
    return u_star

#%% TRAINING THE MODEL

# Setup
lb = np.array([-1.0])
ub = np.array([1.0])
idx_t_0 = 10
idx_t_1 = 90
nu = 0.01/np.pi

# Getting the data
path = os.path.join(appDataPath, "burgers_shock.mat")
x, t, dt, \
  Exact_u, x_0, u_0, x_1, x_star, u_star, \
  IRK_weights, IRK_times = prep_data(path, N_n=N_n, q=q, lb=lb, ub=ub, noise=0.0, idx_t_0=idx_t_0, idx_t_1=idx_t_1)

# Creating the model and training
logger = Logger(frequency=10)
pinn = PhysicsInformedNN(layers, tf_optimizer, logger, dt, x_1, lb, ub, nu, q, IRK_weights, IRK_times)
def error():
  u_pred = pinn.predict(x_star)
  return np.linalg.norm(u_pred - u_star, 2) / np.linalg.norm(u_star, 2)
logger.set_error_fn(error)
pinn.fit(x_0, u_0, tf_epochs, nt_config)

# Getting the model predictions, from the same (x,t) that the predictions were previously gotten from
u_1_pred = pinn.predict(x_star)

#%% PLOTTING
plot_inf_disc_results(x_star, idx_t_0, idx_t_1, x_0, u_0, ub, lb, u_1_pred, Exact_u, x, t)