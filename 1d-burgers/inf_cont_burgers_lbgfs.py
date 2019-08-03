#%% IMPORTING/SETTING UP PATHS

import sys
import os
import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp

# Manually making sure the numpy random seeds are "the same" on all devices
np.random.seed(1234)
tf.random.set_seed(1234)

#%% LOCAL IMPORTS

from custom_lbfgs import lbfgs, Struct
from burgersutil import prep_data, Logger, plot_inf_cont_results, appDataPath

#%% HYPER PARAMETERS

# Data size on the solution u
N_u = 50
# Collocation points size, where we’ll check for f = 0
N_f = 10000
# DeepNN topology (2-sized input [x t], 8 hidden layer of 20-width, 1-sized output [u]
layers = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]
# Setting up the TF SGD-based optimizer (set tf_epochs=0 to cancel it)
tf_epochs = 0
tf_optimizer = tf.keras.optimizers.Adam(
  learning_rate=0.1,
  beta_1=0.99,
  epsilon=1e-1)
# Setting up the quasi-newton LBGFS optimizer (set nt_epochs=0 to cancel it)
nt_epochs = 100
nt_config = Struct()
nt_config.learningRate = 0.8
nt_config.maxIter = nt_epochs
nt_config.nCorrection = 50
nt_config.tolFun = 1.0 * np.finfo(float).eps

#%% DEFINING THE MODEL

class PhysicsInformedNN(object):
  def __init__(self, layers, optimizer, logger, ub, lb, nu):
    # New descriptive Keras model [2, 20, …, 20, 1]
    self.u_model = tf.keras.Sequential()
    self.u_model.add(tf.keras.layers.InputLayer(input_shape=(layers[0],)))
    self.u_model.add(tf.keras.layers.Lambda(lambda X: 2.0*(X - lb)/(ub - lb) - 1.0))
    for width in layers[1:]:
        self.u_model.add(tf.keras.layers.Dense(width, activation=tf.nn.tanh))

    # Computing the sizes of weights/biases for future decomposition
    self.sizes_w = []
    self.sizes_b = []
    for i, width in enumerate(layers):
      if i != 1:
        self.sizes_w.append(int(width * layers[1]))
        self.sizes_b.append(int(width if i != 0 else layers[1]))

    self.nu = nu
    self.optimizer = optimizer
    self.logger = logger

    self.dtype = tf.float32
    
  # Defining custom loss
  def __loss(self, u, u_pred):
    f_pred = self.f_model()
    return tf.reduce_mean(tf.square(u - u_pred)) + \
      tf.reduce_mean(tf.square(f_pred))

  def __grad(self, X, u):
    with tf.GradientTape() as tape:
      loss_value = self.__loss(u, self.u_model(X))
    return loss_value, tape.gradient(loss_value, self.__wrap_training_variables())

  def __wrap_training_variables(self):
    var = self.u_model.trainable_variables
    return var

  def __loss_and_flat_grad(self, w):
    with tf.GradientTape() as tape:
      self.set_weights(w)
      loss_value = self.__loss(self.u, self.u_model(self.X_u))
    grad = tape.gradient(loss_value, self.u_model.trainable_variables)
    return loss_value, self.__flatten_grad(grad)

  def __flatten_grad(self, grad):
    grad_flat = []
    for g in grad:
      grad_flat.append(tf.reshape(g, [-1]))
    return tf.concat(grad_flat, 0)

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
      u = self.u_model(X_f)
      # Deriving INSIDE the tape (since we’ll need the x derivative of this later, u_xx)
      u_x = tape.gradient(u, self.x_f)
    
    # Getting the other derivatives
    u_xx = tape.gradient(u_x, self.x_f)
    u_t = tape.gradient(u, self.t_f)

    # Letting the tape go
    del tape

    nu = self.get_params(numpy=True)

    # Buidling the PINNs
    return u_t + u*u_x - nu*u_xx

  def get_params(self, numpy=False):
    return self.nu

  def get_weights(self):
    w = []
    for layer in self.u_model.layers[1:]:
      weights_biases = layer.get_weights()
      weights = weights_biases[0].flatten()
      biases = weights_biases[1]
      w.extend(weights)
      w.extend(biases)
    return tf.convert_to_tensor(w, dtype=self.dtype)

  def set_weights(self, w):
    for i, layer in enumerate(self.u_model.layers[1:]):
      start_weights = sum(self.sizes_w[:i]) + sum(self.sizes_b[:i])
      end_weights = sum(self.sizes_w[:i+1]) + sum(self.sizes_b[:i])
      weights = w[start_weights:end_weights]
      w_div = int(self.sizes_w[i] / self.sizes_b[i])
      weights = tf.reshape(weights, [w_div, self.sizes_b[i]])
      biases = w[end_weights:end_weights + self.sizes_b[i]]
      weights_biases = [weights, biases]
      layer.set_weights(weights_biases)

  def error(self, x_star, u_star):
    u_pred, _ = self.predict(x_star)
    return np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)

  def summary(self):
    return self.u_model.summary()

  # The training function
  def fit(self, X_u, u, X_f, tf_epochs=5000, nt_config=Struct()):
    self.logger.log_train_start(self)

    # Creating the tensors
    self.X_u = tf.convert_to_tensor(X_u, dtype=self.dtype)
    self.u = tf.convert_to_tensor(u, dtype=self.dtype)
    # Separating the collocation coordinates
    self.x_f = tf.convert_to_tensor(X_f[:, 0:1], dtype=self.dtype)
    self.t_f = tf.convert_to_tensor(X_f[:, 1:2], dtype=self.dtype)

    self.logger.log_train_opt("Adam")
    # Training loop
    for epoch in range(tf_epochs):
      # Optimization step
      loss_value, grads = self.__grad(self.X_u, self.u)
      self.optimizer.apply_gradients(zip(grads, self.__wrap_training_variables()))
      self.logger.log_train_epoch(epoch, loss_value)
    
    self.logger.log_train_opt("LBFGS")

    # tfp.optimizer.lbfgs_minimize(
    #   self.loss_and_flat_grad_function,
    #   initial_position=self.get_weights(),
    #   num_correction_pairs=nt_config.nCorrection,
    #   max_iterations=nt_config.maxIter,
    #   f_relative_tolerance=nt_config.tolFun,
    #   tolerance=nt_config.tolFun,
    #   parallel_iterations=6)

    lbfgs(self.__loss_and_flat_grad,
      self.get_weights(),
      nt_config, Struct(), True, logger)

    self.logger.log_train_end(tf_epochs + nt_config.maxIter)

  def predict(self, X_star):
    u_star = self.u_model(X_star)
    f_star = self.f_model()
    return u_star, f_star

#%% TRAINING THE MODEL

# Getting the data
path = os.path.join(appDataPath, "burgers_shock.mat")
x, t, X, T, Exact_u, X_star, u_star, \
  X_u_train, u_train, X_f_train, ub, lb = prep_data(path, N_u, N_f, noise=0.0)

logger = Logger(X_star, u_star, frequency=10)

# Creating the model and training
pinn = PhysicsInformedNN(layers, tf_optimizer, logger, ub, lb, nu=0.01/np.pi)
pinn.fit(X_u_train, u_train, X_f_train, tf_epochs, nt_config)

# Getting the model predictions, from the same (x,t) that the predictions were previously gotten from
u_pred, f_pred = pinn.predict(X_star)

#%% PLOTTING
plot_inf_cont_results(X_star, u_pred.numpy().flatten(), X_u_train, u_train,
  Exact_u, X, T, x, t)