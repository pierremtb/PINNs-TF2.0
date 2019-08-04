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

sys.path.append("1d-burgers")
from custom_lbfgs import lbfgs, Struct
from burgersutil import prep_data, Logger, plot_ide_disc_results, appDataPath

#%% HYPER PARAMETERS

# Data size on initial condition on u
N_0 = 199
N_1 = 201
# DeepNN topology (1-sized input [x], 3 hidden layer of 50-width, q-sized output defined later [u_1^n(x), ..., u_{q+1}^n(x)]
layers = [1, 50, 50, 50, 0]
# Setting up the TF SGD-based optimizer (set tf_epochs=0 to cancel it)
tf_epochs = 100
tf_optimizer = tf.keras.optimizers.Adam(
  lr=0.001,
  beta_1=0.9,
  beta_2=0.999,
  epsilon=1e-08)
# Setting up the quasi-newton LBGFS optimizer (set nt_epochs=0 to cancel it)
nt_epochs = 2000
nt_config = Struct()
nt_config.learningRate = 0.8
nt_config.maxIter = nt_epochs
nt_config.nCorrection = 50
nt_config.tolFun = 1.0 * np.finfo(float).eps

#%% DEFINING THE MODEL

class PhysicsInformedNN(object):
  def __init__(self, layers, optimizer, logger, dt, lb, ub, q, IRK_alpha, IRK_beta):
    self.lb = lb
    self.ub = ub

    self.dt = dt

    self.q = max(q,1)
    self.IRK_alpha = IRK_alpha
    self.IRK_beta = IRK_beta

    # Descriptive Keras model [2, 50, …, 50, q+1]
    self.U_model = tf.keras.Sequential()
    self.U_model.add(tf.keras.layers.InputLayer(input_shape=(layers[0],)))
    self.U_model.add(tf.keras.layers.Lambda(
      lambda X: 2.0*(X - lb)/(ub - lb) - 1.0))
    for width in layers[1:]:
        self.U_model.add(tf.keras.layers.Dense(
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

    self.optimizer = optimizer
    self.logger = logger

  def __autograd(self, U, x, dummy):
    # Using the new GradientTape paradigm of TF2.0,
    # which keeps track of operations to get the gradient at runtime
    with tf.GradientTape(persistent=True) as tape:
      # Watching the two inputs we’ll need later, x and t
      tape.watch(x)
      tape.watch(dummy)

      # Getting the prediction
      U = self.U_model(x) # shape=(len(x), q)

      # Deriving INSIDE the tape (2-step-dummy grad technique because U is a mat)
      g_U = tape.gradient(U, x, output_gradients=dummy)
      U_x = tape.gradient(g_U, dummy)
      g_U_x = tape.gradient(U_x, x, output_gradients=dummy)
    
    # Doing the last one outside the with, to optimize performance
    # Impossible to do for the earlier grad, because they’re needed after
    U_xx = tape.gradient(g_U_x, dummy)

    # Letting the tape go
    del tape
    return U_x, U_xx

  def U_0_model(self, x, customDummy=None):
    U = self.U_model(x)
    if customDummy != None:
      dummy = customDummy
    else:
      dummy = self.dummy_x_0
    U_x, U_xx = self.__autograd(U, x, dummy)

    # Buidling the PINNs
    l1 = self.lambda_1
    l2 = tf.exp(self.lambda_2)
    N = l1*U*U_x - l2*U_xx # shape=(len(x), q)
    return U + self.dt*tf.matmul(N, self.IRK_alpha.T)

  def U_1_model(self, x, customDummy=None):
    U = self.U_model(x)
    #dummy = customDummy or self.dummy_x_1
    if customDummy != None:
      dummy = customDummy
    else:
      dummy = self.dummy_x_1
    U_x, U_xx = self.__autograd(U, x, dummy)

    # Buidling the PINNs, shape = (len(x), q+1), IRK shape = (q, q+1)
    l1 = self.lambda_1
    l2 = tf.exp(self.lambda_2)
    N = -l1*U*U_x + l2*U_xx # shape=(len(x), q)
    return U + self.dt*tf.matmul(N, (self.IRK_beta - self.IRK_alpha).T)

  # Defining custom loss
  def __loss(self, x_0, u_0, x_1, u_1):
    u_0_pred = self.U_0_model(x_0)
    u_1_pred = self.U_1_model(x_1)
    return tf.reduce_sum(tf.square(u_0_pred - u_0)) + \
      tf.reduce_sum(tf.square(u_1_pred - u_1))

  def __grad(self, x_0, u_0, x_1, u_1):
    with tf.GradientTape() as tape:
      loss_value = self.__loss(x_0, u_0, x_1, u_1)
    return loss_value, tape.gradient(loss_value, self.__wrap_training_variables())

  def __wrap_training_variables(self):
    var = self.U_model.trainable_variables
    var.extend([self.lambda_1, self.lambda_2])
    return var

  def get_weights(self):
      w = []
      for layer in self.U_model.layers[1:]:
        weights_biases = layer.get_weights()
        weights = weights_biases[0].flatten()
        biases = weights_biases[1]
        w.extend(weights)
        w.extend(biases)
      w.extend(self.lambda_1.numpy())
      w.extend(self.lambda_2.numpy())
      return tf.convert_to_tensor(w, dtype=self.dtype)

  def set_weights(self, w):
    for i, layer in enumerate(self.U_model.layers[1:]):
      start_weights = sum(self.sizes_w[:i]) + sum(self.sizes_b[:i])
      end_weights = sum(self.sizes_w[:i+1]) + sum(self.sizes_b[:i])
      weights = w[start_weights:end_weights]
      w_div = int(self.sizes_w[i] / self.sizes_b[i])
      weights = tf.reshape(weights, [w_div, self.sizes_b[i]])
      biases = w[end_weights:end_weights + self.sizes_b[i]]
      weights_biases = [weights, biases]
      layer.set_weights(weights_biases)
    self.lambda_1.assign([w[-2]])
    self.lambda_2.assign([w[-1]])

  def get_params(self, numpy=False):
    l1 = self.lambda_1
    l2 = tf.exp(self.lambda_2)
    if numpy:
      return l1.numpy()[0], l2.numpy()[0]
    return l1, l2

  def summary(self):
    return self.U_model.summary()

  def __createDummy(self, x):
    return tf.ones([x.shape[0], self.q], dtype=self.dtype)

  # The training function
  def fit(self, x_0, u_0, x_1, u_1, tf_epochs=1):
    self.logger.log_train_start(self)

    # Creating the tensors
    x_0 = tf.convert_to_tensor(x_0, dtype=self.dtype)
    u_0 = tf.convert_to_tensor(u_0, dtype=self.dtype)
    x_1 = tf.convert_to_tensor(x_1, dtype=self.dtype)
    u_1 = tf.convert_to_tensor(u_1, dtype=self.dtype)

    self.lambda_1 = tf.Variable([0.0], dtype=self.dtype)
    self.lambda_2 = tf.Variable([-6.0], dtype=self.dtype)

    # Creating dummy tensors for the gradients
    self.dummy_x_0 = self.__createDummy(x_0)
    self.dummy_x_1 = self.__createDummy(x_1)

    def log_train_epoch(epoch, loss, is_iter):
      l1, l2 = self.get_params(numpy=True)
      custom = f"l1 = {l1:5f}  l2 = {l2:8f}"
      self.logger.log_train_epoch(epoch, loss, custom, is_iter)

    self.logger.log_train_opt("Adam")
    for epoch in range(tf_epochs):
      # Optimization step
      loss_value, grads = self.__grad(x_0, u_0, x_1, u_1)
      self.optimizer.apply_gradients(
        zip(grads, self.__wrap_training_variables()))
      log_train_epoch(epoch, loss_value, False)
    
    self.logger.log_train_opt("LBFGS")
    def loss_and_flat_grad(w):
      with tf.GradientTape() as tape:
        self.set_weights(w)
        tape.watch(self.lambda_1)
        tape.watch(self.lambda_2)
        loss_value = self.__loss(x_0, u_0, x_1, u_1)
      grad = tape.gradient(loss_value, self.__wrap_training_variables())
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
      nt_config, Struct(), True, log_train_epoch)
    
    l1, l2 = self.get_params(numpy=True)
    self.logger.log_train_end(tf_epochs, f"l1 = {l1:5f}  l2 = {l2:8f}")

  def predict(self, x_star):
    x_star = tf.convert_to_tensor(x_star, dtype=self.dtype)
    dummy = self.__createDummy(x_star)
    U_0_star = self.U_0_model(x_star, dummy)
    U_1_star = self.U_1_model(x_star, dummy)
    return U_0_star, U_1_star

#%% TRAINING THE MODEL

# Setup
lb = np.array([-1.0])
ub = np.array([1.0])
idx_t_0 = 10
skip = 80
idx_t_1 = idx_t_0 + skip

# Getting the data
path = os.path.join(appDataPath, "burgers_shock.mat")
x_0, u_0, x_1, u_1, x_star, t_star, dt, q, \
  Exact_u, IRK_alpha, IRK_beta = prep_data(path, N_0=N_0, N_1=N_1,
  lb=lb, ub=ub, noise=0.0, idx_t_0=idx_t_0, idx_t_1=idx_t_1)
lambdas_star = (1.0, 0.01/np.pi)

# Setting the output layer dynamically
layers[-1] = q
 
# Creating the model and training
logger = Logger(frequency=10)
pinn = PhysicsInformedNN(layers, tf_optimizer, logger, dt, lb, ub, q, IRK_alpha, IRK_beta)
def error():
  l1, l2 = pinn.get_params(numpy=True)
  l1_star, l2_star = lambdas_star
  error_lambda_1 = np.abs(l1 - l1_star) / l1_star
  error_lambda_2 = np.abs(l2 - l2_star) / l2_star
  return (error_lambda_1 + error_lambda_2) / 2
logger.set_error_fn(error)
pinn.fit(x_0, u_0, x_1, u_1, tf_epochs)

# Getting the model predictions
U_0_pred, U_1_pred = pinn.predict(x_star)
lambda_1_pred, lambda_2_pred = pinn.get_params(numpy=True)

# Noisy case (same as before with a different noise)
x_0, u_0, x_1, u_1, x_star, t_star, dt, q, \
  Exact_u, IRK_alpha, IRK_beta = prep_data(path, N_0=N_0, N_1=N_1,
  lb=lb, ub=ub, noise=0.01, idx_t_0=idx_t_0, idx_t_1=idx_t_1)
layers[-1] = q
pinn = PhysicsInformedNN(layers, tf_optimizer, logger, dt, lb, ub, q, IRK_alpha, IRK_beta)
pinn.fit(x_0, u_0, x_1, u_1, tf_epochs)
U_0_pred, U_1_pred = pinn.predict(x_star)
lambda_1_pred_noisy, lambda_2_pred_noisy = pinn.get_params(numpy=True)

print("l1: ", lambda_1_pred)
print("l2: ", lambda_2_pred)
print("noisy l1: ", lambda_1_pred_noisy)
print("noisy l2: ", lambda_2_pred_noisy)

#%% PLOTTING
plot_ide_disc_results(x_star, t_star, idx_t_0, idx_t_1, x_0, u_0, x_1, u_1,
  ub, lb, U_1_pred, Exact_u, lambda_1_pred, lambda_1_pred_noisy, lambda_2_pred, lambda_2_pred_noisy, x_star, t_star)