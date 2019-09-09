#%% IMPORTING/SETTING UP PATHS

import sys
import os
import json
import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp

# Manually making sure the numpy random seeds are "the same" on all devices
np.random.seed(1234)
tf.random.set_seed(1234)

#%% LOCAL IMPORTS

eqnPath = "1d-burgers"
sys.path.append(eqnPath)
sys.path.append("utils")
from custom_lbfgs import lbfgs, Struct
from burgersutil import prep_data, plot_ide_disc_results
from neuralnetwork import NeuralNetwork
from logger import Logger

#%% HYPER PARAMETERS

if len(sys.argv) > 1:
  with open(sys.argv[1]) as hpFile:
    hp = json.load(hpFile)
else:
  hp = {}
  # Data size on the solution u
  hp["N_0"] = 199
  hp["N_1"] = 201
  # DeepNN topology (1-sized input [x], 3 hidden layer of 50-width, q-sized output defined later [u_1^n(x), ..., u_{q+1}^n(x)]
  hp["layers"] = [1, 50, 50, 50, 0]
  # Setting up the TF SGD-based optimizer (set tf_epochs=0 to cancel it)
  hp["tf_epochs"] = 100
  hp["tf_lr"] = 0.001
  hp["tf_b1"] = 0.9
  hp["tf_eps"] = None 
  # Setting up the quasi-newton LBGFS optimizer (set nt_epochs=0 to cancel it)
  hp["nt_epochs"] = 2000
  hp["nt_lr"] = 0.8
  hp["nt_ncorr"] = 50

#%% DEFINING THE MODEL

class BurgersInformedNN(NeuralNetwork):
  def __init__(self, hp, logger, dt, lb, ub, q, IRK_alpha, IRK_beta):
    super().__init__(hp, logger, ub, lb)

    self.dt = dt
    self.q = max(q, 1)
    self.IRK_alpha = IRK_alpha
    self.IRK_beta = IRK_beta

  def autograd(self, U, x, dummy):
    # Using the new GradientTape paradigm of TF2.0,
    # which keeps track of operations to get the gradient at runtime
    with tf.GradientTape(persistent=True) as tape:
      # Watching the two inputs we’ll need later, x and t
      tape.watch(x)
      tape.watch(dummy)

      # Getting the prediction
      U = self.model(x) # shape=(len(x), q)

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
    U = self.model(x)
    if customDummy != None:
      dummy = customDummy
    else:
      dummy = self.dummy_x_0
    U_x, U_xx = self.autograd(U, x, dummy)

    # Buidling the PINNs
    l1 = self.lambda_1
    l2 = tf.exp(self.lambda_2)
    N = l1*U*U_x - l2*U_xx # shape=(len(x), q)
    return U + self.dt*tf.matmul(N, self.IRK_alpha.T)

  def U_1_model(self, x, customDummy=None):
    U = self.model(x)
    #dummy = customDummy or self.dummy_x_1
    if customDummy != None:
      dummy = customDummy
    else:
      dummy = self.dummy_x_1
    U_x, U_xx = self.autograd(U, x, dummy)

    # Buidling the PINNs, shape = (len(x), q+1), IRK shape = (q, q+1)
    l1 = self.lambda_1
    l2 = tf.exp(self.lambda_2)
    N = -l1*U*U_x + l2*U_xx # shape=(len(x), q)
    return U + self.dt*tf.matmul(N, (self.IRK_beta - self.IRK_alpha).T)

  # Defining custom loss
  def loss(self, x_0, u_0, x_1, u_1):
    u_0_pred = self.U_0_model(x_0)
    u_1_pred = self.U_1_model(x_1)
    return tf.reduce_sum(tf.square(u_0_pred - u_0)) + \
      tf.reduce_sum(tf.square(u_1_pred - u_1))

  def grad(self, x_0, u_0, x_1, u_1):
    with tf.GradientTape() as tape:
      loss_value = self.loss(x_0, u_0, x_1, u_1)
    return loss_value, tape.gradient(loss_value, self.wrap_training_variables())

  def wrap_training_variables(self):
    var = self.model.trainable_variables
    var.extend([self.lambda_1, self.lambda_2])
    return var

  def get_weights(self):
      w = super().get_weights(convert_to_tensor=False)
      w.extend(self.lambda_1.numpy())
      w.extend(self.lambda_2.numpy())
      return tf.convert_to_tensor(w, dtype=self.dtype)

  def set_weights(self, w):
    super().set_weights(w)
    self.lambda_1.assign([w[-2]])
    self.lambda_2.assign([w[-1]])

  def get_params(self, numpy=False):
    l1 = self.lambda_1
    l2 = tf.exp(self.lambda_2)
    if numpy:
      return l1.numpy()[0], l2.numpy()[0]
    return l1, l2

  def createDummy(self, x):
    return tf.ones([x.shape[0], self.q], dtype=self.dtype)

  # The training function
  def fit(self, x_0, u_0, x_1, u_1):
    self.logger.log_train_start(self)

    # Creating the tensors
    x_0 = tf.convert_to_tensor(x_0, dtype=self.dtype)
    u_0 = tf.convert_to_tensor(u_0, dtype=self.dtype)
    x_1 = tf.convert_to_tensor(x_1, dtype=self.dtype)
    u_1 = tf.convert_to_tensor(u_1, dtype=self.dtype)

    self.lambda_1 = tf.Variable([0.0], dtype=self.dtype)
    self.lambda_2 = tf.Variable([-6.0], dtype=self.dtype)

    # Creating dummy tensors for the gradients
    self.dummy_x_0 = self.createDummy(x_0)
    self.dummy_x_1 = self.createDummy(x_1)

    def log_train_epoch(epoch, loss, is_iter):
      l1, l2 = self.get_params(numpy=True)
      custom = f"l1 = {l1:5f}  l2 = {l2:8f}"
      self.logger.log_train_epoch(epoch, loss, custom, is_iter)

    self.logger.log_train_opt("Adam")
    for epoch in range(hp["tf_epochs"]):
      # Optimization step
      loss_value, grads = self.grad(x_0, u_0, x_1, u_1)
      self.tf_optimizer.apply_gradients(
        zip(grads, self.wrap_training_variables()))
      log_train_epoch(epoch, loss_value, False)
    
    self.logger.log_train_opt("LBFGS")
    def loss_and_flat_grad(w):
      with tf.GradientTape() as tape:
        self.set_weights(w)
        tape.watch(self.lambda_1)
        tape.watch(self.lambda_2)
        loss_value = self.loss(x_0, u_0, x_1, u_1)
      grad = tape.gradient(loss_value, self.wrap_training_variables())
      grad_flat = []
      for g in grad:
        grad_flat.append(tf.reshape(g, [-1]))
      grad_flat =  tf.concat(grad_flat, 0)
      return loss_value, grad_flat
    lbfgs(loss_and_flat_grad,
      self.get_weights(),
      self.nt_config, Struct(), True, log_train_epoch)
    
    l1, l2 = self.get_params(numpy=True)
    self.logger.log_train_end(hp["tf_epochs"], f"l1 = {l1:5f}  l2 = {l2:8f}")

  def predict(self, x_star):
    x_star = tf.convert_to_tensor(x_star, dtype=self.dtype)
    dummy = self.createDummy(x_star)
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
path = os.path.join(eqnPath, "data", "burgers_shock.mat")
x_0, u_0, x_1, u_1, x_star, t_star, dt, q, \
  Exact_u, IRK_alpha, IRK_beta = prep_data(path, N_0=hp["N_0"], N_1=hp["N_1"],
  lb=lb, ub=ub, noise=0.0, idx_t_0=idx_t_0, idx_t_1=idx_t_1)
lambdas_star = (1.0, 0.01/np.pi)

# Setting the output layer dynamically
hp["layers"][-1] = q
 
# Creating the model
logger = Logger(frequency=10)
pinn = BurgersInformedNN(hp, logger, dt, lb, ub, q, IRK_alpha, IRK_beta)

# Defining the error function and training
def error():
  l1, l2 = pinn.get_params(numpy=True)
  l1_star, l2_star = lambdas_star
  error_lambda_1 = np.abs(l1 - l1_star) / l1_star
  error_lambda_2 = np.abs(l2 - l2_star) / l2_star
  return (error_lambda_1 + error_lambda_2) / 2
logger.set_error_fn(error)
pinn.fit(x_0, u_0, x_1, u_1)

# Getting the model predictions
U_0_pred, U_1_pred = pinn.predict(x_star)
lambda_1_pred, lambda_2_pred = pinn.get_params(numpy=True)

# Noisy case (same as before with a different noise)
x_0, u_0, x_1, u_1, x_star, t_star, dt, q, \
  Exact_u, IRK_alpha, IRK_beta = prep_data(path, N_0=hp["N_0"], N_1=hp["N_1"],
  lb=lb, ub=ub, noise=0.01, idx_t_0=idx_t_0, idx_t_1=idx_t_1)
hp["layers"][-1] = q
pinn = BurgersInformedNN(hp, logger, dt, lb, ub, q, IRK_alpha, IRK_beta)
pinn.fit(x_0, u_0, x_1, u_1)
U_0_pred, U_1_pred = pinn.predict(x_star)
lambda_1_pred_noisy, lambda_2_pred_noisy = pinn.get_params(numpy=True)

print("l1: ", lambda_1_pred)
print("l2: ", lambda_2_pred)
print("noisy l1: ", lambda_1_pred_noisy)
print("noisy l2: ", lambda_2_pred_noisy)

#%% PLOTTING
plot_ide_disc_results(x_star, t_star, idx_t_0, idx_t_1, x_0, u_0, x_1, u_1,
  ub, lb, U_1_pred, Exact_u,
  lambda_1_pred, lambda_1_pred_noisy, lambda_2_pred, lambda_2_pred_noisy, x_star, t_star,
  save_path=eqnPath, save_hp=hp)