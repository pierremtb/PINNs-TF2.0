#%% IMPORTING/SETTING UP PATHS

import sys
import os
import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
import json

# Manually making sure the numpy random seeds are "the same" on all devices
np.random.seed(1234)
tf.random.set_seed(1234)

#%% LOCAL IMPORTS

eqnPath = "1d-burgers"
sys.path.append(eqnPath)
sys.path.append("utils")
from burgersutil import prep_data, plot_ide_cont_results
from neuralnetwork import NeuralNetwork
from logger import Logger

#%% HYPER PARAMETERS

if len(sys.argv) > 1:
    with open(sys.argv[1]) as hpFile:
        hp = json.load(hpFile)
else:
    hp = {}
  # Data size on the solution u
  hp["N_u"] = 2000
  # DeepNN topology (2-sized input [x t], 8 hidden layer of 20-width, 1-sized output [u]
  hp["layers"] = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]
  # Setting up the TF SGD-based optimizer (set tf_epochs=0 to cancel it)
  hp["tf_epochs"] = 100
  hp["tf_lr"] = 0.001
  hp["tf_b1"] = 0.9
  hp["tf_eps"] = None
  # Setting up the quasi-newton LBGFS optimizer (set nt_epochs=0 to cancel it)
  hp["nt_epochs"] = 500
  hp["nt_lr"] = 0.8
  hp["nt_ncorr"] = 50
  hp["log_frequency"] = 10

#%% DEFINING THE MODEL

class BurgersInformedNN(NeuralNetwork):
    def __init__(self, hp, logger, ub, lb):
        super().__init__(hp, logger, ub, lb)

    # Defining the two additional trainable variables for identification
    self.lambda_1 = tf.Variable([0.0], dtype=self.dtype)
    self.lambda_2 = tf.Variable([-6.0], dtype=self.dtype)

  # The actual PINN
  def f_model(self, X_u):
      l1, l2 = self.get_params()
    # Separating the collocation coordinates
    x_f = tf.convert_to_tensor(X_u[:, 0:1], dtype=self.dtype)
    t_f = tf.convert_to_tensor(X_u[:, 1:2], dtype=self.dtype)

    # Using the new GradientTape paradigm of TF2.0,
    # which keeps track of operations to get the gradient at runtime
    with tf.GradientTape(persistent=True) as tape:
        # Watching the two inputs we’ll need later, x and t
      tape.watch(x_f)
      tape.watch(t_f)
      # Packing together the inputs
      X_f = tf.stack([x_f[:,0], t_f[:,0]], axis=1)


      # Getting the prediction
      u = self.model(X_f)
      # Deriving INSIDE the tape (since we’ll need the x derivative of this later, u_xx)
      u_x = tape.gradient(u, x_f)

    # Getting the other derivatives
    u_xx = tape.gradient(u_x, x_f)
    u_t = tape.gradient(u, t_f)

    # Letting the tape go
    del tape

    # Buidling the PINNs
    return u_t + l1*u*u_x - l2*u_xx

# Defining custom loss
  def loss(self, u, u_pred):
      f_pred = self.f_model(self.X_u)
    return tf.reduce_mean(tf.square(u - u_pred)) + \
            tf.reduce_mean(tf.square(f_pred))

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

def fit(self, X_u, u):
    self.X_u =  tf.convert_to_tensor(X_u, dtype=self.dtype)
    super().fit(X_u, u)

  # # The training function
  # def fit(self, X_u, u, tf_epochs, nt_config):
  #   self.logger.log_train_start(self)

  #   # Creating the tensors
  #   X_u = tf.convert_to_tensor(X_u, dtype=self.dtype)
  #   u = tf.convert_to_tensor(u, dtype=self.dtype)

  #   def log_train_epoch(epoch, loss, is_iter):
  #     l1, l2 = self.get_params(numpy=True)
  #     custom = f"l1 = {l1:5f}  l2 = {l2:8f}"
  #     self.logger.log_train_epoch(epoch, loss, custom, is_iter)

  #   self.logger.log_train_opt("Adam")
  #   for epoch in range(tf_epochs):
  #     # Optimization step
  #     loss_value, grads = self.grad(X_u, u)
  #     self.optimizer.apply_gradients(
  #       zip(grads, self.__wrap_training_variables()))
  #     log_train_epoch(epoch, loss_value, False)

  #   self.logger.log_train_opt("LBFGS")
  #   def loss_and_flat_grad(w):
  #     with tf.GradientTape() as tape:
  #       self.set_weights(w)
  #       tape.watch(self.lambda_1)
  #       tape.watch(self.lambda_2)
  #       loss_value = self.loss(X_u, u, self.model(X_u))
  #     grad = tape.gradient(loss_value, self.__wrap_training_variables())
  #     grad_flat = []
  #     for g in grad:
  #       grad_flat.append(tf.reshape(g, [-1]))
  #     grad_flat =  tf.concat(grad_flat, 0)
  #     return loss_value, grad_flat
  #   # tfp.optimizer.lbfgs_minimize(
  #   #   loss_and_flat_grad,
  #   #   initial_position=self.get_weights(),
  #   #   num_correction_pairs=nt_config.nCorrection,
  #   #   max_iterations=nt_config.maxIter,
  #   #   f_relative_tolerance=nt_config.tolFun,
  #   #   tolerance=nt_config.tolFun,
  #   #   parallel_iterations=6)
  #   lbfgs(loss_and_flat_grad,
  #     self.get_weights(),
  #     nt_config, Struct(), True, log_train_epoch)

  #   l1, l2 = self.get_params(numpy=True)
  #   self.logger.log_train_end(tf_epochs, f"l1 = {l1:5f}  l2 = {l2:8f}")

  def predict(self, X_star):
      u_star = self.model(X_star)
    f_star = self.f_model(X_star)
    return u_star.numpy(), f_star.numpy()

#%% TRAINING THE MODEL

# Getting the data
path = os.path.join(eqnPath, "data", "burgers_shock.mat")
x, t, X, T, Exact_u, X_star, u_star, \
        X_u_train, u_train, ub, lb = prep_data(path, hp["N_u"], noise=0.0)
lambdas_star = (1.0, 0.01/np.pi)

# Creating the model
logger = Logger(hp)
pinn = BurgersInformedNN(hp, logger, ub, lb)

# Defining the error function and training
def error():
    l1, l2 = pinn.get_params(numpy=True)
  l1_star, l2_star = lambdas_star
  error_lambda_1 = np.abs(l1 - l1_star) / l1_star
  error_lambda_2 = np.abs(l2 - l2_star) / l2_star
  return (error_lambda_1 + error_lambda_2) / 2
logger.set_error_fn(error)
pinn.fit(X_u_train, u_train)

# Getting the model predictions, from the same (x,t) that the predictions were previously gotten from
u_pred, f_pred = pinn.predict(X_star)
lambda_1_pred, lambda_2_pred = pinn.get_params(numpy=True)

# Noise case
x, t, X, T, Exact_u, X_star, u_star, \
        X_u_train, u_train, ub, lb = prep_data(path, hp["N_u"], noise=0.01)
pinn = BurgersInformedNN(hp, logger, ub, lb)
pinn.fit(X_u_train, u_train)
lambda_1_pred_noise, lambda_2_pred_noise = pinn.get_params(numpy=True)

print("l1: ", lambda_1_pred)
print("l2: ", lambda_2_pred)
print("l1_noise: ", lambda_1_pred_noise)
print("l2_noise: ", lambda_2_pred_noise)


#%% PLOTTING
plot_ide_cont_results(X_star, u_pred, X_u_train, u_train,
        Exact_u, X, T, x, t, lambda_1_pred, lambda_1_pred_noise, lambda_2_pred, lambda_2_pred_noise)
