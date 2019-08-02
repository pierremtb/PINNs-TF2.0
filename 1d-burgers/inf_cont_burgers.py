#%% IMPORTING/SETTING UP PATHS

import sys
import os
import tensorflow as tf
import numpy as np

# Manually making sure the numpy random seeds are "the same" on all devices, for reproducibility in random processes
np.random.seed(1234)
# Same for tensorflow
tf.random.set_seed(1234)

#%% LOCAL IMPORTS

from burgersutil import prep_data, Logger, plot_inf_cont_results, appDataPath

#%% HYPER PARAMETERS

# Data size on the solution u
N_u = 50
# Collocation points size, where we’ll check for f = 0
N_f = 10000
# DeepNN topology (2-sized input [x t], 8 hidden layer of 20-width, 1-sized output [u]
layers = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]
# Creating the optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.15, beta_1=0.99, epsilon=1e-1)
epochs = 2000

#%% DEFINING THE MODEL

class PhysicsInformedNN(object):
  def __init__(self, layers, optimizer, logger, ub, lb, nu):
    # New descriptive Keras model [2, 20, …, 20, 1]
    self.u_model = tf.keras.Sequential()
    self.u_model.add(tf.keras.layers.InputLayer(input_shape=(layers[0],)))
    self.u_model.add(tf.keras.layers.Lambda(lambda X: 2.0*(X - lb)/(ub - lb) - 1.0))
    for width in layers[1:]:
        self.u_model.add(tf.keras.layers.Dense(width, activation=tf.nn.tanh))
    
    self.dtype = tf.float32

    self.nu = nu
    self.optimizer = optimizer
    self.logger = logger

  # The actual PINN
  def __f_model(self):
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

  # Defining custom loss
  def __loss(self, u, u_pred):
    f_pred = self.__f_model()
    return tf.reduce_mean(tf.square(u - u_pred)) + \
      tf.reduce_mean(tf.square(f_pred))

  def __grad(self, X, u):
    with tf.GradientTape() as tape:
      loss_value = self.__loss(u, self.u_model(X))
    return loss_value, tape.gradient(loss_value, self.__wrap_training_variables())

  def __wrap_training_variables(self):
    var = self.u_model.trainable_variables
    return var

  def get_params(self, numpy=False):
    return self.nu

  def error(self, x_star, u_star):
    u_pred, _ = self.predict(x_star)
    return np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)

  def summary(self):
    return self.u_model.summary()

  # The training function
  def fit(self, X_u, u, X_f, epochs=1, log_epochs=50):
    self.logger.log_train_start(self)

    # Creating the tensors
    self.X_u = tf.convert_to_tensor(X_u, dtype=self.dtype)
    self.u = tf.convert_to_tensor(u, dtype=self.dtype)
    # Separating the collocation coordinates
    self.x_f = tf.convert_to_tensor(X_f[:, 0:1], dtype=self.dtype)
    self.t_f = tf.convert_to_tensor(X_f[:, 1:2], dtype=self.dtype)

    # Training loop
    for epoch in range(epochs):
      # Optimization step
      loss_value, grads = self.__grad(self.X_u, self.u)
      self.optimizer.apply_gradients(zip(grads, self.__wrap_training_variables()))

      # Logging every so often
      if epoch % log_epochs == 0:
        self.logger.log_train_epoch(epoch, loss_value)
    
    self.logger.log_train_end(epochs)

  def predict(self, X_star):
    u_star = self.u_model(X_star)
    f_star = self.__f_model()
    return u_star, f_star

#%% TRAINING THE MODEL

# Getting the data
path = os.path.join(appDataPath, "burgers_shock.mat")
x, t, X, T, Exact_u, X_star, u_star, \
  X_u_train, u_train, X_f_train, ub, lb = prep_data(path, N_u, N_f, noise=0.0)

logger = Logger(X_star, u_star)

# Creating the model and training
pinn = PhysicsInformedNN(layers, optimizer, logger, ub, lb, nu=0.01/np.pi)
pinn.fit(X_u_train, u_train, X_f_train, epochs)

# Getting the model predictions, from the same (x,t) that the predictions were previously gotten from
u_pred, f_pred = pinn.predict(X_star)

#%% PLOTTING
plot_inf_cont_results(X_star, u_pred.numpy().flatten(), X_u_train, u_train,
  Exact_u, X, T, x, t)