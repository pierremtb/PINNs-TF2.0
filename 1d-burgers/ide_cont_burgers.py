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

from burgersutil import prep_data, Logger, plot_ide_cont_results, appDataPath

#%% HYPER PARAMETERS

# Data size on the solution u
N_u = 2000
# DeepNN topology (2-sized input [x t], 8 hidden layer of 20-width, 1-sized output [u]
layers = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]
# Creating the optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
epochs = 5000

#%% DEFINING THE MODEL

class PhysicsInformedNN(object):
  def __init__(self, layers, optimizer, logger):
    # New descriptive Keras model [2, 20, …, 20, 1]
    self.u_model = tf.keras.Sequential()
    self.u_model.add(tf.keras.layers.InputLayer(input_shape=(layers[0],)))
    for width in layers[1:]:
        self.u_model.add(tf.keras.layers.Dense(width, activation=tf.nn.tanh))

    self.dtype = tf.float32

    # Defining the two additional trainable variables for identification
    self.lambda_1 = tf.Variable([0.0], dtype=self.dtype)
    self.lambda_2 = tf.Variable([-6.0], dtype=self.dtype)
    
    self.optimizer = optimizer
    self.logger = logger

  # The actual PINN
  def __f_model(self):
    l1, l2 = self.get_params()

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

    # Buidling the PINNs
    return u_t + l1*u*u_x - l2*u_xx

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
    var.extend([self.lambda_1, self.lambda_2])
    return var

  def get_params(self, numpy=False):
    l1 = self.lambda_1
    l2 = tf.exp(self.lambda_2)
    if numpy:
      return l1.numpy()[0], l2.numpy()[0]
    return l1, l2

  def error(self, x_star, u_star):
    l1, l2 = self.get_params(numpy=True)
    error_lambda_1 = np.abs(l1 - 1.0)/1.0 *100
    nu = 0.01 / np.pi
    error_lambda_2 = np.abs(l2 - nu)/nu * 100
    return (error_lambda_1 + error_lambda_2) / 2

  def summary(self):
    return self.u_model.summary()

  # The training function
  def fit(self, X_u, u, epochs=1, log_epochs=50):
    self.logger.log_train_start(self)

    # Creating the tensors
    self.X_u = tf.convert_to_tensor(X_u, dtype=self.dtype)
    self.u = tf.convert_to_tensor(u, dtype=self.dtype)
    # Separating the collocation coordinates (here the same as data)
    self.x_f = tf.convert_to_tensor(X_u[:, 0:1], dtype=self.dtype)
    self.t_f = tf.convert_to_tensor(X_u[:, 1:2], dtype=self.dtype)

    # Training loop
    for epoch in range(epochs):
      # Optimization step
      loss_value, grads = self.__grad(self.X_u, self.u)
      self.optimizer.apply_gradients(zip(grads, self.__wrap_training_variables()))

      # Logging every so often
      if epoch % log_epochs == 0:
        l1, l2 = self.get_params(numpy=True)
        custom = f"l1 = {l1:5f}  l2 = {l2:8f}"
        self.logger.log_train_epoch(epoch, loss_value, custom)
    
    self.logger.log_train_end(epochs, f"l1 = {l1:5f}  l2 = {l2:8f}")

  def predict(self, X_star):
    u_star = self.u_model(X_star)
    f_star = self.__f_model()
    return u_star.numpy(), f_star.numpy()

#%% TRAINING THE MODEL

# Getting the data
path = os.path.join(appDataPath, "burgers_shock.mat")
x, t, X, T, Exact_u, X_star, u_star, \
  X_u_train, u_train = prep_data(path, N_u, noise=0.0)

logger = Logger(X_star, u_star)

# Creating the model and training
pinn = PhysicsInformedNN(layers, optimizer, logger)
pinn.fit(X_u_train, u_train, epochs)

# Getting the model predictions, from the same (x,t) that the predictions were previously gotten from
u_pred, f_pred = pinn.predict(X_star)
lambda_1_pred, lambda_2_pred = pinn.get_params(numpy=True)
print("l1: ", lambda_1_pred)
print("l2: ", lambda_2_pred)

# Noise case
x, t, X, T, Exact_u, X_star, u_star, \
  X_u_train, u_train = prep_data(path, N_u, noise=0.01)
logger = Logger(X_star, u_star)
pinn = PhysicsInformedNN(layers, optimizer, logger)
pinn.fit(X_u_train, u_train, epochs)
lambda_1_pred_noise, lambda_2_pred_noise = pinn.get_params(numpy=True)
print("l1_noise: ", lambda_1_pred_noise)
print("l2_noise: ", lambda_2_pred_noise)


#%% PLOTTING
plot_ide_cont_results(X_star, u_pred, X_u_train, u_train,
  Exact_u, X, T, x, t, lambda_1_pred, lambda_1_pred_noise, lambda_2_pred, lambda_2_pred_noise)