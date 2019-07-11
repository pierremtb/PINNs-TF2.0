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

from burgersutil import prep_data, Logger, plot_disc_results, appDataPath

#%% HYPER PARAMETERS

# Data size on initial condition on u
N_n = 250
# Number of RK stages
q = 500
# DeepNN topology (1-sized input [x], 3 hidden layer of 50-width, q+1-sized output [u_1^n(x), ..., u_{q+1}^n(x)]
layers = [1, 50, 50, 50, q + 1]
# Creating the optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
epochs = 5000

#%% DEFINING THE MODEL

class PhysicsInformedNN(object):
  def __init__(self, layers, optimizer, logger, dt, lb, ub, nu, q, IRK_weights, IRK_times):
    self.lb = lb
    self.ub = ub
    self.nu = nu

    self.dt = dt

    self.q = max(q,1)
    self.IRK_weights = IRK_weights
    self.IRK_times = IRK_times

    # New descriptive Keras model [2, 50, …, 50, q+1]
    self.u_1_model = tf.keras.Sequential()
    self.u_1_model.add(tf.keras.layers.InputLayer(input_shape=(layers[0],)))
    for width in layers[1:]:
        self.u_1_model.add(tf.keras.layers.Dense(width, activation=tf.nn.tanh, kernel_initializer='glorot_normal'))
    # print(self.u_1_model.summary())

    self.dtype = tf.float32

    self.optimizer = optimizer
    self.logger = logger

  # The actual PINN
  def __u_0_model(self):
    # Using the new GradientTape paradigm of TF2.0,
    # which keeps track of operations to get the gradient at runtime
    with tf.GradientTape(persistent=True) as tape:
      # Watching the two inputs we’ll need later, x and t
      x = self.x_0
      tape.watch(x)

      # Getting the prediction
      u_1 = self.u_1_model(x)
      u = u_1[:, :-1]
      # Deriving INSIDE the tape (since we’ll need the x derivative of this later, u_xx)
      u_x = tape.gradient(u, x)
    
    # Getting the other derivatives
    u_xx = tape.gradient(u_x, x)

    # Letting the tape go
    del tape

    nu = self.get_params(numpy=True)

    f = -u*u_x + nu*u_xx

    u_0 = u_1 - self.dt*tf.matmul(f, self.IRK_weights.T)

    # Buidling the PINNs
    return u_0

  # Defining custom loss
  def __loss(self, u_0, u_1_pred):
    u_0_pred = self.__u_0_model()
    return tf.reduce_mean(tf.square(u_0 - u_0_pred)) + \
      tf.reduce_mean(tf.square(u_1_pred))

  def __grad(self, X, u):
    with tf.GradientTape() as tape:
      loss_value = self.__loss(u, self.u_1_model(X))
    return loss_value, tape.gradient(loss_value, self.__wrap_training_variables())

  def __wrap_training_variables(self):
    var = self.u_1_model.trainable_variables
    return var

  def get_params(self, numpy=False):
    return self.nu

  # The training function
  def fit(self, x_0, u_0, x_1, epochs=1, log_epochs=50):
    self.logger.log_train_start(self.u_1_model)

    # Creating the tensors
    self.x_0 = tf.convert_to_tensor(x_0, dtype=self.dtype)
    self.u_0 = tf.convert_to_tensor(u_0, dtype=self.dtype)
    self.x_1 = tf.convert_to_tensor(x_1, dtype=self.dtype)

    # Creating dummy tensors for the gradients
    self.dummy_x0_tf = np.ones((self.x_0.shape[0], self.q))
    self.dummy_x1_tf = np.ones((self.x_1.shape[0], self.q+1))

    # Training loop
    for epoch in range(epochs):
      # Optimization step
      loss_value, grads = self.__grad(self.x_1, self.u_0)
      self.optimizer.apply_gradients(zip(grads, self.__wrap_training_variables()))

      # Logging every so often
      if epoch % log_epochs == 0:
        self.logger.log_train_epoch(epoch, loss_value)
    
    self.logger.log_train_end(epochs)

  def predict(self, x_star):
    u_star = self.u_1_model(x_star)
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

logger = Logger(x_star, u_star, pred_keep_last_column=True)

# Creating the model and training
pinn = PhysicsInformedNN(layers, optimizer, logger, dt, lb, ub, nu, q, IRK_weights, IRK_times)
pinn.fit(x_0, u_0, x_1, epochs)

# Getting the model predictions, from the same (x,t) that the predictions were previously gotten from
u_1_pred = pinn.predict(x_star)

#%% PLOTTING
plot_disc_results(x_star, idx_t_0, idx_t_1, x_0, u_0, ub, lb, u_1_pred, Exact_u, x, t)