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

sys.path.append("1d-burgers")
from custom_lbfgs import lbfgs, Struct
from burgersutil import prep_data, Logger, plot_ide_cont_results, appDataPath

#%% HYPER PARAMETERS

# Data size on the solution u
N_u = 2000
# DeepNN topology (2-sized input [x t], 8 hidden layer of 20-width, 1-sized output [u]
layers = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]
# Setting up the TF SGD-based optimizer (set tf_epochs=0 to cancel it)
tf_epochs = 10000
tf_optimizer = tf.keras.optimizers.Adam(
  learning_rate=0.001)
# Setting up the quasi-newton LBGFS optimizer (set nt_epochs=0 to cancel it)
nt_epochs = 500
nt_config = Struct()
nt_config.learningRate = 0.8
nt_config.maxIter = nt_epochs
nt_config.nCorrection = 50
nt_config.tolFun = 1.0 * np.finfo(float).eps

#%% DEFINING THE MODEL

class PhysicsInformedNN(object):
  def __init__(self, layers, optimizer, logger, ub, lb):
    # Descriptive Keras model [2, 20, …, 20, 1]
    self.u_model = tf.keras.Sequential()
    self.u_model.add(tf.keras.layers.InputLayer(input_shape=(layers[0],)))
    self.u_model.add(tf.keras.layers.Lambda(
      lambda X: 2.0*(X - lb)/(ub - lb) - 1.0))
    for width in layers[1:]:
        self.u_model.add(tf.keras.layers.Dense(
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

    # Defining the two additional trainable variables for identification
    self.lambda_1 = tf.Variable([0.0], dtype=self.dtype)
    self.lambda_2 = tf.Variable([-6.0], dtype=self.dtype)
    
    self.optimizer = optimizer
    self.logger = logger

  # The actual PINN
  def __f_model(self, X_u):
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
      u = self.u_model(X_f)
      # Deriving INSIDE the tape (since we’ll need the x derivative of this later, u_xx)
      u_x = tape.gradient(u, x_f)
    
    # Getting the other derivatives
    u_xx = tape.gradient(u_x, x_f)
    u_t = tape.gradient(u, t_f)

    # Letting the tape go
    del tape

    l1 = tf.reduce_mean(-u_t/(u * u_x) * (1+1/u_xx) / (1-1/u_xx))
    l2 = tf.reduce_mean((u_t + l1 * u * u_x) / u_xx)
    self.lambda_1.assign([l1])
    self.lambda_2.assign([l2])
    
    # Buidling the PINNs
    return u_t + l1*u*u_x - l2*u_xx

  # Defining custom loss
  def __loss(self, X_u, u, u_pred):
    self.__f_model(X_u)
    return tf.reduce_mean(tf.square(u - u_pred))
    f_pred = self.__f_model(X_u)
    return tf.reduce_mean(tf.square(u - u_pred)) + \
      tf.reduce_mean(tf.square(f_pred))

  def __grad(self, X, u):
    with tf.GradientTape() as tape:
      loss_value = self.__loss(X, u, self.u_model(X))
    return loss_value, tape.gradient(loss_value, self.__wrap_training_variables())

  def __wrap_training_variables(self):
    var = self.u_model.trainable_variables
    # var.extend([self.lambda_1, self.lambda_2])
    return var

  def get_weights(self):
      w = []
      for layer in self.u_model.layers[1:]:
        weights_biases = layer.get_weights()
        weights = weights_biases[0].flatten()
        biases = weights_biases[1]
        w.extend(weights)
        w.extend(biases)
      # w.extend(self.lambda_1.numpy())
      # w.extend(self.lambda_2.numpy())
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
    # self.lambda_1.assign([w[-2]])
    # self.lambda_2.assign([w[-1]])

  def get_params(self, numpy=False):
    l1 = self.lambda_1
    l2 = tf.exp(self.lambda_2)
    if numpy:
      return l1.numpy()[0], l2.numpy()[0]
    return l1, l2

  def summary(self):
    return self.u_model.summary()

  # The training function
  def fit(self, X_u, u, tf_epochs, nt_config):
    self.logger.log_train_start(self)

    # Creating the tensors
    X_u = tf.convert_to_tensor(X_u, dtype=self.dtype)
    u = tf.convert_to_tensor(u, dtype=self.dtype)

    def log_train_epoch(epoch, loss, is_iter):
      l1, l2 = self.get_params(numpy=True)
      custom = f"l1 = {l1:5f}  l2 = {l2:8f}"
      self.logger.log_train_epoch(epoch, loss, custom, is_iter)

    self.logger.log_train_opt("Adam")
    for epoch in range(tf_epochs):
      # Optimization step
      loss_value, grads = self.__grad(X_u, u)
      self.optimizer.apply_gradients(
        zip(grads, self.__wrap_training_variables()))
      log_train_epoch(epoch, loss_value, False)

    self.logger.log_train_opt("LBFGS")
    def loss_and_flat_grad(w):
      with tf.GradientTape() as tape:
        self.set_weights(w)
        tape.watch(self.lambda_1)
        tape.watch(self.lambda_2)
        loss_value = self.__loss(X_u, u, self.u_model(X_u))
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

  def predict(self, X_star):
    u_star = self.u_model(X_star)
    f_star = self.__f_model(X_star)
    return u_star.numpy(), f_star.numpy()

#%% TRAINING THE MODEL

# Getting the data
path = os.path.join(appDataPath, "burgers_shock.mat")
x, t, X, T, Exact_u, X_star, u_star, \
  X_u_train, u_train, ub, lb = prep_data(path, N_u, noise=0.0)
lambdas_star = (1.0, 0.01/np.pi)

#%%

# Creating the model and training
logger = Logger(frequency=10)
pinn = PhysicsInformedNN(layers, tf_optimizer, logger, ub, lb)
def error():
  l1, l2 = pinn.get_params(numpy=True)
  l1_star, l2_star = lambdas_star
  error_lambda_1 = np.abs(l1 - l1_star) / l1_star
  error_lambda_2 = np.abs(l2 - l2_star) / l2_star
  return (error_lambda_1 + error_lambda_2) / 2
logger.set_error_fn(error)
pinn.fit(X_u_train, u_train, tf_epochs, nt_config)

# Getting the model predictions, from the same (x,t) that the predictions were previously gotten from
u_pred, f_pred = pinn.predict(X_star)
lambda_1_pred, lambda_2_pred = pinn.get_params(numpy=True)

# Noise case
x, t, X, T, Exact_u, X_star, u_star, \
  X_u_train, u_train, ub, lb = prep_data(path, N_u, noise=0.01)
pinn = PhysicsInformedNN(layers, tf_optimizer, logger, ub, lb)
pinn.fit(X_u_train, u_train, tf_epochs, nt_config)
lambda_1_pred_noise, lambda_2_pred_noise = pinn.get_params(numpy=True)

print("l1: ", lambda_1_pred)
print("l2: ", lambda_2_pred)
print("l1_noise: ", lambda_1_pred_noise)
print("l2_noise: ", lambda_2_pred_noise)


#%% PLOTTING
plot_ide_cont_results(X_star, u_pred, X_u_train, u_train,
  Exact_u, X, T, x, t, lambda_1_pred, lambda_1_pred_noise, lambda_2_pred, lambda_2_pred_noise)

#%%
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(X_u_train[:, 0], X_u_train[:, 1], u_train)
plt.show()

#%%
model = tf.keras.Sequential()
model.add(tf.keras.layers.InputLayer(input_shape=(layers[0],)))
model.add(tf.keras.layers.Lambda(
  lambda X: 2.0*(X - lb)/(ub - lb) - 1.0))
for width in layers[1:]:
    model.add(tf.keras.layers.Dense(
      width, activation=tf.nn.tanh,
      kernel_initializer='glorot_normal'))

class PrintDot(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

model.compile(optimizer="adam", loss="mse")

#%%
model.fit(X_u_train, u_train, epochs=1000, verbose=0, callbacks=[PrintDot()])


#%%
u_pred = model.predict(X_star)
from scipy.interpolate import griddata
U_pred = griddata(X_star, u_pred.flatten(), (X, T), method='cubic')

#%%
plot_ide_cont_results(X_star, u_pred, X_u_train, u_train,
  Exact_u, X, T, x, t, 0.0, 0.0, 0.0, 0.0)
#%%
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(x, t, U_pred)
plt.show()

