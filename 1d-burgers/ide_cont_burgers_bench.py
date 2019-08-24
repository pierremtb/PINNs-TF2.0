#%% IMPORTING/SETTING UP PATHS

import sys
import os
import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp

# Manually making sure the numpy random seeds are "the same" on all devices
np.random.seed(1234)
tf.random.set_seed(1234)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata

sys.path.append("1d-burgers")
from custom_lbfgs import lbfgs, Struct
from burgersutil import prep_data, Logger, plot_ide_cont_results, appDataPath

#%% PINN Model to benchmark

# Creating the model and training
from ide_cont_burgers import N_u, layers, tf_epochs, nt_epochs, u_pred, u_star, lambda_1_pred, lambda_2_pred

print("Error PINN: ", np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2))


#%% GETTING THE DATA

path = os.path.join(appDataPath, "burgers_shock.mat")
x, t, X, T, Exact_u, X_star, u_star, \
  X_u_train, u_train, ub, lb = prep_data(path, N_u, noise=0.0)
lambdas_star = (1.0, 0.01/np.pi)

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(X_u_train[:, 0], X_u_train[:, 1], u_train)
plt.show()

#%% Plain NN Model
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
model.fit(X_u_train, u_train, epochs=tf_epochs+nt_epochs, verbose=1)
u_pred_nn = model.predict(X_star)

print("Error NN: ", np.linalg.norm(u_star - u_pred_nn, 2) / np.linalg.norm(u_star, 2))

#%% PLOTTING u(x,t) (PINN, NN, STAR)
ax = plt.subplot()
U_pred = griddata(X_star, u_pred.flatten(), (X, T), method='cubic')
h = ax.imshow(U_pred.T, interpolation='nearest', cmap='rainbow', 
                  extent=[t.min(), t.max(), x.min(), x.max()], 
                  origin='lower', aspect='auto')
plt.show()
ax = plt.subplot()
U_pred_nn = griddata(X_star, u_pred_nn.flatten(), (X, T), method='cubic')
h = ax.imshow(U_pred_nn.T, interpolation='nearest', cmap='rainbow', 
                  extent=[t.min(), t.max(), x.min(), x.max()], 
                  origin='lower', aspect='auto')
plt.show()
ax = plt.subplot()
U_star = griddata(X_star, u_star.flatten(), (X, T), method='cubic')
h = ax.imshow(U_star.T, interpolation='nearest', cmap='rainbow', 
                  extent=[t.min(), t.max(), x.min(), x.max()], 
                  origin='lower', aspect='auto')
plt.show()

#%% REGULAR PLOTTING
plot_ide_cont_results(X_star, u_pred, X_u_train, u_train,
Exact_u, X, T, x, t, 0.0, 0.0, 0.0, 0.0)
plot_ide_cont_results(X_star, u_pred_nn, X_u_train, u_train,
  Exact_u, X, T, x, t, 0.0, 0.0, 0.0, 0.0)

#%% TRYING TO FIND LAMBDA_2
def hessian(x):
    """
    Calculate the hessian matrix with finite differences
    Parameters:
       - x : ndarray
    Returns:
       an array of shape (x.dim, x.ndim) + x.shape
       where the array[i, j, ...] corresponds to the second derivative x_ij
    """
    x_grad = np.gradient(x) 
    hessian = np.empty((x.ndim, x.ndim) + x.shape, dtype=x.dtype) 
    for k, grad_k in enumerate(x_grad):
        # iterate over dimensions
        # apply gradient again to every component of the first derivative.
        tmp_grad = np.gradient(grad_k) 
        for l, grad_kl in enumerate(tmp_grad):
            hessian[k, l, :, :] = grad_kl
    return hessian

U = griddata(X_star, u_pred_nn.flatten(), (X, T), method='cubic').T
dx = x[1, 0] - x[0, 0]
dt = t[1, 0] - t[0, 0]
grads = np.gradient(U)
U_x = grads[0]
U_t = grads[1]
grads_2 = hessian(U)
U_xx = grads_2[0, 0, :, :]

lambdas = np.linspace(-1, 1, 2000)
print(lambdas)

def r(l):
    return np.square(U_t + U * U_x - l * U_xx).mean()

res = [r(l) for l in lambdas]

plt.plot(lambdas, res)

#%%
