#%% IMPORTING/SETTING UP PATHS

import sys
import os
import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
import time

# Manually making sure the numpy random seeds are "the same" on all devices
np.random.seed(1234)
tf.random.set_seed(1234)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata

sys.path.append("1d-burgers")
from custom_lbfgs import lbfgs, Struct
from burgersutil import prep_data, Logger, plot_inf_cont_results, appDataPath

#%% PINN Model to benchmark

startPinn = time.time()
# Creating the model and training
from inf_cont_burgers import hp, u_pred, u_star
N_u = ["N_u"]
N_f = ["N_f"]
layers = ["layers"]
tf_epochs = ["tf_epochs"]
nt_epochs = ["nt_epochs"]
N_u_pinn = N_u
durationPinn = time.time() - startPinn
errorPinn = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)
print("Error PINN: ", errorPinn)
print("Time PINN: ", durationPinn)

#%% Different data sizes on Plain NN Model
path = os.path.join(appDataPath, "burgers_shock.mat")
x, t, X, T, Exact_u, X_star, u_star, \
    X_u_train, u_train, ub, lb = prep_data(path, N_u, noise=0.0)
model = tf.keras.Sequential()
model.add(tf.keras.layers.InputLayer(input_shape=(layers[0],)))
model.add(tf.keras.layers.Lambda(
  lambda X: 2.0*(X - lb)/(ub - lb) - 1.0))
for width in layers[1:]:
    model.add(tf.keras.layers.Dense(
      width, activation=tf.nn.tanh,
      kernel_initializer='glorot_normal'))

model.compile(optimizer="adam", loss="mse")
initial_w = model.get_weights()

errors = []
durations = []
N_u_sizes = [50, 200, 400, 1000, 2000]
for N_u in N_u_sizes:
  start = time.time()
  print(f"Training NN (N_u={N_u})...")
  x, t, X, T, Exact_u, X_star, u_star, \
    X_u_train, u_train, ub, lb = prep_data(path, N_u, noise=0.0)
  model.set_weights(initial_w)
  model.fit(X_u_train, u_train, epochs=tf_epochs+nt_epochs, verbose=0)
  u_pred_nn = model.predict(X_star)
  error = np.linalg.norm(u_star - u_pred_nn, 2) / np.linalg.norm(u_star, 2)
  errors.append(error)
  duration = time.time() - start
  durations.append(duration)
  print(f"Error NN (N_u={N_u}): {error}")
  print(f"Time NN (N_u={N_u}): {duration}")

#%%
errors_0 = []
durations_0 = []
N_u_0_sizes = [50, 100, 200]
for N_u_0 in N_u_0_sizes:
  start = time.time()
  print(f"Training NN (N_u_0={N_u_0})...")
  x, t, X, T, Exact_u, X_star, u_star, \
    X_u_train, u_train, X_f, ub, lb = prep_data(path, N_u_0, N_f, noise=0.0)
  model.set_weights(initial_w)
  model.fit(X_u_train, u_train, epochs=tf_epochs+nt_epochs, verbose=0)
  u_pred_nn = model.predict(X_star)
  error_nn = np.linalg.norm(u_star - u_pred_nn, 2) / np.linalg.norm(u_star, 2)
  time_nn = time.time() - start
  errors_0.append(error_nn)
  durations_0.append(time_nn)
  print(f"Error NN (N_u_0={N_u_0}): {error_nn}")
  print(f"Time NN (N_u_0={N_u_0}): {time_nn}")

#%%
fig, ax = plt.subplots()
plt.plot(N_u_0_sizes, errors_0, 'ro--', label="Data on bnd/ini")
plt.plot(N_u_sizes, errors, 'bo--', label="Data on domain")
plt.plot([N_u_pinn], [errorPinn], 'go--', label="PINN, data on bnd/ini")
plt.xlabel("Data size $N_u$")
plt.ylabel("Relative error")
plt.title("Inference on 1D Burger's equation")
ax.legend()

xOff = 30
yOff = 0.02
for i, _ in enumerate(N_u_sizes):
    ax.annotate(f"{durations[i]:.4}", (N_u_sizes[i] + xOff, errors[i] + yOff))
for i, _ in enumerate(N_u_0_sizes):
    ax.annotate(f"{durations[i]:.4}", (N_u_0_sizes[i] + xOff, errors_0[i] + yOff))
ax.annotate(f"{durationPinn:.4}", (N_u_pinn + xOff, errorPinn - yOff))
plt.savefig("./figures/inf_cont_burgers_bench.png", dpi=300)
plt.show()

#%% Data plots
path = os.path.join(appDataPath, "burgers_shock.mat")
x, t, X, T, Exact_u, X_star, u_star, \
  X_u_train, u_train, ub, lb = prep_data(path, 2000, noise=0.0)

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(X_u_train[:, 0], X_u_train[:, 1], u_train)
ax.set_xlabel("x")
ax.set_ylabel("t")
ax.set_zlabel("u(x, t)")
plt.savefig("./figures/burgers_data_domain.png", dpi=300)
plt.show()

path = os.path.join(appDataPath, "burgers_shock.mat")
x, t, X, T, Exact_u, X_star, u_star, \
  X_u_train, u_train, X_f, ub, lb = prep_data(path, 100, 1000, noise=0.0)

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(X_u_train[:, 0], X_u_train[:, 1], u_train)
ax.set_xlabel("x")
ax.set_ylabel("t")
ax.set_zlabel("u(x, t)")
plt.savefig("./figures/burgers_data_inibnd.png", dpi=300)
plt.show()