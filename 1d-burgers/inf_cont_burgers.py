import sys
import json
import os
import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp

# Manually making sure the numpy random seeds are "the same" on all devices
np.random.seed(1234)
tf.random.set_seed(1234)

# LOCAL IMPORTS

eqnPath = "1d-burgers"
sys.path.append(eqnPath)
sys.path.append("utils")
from logger import Logger
from neuralnetwork import NeuralNetwork
from burgersutil import prep_data, plot_inf_cont_results

# HYPER PARAMETERS

if len(sys.argv) > 1:
    with open(sys.argv[1]) as hpFile:
        hp = json.load(hpFile)
else:
    hp = {}
    # Data size on the solution u
    hp["N_u"] = 100
    # Collocation points size, where we’ll check for f = 0
    hp["N_f"] = 10000
    # DeepNN topology (2-sized input [x t], 8 hidden layer of 20-width, 1-sized output [u]
    hp["layers"] = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]
    # Setting up the TF SGD-based optimizer (set tf_epochs=0 to cancel it)
    hp["tf_epochs"] = 100
    hp["tf_lr"] = 0.03
    hp["tf_b1"] = 0.9
    hp["tf_eps"] = None
    # Setting up the quasi-newton LBGFS optimizer (set nt_epochs=0 to cancel it)
    hp["nt_epochs"] = 200
    hp["nt_lr"] = 0.8
    hp["nt_ncorr"] = 50
    hp["log_frequency"] = 10

# %% DEFINING THE MODEL


class BurgersInformedNN(NeuralNetwork):
    def __init__(self, hp, logger, X_f, ub, lb, nu):
        super().__init__(hp, logger, ub, lb)

        self.nu = nu

        # Separating the collocation coordinates
        self.x_f = self.tensor(X_f[:, 0:1])
        self.t_f = self.tensor(X_f[:, 1:2])

    # Defining custom loss
    def loss(self, u, u_pred):
        f_pred = self.f_model()
        return tf.reduce_mean(tf.square(u - u_pred)) + \
            tf.reduce_mean(tf.square(f_pred))

    # The actual PINN
    def f_model(self):
        # Using the new GradientTape paradigm of TF2.0,
        # which keeps track of operations to get the gradient at runtime
        with tf.GradientTape(persistent=True) as tape:
            # Watching the two inputs we’ll need later, x and t
            tape.watch(self.x_f)
            tape.watch(self.t_f)
            # Packing together the inputs
            X_f = tf.stack([self.x_f[:, 0], self.t_f[:, 0]], axis=1)

            # Getting the prediction
            u = self.model(X_f)
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

    def predict(self, X_star):
        u_star = self.model(X_star)
        f_star = self.f_model()
        return u_star.numpy(), f_star.numpy()

# %% TRAINING THE MODEL


# Getting the data
path = os.path.join(eqnPath, "data", "burgers_shock.mat")
x, t, X, T, Exact_u, X_star, u_star, \
    X_u_train, u_train, X_f, ub, lb = prep_data(
        path, hp["N_u"], hp["N_f"], noise=0.0)

# Creating the model
logger = Logger(hp)
pinn = BurgersInformedNN(hp, logger, X_f, ub, lb, nu=0.01/np.pi)

# Defining the error function for the logger and training
def error():
    u_pred, _ = pinn.predict(X_star)
    return np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)


logger.set_error_fn(error)
pinn.fit(X_u_train, u_train)

# Getting the model predictions
u_pred, _ = pinn.predict(X_star)

# %% PLOTTING
plot_inf_cont_results(X_star, u_pred.flatten(), X_u_train, u_train,
                      Exact_u, X, T, x, t, save_path=eqnPath, save_hp=hp)
