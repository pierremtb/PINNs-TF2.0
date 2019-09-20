import sys
import json
import os
import tensorflow as tf
import numpy as np

from schrodingerutil import prep_data, plot_inf_cont_results
from logger import Logger
from neuralnetwork import NeuralNetwork

# Manually making sure the numpy random seeds are "the same" on all devices
np.random.seed(1234)
tf.random.set_seed(1234)


# HYPER PARAMETERS

eqnPath = "1dcomplex-schrodinger"
if len(sys.argv) > 1:
    with open(sys.argv[1]) as hpFile:
        hp = json.load(hpFile)
else:
    hp = {}
    # Data size on the initial condition solution
    hp["N_0"] = 50
    # Collocation points on the boundaries
    hp["N_b"] = 50
    # Collocation points on the domain
    hp["N_f"] = 20000
    # DeepNN topology (2-sized input [x t], 4 hidden layer of 100-width, 2-sized output [u, v])
    hp["layers"] = [2, 100, 100, 100, 100, 2]
    # Setting up the TF SGD-based optimizer (set tf_epochs=0 to cancel it)
    hp["tf_epochs"] = 200
    hp["tf_lr"] = 0.05
    hp["tf_b1"] = 0.99
    hp["tf_eps"] = 1e-1
    # Setting up the quasi-newton LBGFS optimizer (set nt_epochs=0 to cancel it)
    hp["nt_epochs"] = 0
    hp["nt_lr"] = 1.2
    hp["nt_ncorr"] = 50
    hp["log_frequency"] = 10

# %% DEFINING THE MODEL


class SchrodingerInformedNN(NeuralNetwork):
    def __init__(self, hp, logger, X_f, tb, ub, lb):
        super().__init__(hp, logger, ub, lb)

        X_lb = np.concatenate((0*tb + lb[0], tb), 1)  # (lb[0], tb)
        X_ub = np.concatenate((0*tb + ub[0], tb), 1)  # (ub[0], tb)
        self.X_lb = self.tensor(X_lb)
        self.X_ub = self.tensor(X_ub)

        # Separating the collocation coordinates
        self.x_f = self.tensor(X_f[:, 0:1])
        self.t_f = self.tensor(X_f[:, 1:2])

    # Decomposes the multi-output into the complex values and spatial derivatives
    def uvx_model(self, X):
        x = X[:, 0:1]
        t = X[:, 1:2]
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            tape.watch(t)
            Xtemp = tf.concat([x, t], axis=1)

            h = self.model(Xtemp)
            u = h[:, 0:1]
            v = h[:, 1:2]

        u_x = tape.gradient(u, x)
        v_x = tape.gradient(v, x)
        del tape

        return u, v, u_x, v_x

    # The actual PINN
    def f_model(self):
        # Using the new GradientTape paradigm of TF2.0,
        # which keeps track of operations to get the gradient at runtime
        with tf.GradientTape(persistent=True) as tape:
            # Watching the two inputs we’ll need later, x and t
            tape.watch(self.x_f)
            tape.watch(self.t_f)
            # Packing together the inputs
            X_f = tf.concat([self.x_f, self.t_f], axis=1)

            # Getting the prediction
            u, v, u_x, v_x = self.uvx_model(X_f)

        # Getting the other derivatives
        u_xx = tape.gradient(u_x, self.x_f)
        v_xx = tape.gradient(v_x, self.x_f)
        u_t = tape.gradient(u, self.t_f)
        v_t = tape.gradient(v, self.t_f)

        # Letting the tape go
        del tape

        h2 = (u**2 + v**2)
        f_u = u_t + 0.5*v_xx + h2*v
        f_v = v_t - 0.5*u_xx - h2*u

        return f_u, f_v

    def loss(self, uv, uv_pred):
        u0 = uv[:, 0:1]
        v0 = uv[:, 1:2]
        u0_pred = uv_pred[:, 0:1]
        v0_pred = uv_pred[:, 1:2]
        u_lb_pred, v_lb_pred, u_x_lb_pred, v_x_lb_pred = \
                self.uvx_model(self.X_lb)
        u_ub_pred, v_ub_pred, u_x_ub_pred, v_x_ub_pred = \
                self.uvx_model(self.X_ub)
        f_u_pred, f_v_pred = self.f_model()

        mse_0 = tf.reduce_mean(tf.square(u0 - u0_pred)) + \
            tf.reduce_mean(tf.square(v0 - v0_pred))
        mse_b = tf.reduce_mean(tf.square(u_lb_pred - u_ub_pred)) + \
            tf.reduce_mean(tf.square(v_lb_pred - v_ub_pred)) + \
            tf.reduce_mean(tf.square(u_x_lb_pred - u_x_ub_pred)) + \
            tf.reduce_mean(tf.square(v_x_lb_pred - v_x_ub_pred))

        mse_f = tf.reduce_mean(tf.square(f_u_pred)) + \
            tf.reduce_mean(tf.square(f_v_pred))

        tf.print(f"mse_0 {mse_0}    mse_b {mse_b}    mse_f    {mse_f}")
        return mse_0 + mse_b + mse_f

    def predict(self, X_star):
        h_pred = self.model(X_star)
        u_pred = h_pred[:, 0:1]
        v_pred = h_pred[:, 1:2]
        return u_pred.numpy(), v_pred.numpy()

# %% TRAINING THE MODEL


# Getting the data
path = os.path.join(eqnPath, "data", "NLS.mat")
x, t, X, T, Exact_u, Exact_v, Exact_h, \
    X_star, u_star, v_star, h_star, X_f, \
    ub, lb, tb, x0, u0, v0, X0, H0 = prep_data(
        path, hp["N_0"], hp["N_b"], hp["N_f"], noise=0.0)

# Creating the model
logger = Logger(hp)

pinn = SchrodingerInformedNN(hp, logger, X_f, tb, ub, lb)

# Defining the error function for the logger


def error():
    u_pred, v_pred = pinn.predict(X_star)
    h_pred = np.sqrt(u_pred**2 + v_pred**2)
    return np.linalg.norm(h_star - h_pred, 2) / np.linalg.norm(h_star, 2)


logger.set_error_fn(error)

# Training the PINN
pinn.fit(x0, tf.concat([u0, v0], axis=1))

# Getting the model predictions, from the same (x,t) that the predictions were previously gotten from
u_pred, v_pred = pinn.predict(X_star)
h_pred = np.sqrt(u_pred**2 + v_pred**2)

# %% PLOTTING
plot_inf_cont_results(X_star, u_pred, v_pred, h_pred, Exact_h, X, T, x, t, ub, lb, x0, tb,
                      save_path=eqnPath, save_hp=hp)
