#%% IMPORTING/SETTING UP PATHS

import sys
import json
import os
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
from burgersutil import prep_data, plot_inf_disc_results
from neuralnetwork import NeuralNetwork
from logger import Logger

#%% HYPER PARAMETERS

if len(sys.argv) > 1:
    with open(sys.argv[1]) as hpFile:
        hp = json.load(hpFile)
else:
    hp = {}
    # Data size on the initial condition on u
    hp["N_n"] = 250
    # Number of RK stages
    hp["q"] = 500
    # DeepNN topology (1-sized input [x], 3 hidden layer of 50-width, q+1-sized output [u_1^n(x), ..., u_{q+1}^n(x)]
    hp["layers"] = [1, 50, 50, 50, hp["q"] + 1]
    # Setting up the TF SGD-based optimizer (set tf_epochs=0 to cancel it)
    hp["tf_epochs"] = 200
    hp["tf_lr"] = 0.001
    hp["tf_b1"] = 0.9
    hp["tf_eps"] = 1e-08
    # Setting up the quasi-newton LBGFS optimizer (set nt_epochs=0 to cancel it)
    hp["nt_epochs"] = 1000
    hp["nt_lr"] = 0.8
    hp["nt_ncorr"] = 50
    hp["log_frequency"] = 10

#%% DEFINING THE MODEL

class BurgersInformedNN(NeuralNetwork):
    def __init__(self, hp, logger, dt, x_1, lb, ub, nu, IRK_weights, IRK_times):
        super().__init__(hp, logger, ub, lb)

        self.nu = nu
        self.dt = dt
        self.q = max(hp["q"],1)
        self.IRK_weights = IRK_weights
        self.IRK_times = IRK_times
        self.x_1 = tf.convert_to_tensor(x_1, dtype=self.dtype)

    # The actual PINN
    def U_0_model(self, x):
        # Using the new GradientTape paradigm of TF2.0,
        # which keeps track of operations to get the gradient at runtime
        with tf.GradientTape(persistent=True) as tape:
            # Watching the two inputs we’ll need later, x and t
            tape.watch(x)
            tape.watch(self.dummy_x0_tf)

            # Getting the prediction, and removing the last item (q+1)
            U_1 = self.model(x) # shape=(len(x), q+1)
            U = U_1[:, :-1] # shape=(len(x), q)

            # Deriving INSIDE the tape (2-step-dummy grad technique because U is a mat)
            g_U = tape.gradient(U, x, output_gradients=self.dummy_x0_tf)
            U_x = tape.gradient(g_U, self.dummy_x0_tf)
            g_U_x = tape.gradient(U_x, x, output_gradients=self.dummy_x0_tf)

        # Doing the last one outside the with, to optimize performance
        # Impossible to do for the earlier grad, because they’re needed after
        U_xx = tape.gradient(g_U_x, self.dummy_x0_tf)

        # Letting the tape go
        del tape

        # Buidling the PINNs, shape = (len(x), q+1), IRK shape = (q, q+1)
        nu = self.nu
        N = U*U_x - nu*U_xx # shape=(len(x), q)
        return U_1 + self.dt*tf.matmul(N, self.IRK_weights.T)

    # Defining custom loss
    def loss(self, u_0, u_0_pred):
        u_1_pred = self.model(self.x_1)
        return tf.reduce_sum(tf.square(u_0_pred - u_0)) + \
            tf.reduce_sum(tf.square(u_1_pred))

    # Specifying that the loss depends directly on the PINN
    def grad(self, x_0, u_0):
        with tf.GradientTape() as tape:
            loss_value = self.loss(u_0, self.U_0_model(x_0))
        return loss_value, tape.gradient(loss_value, self.wrap_training_variables())

    # Doing the same for the custom grad
    def get_loss_and_flat_grad(self, X, u):
        def loss_and_flat_grad(w):
            with tf.GradientTape() as tape:
                self.set_weights(w)
            loss_value = self.loss(u, self.U_0_model(X))
            grad = tape.gradient(loss_value, self.wrap_training_variables())
            grad_flat = []
            for g in grad:
                grad_flat.append(tf.reshape(g, [-1]))
            grad_flat =  tf.concat(grad_flat, 0)
            return loss_value, grad_flat

        return loss_and_flat_grad

    # Providing the dummy tensor before fitting
    def fit(self, x_0, u_0):
        # Creating dummy tensors for the gradients
        self.dummy_x0_tf = tf.ones([x_0.shape[0], self.q], dtype=self.dtype)

        super().fit(x_0, u_0)

    # Returning just what we need as prediction
    def predict(self, x_star):
        u_star = self.model(x_star)[:, -1]
        return u_star

#%% TRAINING THE MODEL

# Setup
lb = np.array([-1.0])
ub = np.array([1.0])
idx_t_0 = 10
idx_t_1 = 90
nu = 0.01/np.pi

# Getting the data
path = os.path.join(eqnPath, "data", "burgers_shock.mat")
x, t, dt, \
    Exact_u, x_0, u_0, x_1, x_star, u_star, \
    IRK_weights, IRK_times = prep_data(path, N_n=hp["N_n"], q=hp["q"],
                                       lb=lb, ub=ub, noise=0.0,
                                       idx_t_0=idx_t_0, idx_t_1=idx_t_1)

# Creating the model
logger = Logger(hp)
pinn = BurgersInformedNN(hp, logger, dt, x_1, lb, ub, nu, IRK_weights, IRK_times)

# Defining the error function for the logger and training
def error():
    u_pred = pinn.predict(x_star)
    return np.linalg.norm(u_pred - u_star, 2) / np.linalg.norm(u_star, 2)
logger.set_error_fn(error)
pinn.fit(x_0, u_0)

# Getting the model predictions, from the same (x,t) that the predictions were previously gotten from
u_1_pred = pinn.predict(x_star)

#%% PLOTTING
plot_inf_disc_results(x_star, idx_t_0, idx_t_1, x_0, u_0,
    ub, lb, u_1_pred, Exact_u, x, t,
    save_path=eqnPath, save_hp=hp)
