# Databricks notebook source
# Databricks notebook source

# COMMAND ----------

 np

# COMMAND ----------

# MAGIC %run ./ipy_nbs/schrodingerutil

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

print('this is a test notebook for AK to try the data read')

# COMMAND ----------

import sklearn
import numpy as np
import scipy 

import os, sys
from pyDOE import lhs

import matplotlib.pyplot as plt

# COMMAND ----------

# MAGIC %run '/Repos/adkiran@redventures.net/recsys_using_nn/1dcomplex-schrodinger/schrodingerutil'

# COMMAND ----------

from schrodingerutil import prep_data

# COMMAND ----------

sp.__version__

# COMMAND ----------

#CONSTANTS

eqnPath = '.' #"1dcomplex-schrodinger"
path = os.path.join(eqnPath, "data", "NLS.mat")

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



path, N_0, N_b, N_f, noise = path, hp["N_0"], hp["N_b"], hp["N_f"], 0.0

# COMMAND ----------

print(path)
print(f'num initial pts = {N_0}')
print(f'num boundary pts = {N_b}')
print(f'num interior pts = {N_f}')

# COMMAND ----------

data = scipy.io.loadmat(path)

t = data['tt'].flatten()[:,None]
x = data['x'].flatten()[:,None]
Exact = data['uu']
Exact_u = np.real(Exact)
Exact_v = np.imag(Exact)
Exact_h = np.sqrt(Exact_u**2 + Exact_v**2)

# COMMAND ----------

 

# COMMAND ----------

X, T = np.meshgrid(x,t)

X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
u_star = Exact_u.T.flatten()[:,None]
v_star = Exact_v.T.flatten()[:,None]
h_star = Exact_h.T.flatten()[:,None]

# lb = X_star.min(axis=0)
# ub = X_star.max(axis=0) 
lb = np.array([-5.0, 0.0])
ub = np.array([5.0, np.pi/2])

###########################

idx_x = np.random.choice(x.shape[0], N_0, replace=False)
x0 = x[idx_x,:]
u0 = Exact_u[idx_x,0:1]
v0 = Exact_v[idx_x,0:1]

idx_t = np.random.choice(t.shape[0], N_b, replace=False)
tb = t[idx_t,:]

# X0 = np.hstack((x0, tb))
X0 = np.concatenate((x0, 0*x0), 1) # (x0, 0)
H0 = np.hstack((u0, v0))

X_f = lb + (ub-lb)*lhs(2, N_f)

# return x, t, X, T, Exact_u, Exact_v, Exact_h, \
#     X_star, u_star, v_star, h_star, X_f, ub, lb, tb, x0, u0, v0, X0, H0

# COMMAND ----------

plt.figure(figsize=(15,3))
t_init = t.min()
plt.scatter(x0, t_init*np.array(x0))
plt.title('initial points')
plt.grid()

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------


