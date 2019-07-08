#%% IMPORTING/SETTING UP PATHS

import sys
import os
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import tensorflow as tf
import numpy as np
import scipy.io
from scipy.interpolate import griddata
from pyDOE import lhs
import time
from datetime import datetime
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

repoPath = 'PINNs'
utilsPath = os.path.join(repoPath, 'Utilities')
dataPath = os.path.join(repoPath, 'main', 'Data')
appDataPath = os.path.join(repoPath, 'appendix', 'Data')

sys.path.insert(0, utilsPath)
from plotting import newfig, savefig

#%% DEFINING THE MODEL

class PhysicsInformedNN(object):
  def __init__(self, layers):
    # New descriptive Keras model [2, 20, …, 20, 1]
    self.u_model = tf.keras.Sequential()
    self.u_model.add(tf.keras.layers.InputLayer(input_shape=(layers[0],)))
    for width in layers[1:]:
        self.u_model.add(tf.keras.layers.Dense(width, activation=tf.nn.tanh))
    print(self.u_model.summary())
    
    # Creating the optimizer
    self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

  # The actual PINN
  def f_model(self):
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
    return u_t + u*u_x - (0.01/np.pi)*u_xx

  # Defining custom loss
  def loss(self, u, u_pred):
    f_pred = self.f_model()
    return tf.reduce_mean(tf.square(u - u_pred)) + tf.reduce_mean(tf.square(f_pred))

  # Computing the gradients
  def grad(self, X, u):
    with tf.GradientTape() as tape:
      loss_value = self.loss(u, self.u_model(X))
    return loss_value, tape.gradient(loss_value, self.u_model.trainable_variables)

  def minimize(self, initial_val, loss_grad_func, equality_funcs,
                equality_grad_funcs, inequality_funcs, inequality_grad_funcs,
                packed_bounds, step_callback, optimizer_kwargs):

    def loss_grad_func_wrapper(x):
      # SciPy's L-BFGS-B Fortran implementation requires gradients as doubles.
      loss, gradient = loss_grad_func(x)
      return loss, gradient.astype('float64')

    optimizer_kwargs = dict(optimizer_kwargs.items())
    method = optimizer_kwargs.pop('method', self._DEFAULT_METHOD)

    constraints = []
    for func, grad_func in zip(equality_funcs, equality_grad_funcs):
      constraints.append({'type': 'eq', 'fun': func, 'jac': grad_func})
    for func, grad_func in zip(inequality_funcs, inequality_grad_funcs):
      constraints.append({'type': 'ineq', 'fun': func, 'jac': grad_func})

    minimize_args = [loss_grad_func_wrapper, initial_val]
    minimize_kwargs = {
        'jac': True,
        'callback': step_callback,
        'method': method,
        'constraints': constraints,
        'bounds': packed_bounds,
    }

    for kwarg in minimize_kwargs:
      if kwarg in optimizer_kwargs:
        if kwarg == 'bounds':
          # Special handling for 'bounds' kwarg since ability to specify bounds
          # was added after this module was already publicly released.
          raise ValueError(
              'Bounds must be set using the var_to_bounds argument')
        raise ValueError(
            'Optimizer keyword arg \'{}\' is set '
            'automatically and cannot be injected manually'.format(kwarg))

    minimize_kwargs.update(optimizer_kwargs)

    import scipy.optimize  # pylint: disable=g-import-not-at-top
    result = scipy.optimize.minimize(*minimize_args, **minimize_kwargs)

    message_lines = [
        'Optimization terminated with:',
        '  Message: %s',
        '  Objective function value: %f',
    ]
    message_args = [result.message, result.fun]
    if hasattr(result, 'nit'):
      # Some optimization methods might not provide information such as nit and
      # nfev in the return. Logs only available information.
      message_lines.append('  Number of iterations: %d')
      message_args.append(result.nit)
    if hasattr(result, 'nfev'):
      message_lines.append('  Number of functions evaluations: %d')
      message_args.append(result.nfev)
    logging.info('\n'.join(message_lines), *message_args)

    return result['x']

  def fit(self, X_u, u, X_f, epochs=1):
    # Creating the tensors
    self.X_u = tf.convert_to_tensor(X_u, dtype="float32")
    self.u = tf.convert_to_tensor(u, dtype="float32")
    # Separating the collocation coordinates
    self.x_f = tf.convert_to_tensor(X_f[:, 0:1], dtype="float32")
    self.t_f = tf.convert_to_tensor(X_f[:, 1:2], dtype="float32")

    # Keep results for plotting
    self.train_loss_results = []

    # Training loop
    for epoch in range(epochs):
      # Optimization step
      loss_value, grads = self.grad(self.X_u, self.u)
      self.optimizer.apply_gradients(zip(grads, self.u_model.trainable_variables))

      # Keeping track of loss
      self.train_loss_results.append(loss_value)

      # Logging every so often
      if epoch % 10 == 0:
        print("Epoch {:03d}: Loss: {:.3f}".format(epoch, loss_value))

  def predict(self, X_star):
    u_star = self.u_model(X_star)
    f_star = self.f_model()
    return u_star, f_star

#%% RUNNING THE MODEL

print("TensorFlow version: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))
print("GPU-accerelated: {}".format(tf.test.is_gpu_available()))

# Initial data size on the solution u
N_u = 100
# Collocation data size on f(t,x)
N_f = 10000
# DeepNN topology (2 input, 8 hidden layer of 20-width, 1 output ([u])
layers = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]

# Reading external data [t is 100x1, usol is 256x100 (solution), x is 256x1]
data = scipy.io.loadmat(os.path.join(appDataPath, 'burgers_shock.mat'))

# Flatten makes [[]] into [], [:,None] makes it a column vector
t = data['t'].flatten()[:,None]
x = data['x'].flatten()[:,None]

# Keeping the 2D data for the solution data (real() is maybe to make it float by default, in case of zeroes)
Exact_u = np.real(data['usol']).T

# Meshing x and t in 2D (256,100)
X, T = np.meshgrid(x,t)

# Preparing the inputs x and t (meshed as X, T) for predictions in one single array, as X_star
X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))

# Preparing the testing u_star
u_star = Exact_u.flatten()[:,None]

# Domain bounds (lowerbounds upperbounds) [x, t], which are here ([-1.0, 0.0] and [1.0, 1.0])
lb = X_star.min(axis=0)
ub = X_star.max(axis=0) 
               
# Getting the initial conditions (t=0)
xx1 = np.hstack((X[0:1,:].T, T[0:1,:].T))
uu1 = Exact_u[0:1,:].T
# Getting the lowest boundary conditions (x=-1) 
xx2 = np.hstack((X[:,0:1], T[:,0:1]))
uu2 = Exact_u[:,0:1]
# Getting the highest boundary conditions (x=1) 
xx3 = np.hstack((X[:,-1:], T[:,-1:]))
uu3 = Exact_u[:,-1:]
# Stacking them in multidimensional tensors for training (X_u_train is for now the continuous boundaries)
X_u_train = np.vstack([xx1, xx2, xx3])
u_train = np.vstack([uu1, uu2, uu3])

# Generating the x and t collocation points for f, with each having a N_f size
# We pointwise add and multiply to spread the LHS over the 2D domain
X_f_train = lb + (ub-lb)*lhs(2, N_f)
# Pretty sure this next line isn't useful
#X_f_train = np.vstack((X_f_train, X_u_train))

# Generating a uniform random sample from ints between 0, and the size of x_u_train, of size N_u (initial data size) and without replacement (unique)
idx = np.random.choice(X_u_train.shape[0], N_u, replace=False)
# Getting the corresponding X_u_train (which is now scarce boundary/initial coordinates)
X_u_train = X_u_train[idx,:]
# Getting the corresponding u_train
u_train = u_train [idx,:]

# Creating the model and training
pinn = PhysicsInformedNN(layers)
pinn.fit(X_u_train, u_train, X_f_train, epochs=50000)

# Getting the model predictions, from the same (x,t) that the predictions were previously gotten from
u_pred, f_pred = pinn.predict(X_star)

# Getting the relative error for u
error_u = np.linalg.norm(u_star-u_pred,2)/np.linalg.norm(u_star,2)
print('Error u: %e' % (error_u))

# Interpolating the results on the whole (x,t) domain.
# griddata(points, values, points at which to interpolate, method)
U_pred = griddata(X_star, u_pred.numpy().flatten(), (X, T), method='cubic')

#%% PLOTTING THE RESULTS
fig, ax = newfig(1.0, 1.1)
ax.axis('off')

####### Row 0: u(t,x) ##################    
gs0 = gridspec.GridSpec(1, 2)
gs0.update(top=1-0.06, bottom=1-1/3, left=0.15, right=0.85, wspace=0)
ax = plt.subplot(gs0[:, :])

h = ax.imshow(U_pred.T, interpolation='nearest', cmap='rainbow', 
              extent=[t.min(), t.max(), x.min(), x.max()], 
              origin='lower', aspect='auto')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(h, cax=cax)

ax.plot(X_u_train[:,1], X_u_train[:,0], 'kx', label = 'Data (%d points)' % (u_train.shape[0]), markersize = 4, clip_on = False)

line = np.linspace(x.min(), x.max(), 2)[:,None]
ax.plot(t[25]*np.ones((2,1)), line, 'w-', linewidth = 1)
ax.plot(t[50]*np.ones((2,1)), line, 'w-', linewidth = 1)
ax.plot(t[75]*np.ones((2,1)), line, 'w-', linewidth = 1)    

ax.set_xlabel('$t$')
ax.set_ylabel('$x$')
ax.legend(frameon=False, loc = 'best')
ax.set_title('$u(t,x)$', fontsize = 10)

####### Row 1: u(t,x) slices ##################    
gs1 = gridspec.GridSpec(1, 3)
gs1.update(top=1-1/3, bottom=0, left=0.1, right=0.9, wspace=0.5)

ax = plt.subplot(gs1[0, 0])
ax.plot(x,Exact_u[25,:], 'b-', linewidth = 2, label = 'Exact')       
ax.plot(x,U_pred[25,:], 'r--', linewidth = 2, label = 'Prediction')
ax.set_xlabel('$x$')
ax.set_ylabel('$u(t,x)$')    
ax.set_title('$t = 0.25$', fontsize = 10)
ax.axis('square')
ax.set_xlim([-1.1,1.1])
ax.set_ylim([-1.1,1.1])

ax = plt.subplot(gs1[0, 1])
ax.plot(x,Exact_u[50,:], 'b-', linewidth = 2, label = 'Exact')       
ax.plot(x,U_pred[50,:], 'r--', linewidth = 2, label = 'Prediction')
ax.set_xlabel('$x$')
ax.set_ylabel('$u(t,x)$')
ax.axis('square')
ax.set_xlim([-1.1,1.1])
ax.set_ylim([-1.1,1.1])
ax.set_title('$t = 0.50$', fontsize = 10)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.35), ncol=5, frameon=False)

ax = plt.subplot(gs1[0, 2])
ax.plot(x,Exact_u[75,:], 'b-', linewidth = 2, label = 'Exact')       
ax.plot(x,U_pred[75,:], 'r--', linewidth = 2, label = 'Prediction')
ax.set_xlabel('$x$')
ax.set_ylabel('$u(t,x)$')
ax.axis('square')
ax.set_xlim([-1.1,1.1])
ax.set_ylim([-1.1,1.1])    
ax.set_title('$t = 0.75$', fontsize = 10)

plt.show()
# savefig('./inf_cont_burgers')