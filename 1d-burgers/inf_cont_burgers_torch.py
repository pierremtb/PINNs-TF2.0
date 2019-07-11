#%% IMPORTING/SETTING UP PATHS

import sys
import os
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
import scipy.io
from scipy.interpolate import griddata
import time
from datetime import datetime
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import torch
import torch.nn as nn
import torch.optim
import torch.nn.functional as F

# Manually making sure the numpy random seeds are "the same" on all devices, for reproducibility in random processes
np.random.seed(1234)
# Same for tensorflow
#tf.random.set_seed(1234)

repoPath = "../PINNs"
utilsPath = os.path.join(repoPath, "Utilities")
dataPath = os.path.join(repoPath, "main", "Data")
appDataPath = os.path.join(repoPath, "appendix", "Data")

sys.path.insert(0, utilsPath)
from plotting import newfig, savefig
import burgersutil

dtype = torch.float
device = torch.device("cpu")
# device = torch.device("cuda:0") # Uncomment this to run on GPU

#%% HYPER PARAMETERS

# Data size on the solution u
N_u = 50
# Collocation data size on f(t,x)
N_f = 10000
# DeepNN topology (2-sized input [x t], 8 hidden layer of 20-width, 1-sized output [u]
layers = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]
# Creating the optimizer
#optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
epochs = 500

#%% DEFINING THE MODEL

class ConvNN(torch.nn.Module):

    def __init__(self, layers, N_u, N_f):
        super(ConvNN, self).__init__()  # call the inherited class constructor

        self.layers = layers

        self.input = torch.nn.Linear(2, 20)
        self.layer = torch.nn.Linear(20, 20)
        self.output = torch.nn.Linear(20, 1)

        self.act = torch.nn.Tanh()

        self.losses = []
        self.accuracies = []
        self.val_accuracies = []
        self.loss_LPF = 2.3
        self.optimizer = None

        self.N_u = N_u
        self.N_f = N_f

    def nth_derivative(self, f, wrt, n):
      for i in range(n):
          grads = torch.autograd.grad(f, wrt, create_graph=True)[0]
          f = grads.sum()
      return grads

    def get_second_order_grad(self, grads, xs):
      start = time.time()
      grads2 = []
      for j, (grad, x) in enumerate(zip(grads, xs)):
          print('2nd order on ', j, 'th layer')
          print(x.size())
          grad = torch.reshape(grad, [-1])
          grads2_tmp = []
          for count, g in enumerate(grad):
              g2 = torch.autograd.grad(g, x, retain_graph=True)[0]
              g2 = torch.reshape(g2, [-1])
              grads2_tmp.append(g2[count].data.cpu().numpy())
          grads2.append(torch.from_numpy(np.reshape(grads2_tmp, x.size())).to(DEVICE_IDS[0]))
          print('Time used is ', time.time() - start)

    def f_model(self):
      u = self(self.X_f)
      u.backward(self.X_f, retain_graph=True)
      grads = self.X_f.grad
      u_x = grads[:, 0]
      u_t = grads[:, 1]
      x = self.X_f[:, 0]

      # No way to get the second order derivatives via autograd in PyTorch :'(
      u_xx = torch.tensor(np.gradient(u_x.detach().numpy(), x.detach().numpy()), dtype=torch.float32)
      
      nu = (0.01)/np.pi
      return u_t + u*u_x - nu*u_xx

    def loss(self, u_pred, u):
      mse_u = ((u_pred - u) ** 2).sum() / self.N_u
      mse_f = (self.f_model() ** 2).sum() / self.N_f
      return mse_u + mse_f


    def init_optimizer(self):
        # loss function
        self.criterion = torch.nn.MSELoss(reduction='sum')
        #self.criterion = torch.nn.CrossEntropyLoss()

        # optimizer
        self.optimizer = torch.optim.LBFGS(self.parameters(), lr=0.1)

    def forward(self, x):
      h = self.act(self.input(x))
      for i in range(len(layers) - 2):
        h = self.act(self.layer(h))
      y_pred = self.act(self.output(h))
      return y_pred

    def train_batch(self, x, y):
      # Forward pass: Compute predicted y by passing x to the model
      for i in range(200):
        def closure():
          self.optimizer.zero_grad()
          y_pred = self(x)
          loss = self.loss(y_pred, y)
          loss.backward()
          self.losses.append(float(loss.data.item()))
          return loss

        # Record accuracy
        #total = y.size(0)
        #_, predicted = torch.max(y_pred.data, 1)
        #correct = (predicted == y).sum().item()
        #acc = correct / total
        #self.accuracies.append(acc)

        res = self.optimizer.step(closure)
        print("STEP: ", i)
        print(res)
      # print(self.losses)
      # print(self(x))
      # print(y)
        

    def train_all_batches(self, X, u, X_f, device):
      x_batch = torch.tensor(X, dtype=torch.float32, requires_grad=True, device=device)
      y_batch = torch.tensor(u, dtype=torch.float32, requires_grad=True, device=device)
      self.X_f = torch.tensor(X_f, dtype=torch.float32, requires_grad=True, device=device)
      self.train_batch(x_batch, y_batch)
            
    def plot_loss(self):
        plt.title('Loss over time')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.plot(self.losses)
        plt.show()

    def plot_acc(self):
        plt.title('Accuracy over time')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.plot(self.accuracies)
        plt.plot(self.val_accuracies)
        plt.show()

#%% RUNNING THE MODEL and PLOTTING

# Getting the data
path = os.path.join(appDataPath, "burgers_shock.mat")
x, t, X, T, Exact_u, X_star, u_star, \
  X_u_train, u_train, X_f_train = burgersutil.prep(path, N_u, N_f, noise=0.0)

logger = burgersutil.Logger(X_star, u_star)

# Creating the model and training
#pinn = PhysicsInformedNN(layers, optimizer, logger, nu=0.01/np.pi)
#pinn.fit(X_u_train, u_train, X_f_train, epochs)

pinn = ConvNN(layers, N_u, N_f).to(device)
pinn.init_optimizer()
pinn.train_all_batches(X_u_train, u_train, X_f_train, device)

exit(0)

# Getting the model predictions, from the same (x,t) that the predictions were previously gotten from
u_pred, f_pred = pinn.predict(X_star)

# Interpolating the results on the whole (x,t) domain.
# griddata(points, values, points at which to interpolate, method)
U_pred = griddata(X_star, u_pred.numpy().flatten(), (X, T), method='cubic')

# Creating the figures
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