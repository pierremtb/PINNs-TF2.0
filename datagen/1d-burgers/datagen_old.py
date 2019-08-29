#%%
import numpy as np
import sys
sys.path.insert(0, "datagen/1d-burgers")
from burgers_viscous_time_exact1 import burgers_viscous_time_exact1

nu = 0.01/np.pi
vxn = 256
vtn = 100
x = np.linspace(-1.0, 1, vxn)
t = np.linspace(0, 1, vtn)
u = burgers_viscous_time_exact1(nu, vxn, x, vtn, t)

np.save("1d-burgers/data/burgers_x", x)
np.save("1d-burgers/data/burgers_t", t)
np.save("1d-burgers/data/burgers_u", u)

import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import scipy.io
import matplotlib.gridspec as gridspec

X, T = np.meshgrid(x,t)

# Preparing the inputs x and t (meshed as X, T) for predictions in one single array, as X_star
X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
U_pred = griddata(X_star, u.T.flatten(), (X, T), method='cubic')
    
####### Row 0: u(t,x) ##################    
gs0 = gridspec.GridSpec(1, 2)
gs0.update(top=1-0.06, bottom=1-1.0/3.0+0.06, left=0.15, right=0.85, wspace=0)
ax = plt.subplot(gs0[:, :])

h = ax.imshow(U_pred.T, interpolation='nearest', cmap='rainbow', 
            extent=[t.min(), t.max(), x.min(), x.max()], 
                  origin='lower', aspect='auto')
plt.show()

data = scipy.io.loadmat("PINNs/appendix/Data/burgers_shock.mat")

# Flatten makes [[]] into [], [:,None] makes it a column vector
t = data['t'].flatten()[:,None] # T x 1
x = data['x'].flatten()[:,None] # N x 1

# Keeping the 2D data for the solution data (real() is maybe to make it float by default, in case of zeroes)
Exact_u = np.real(data['usol']).T # T x N

X, T = np.meshgrid(x,t)

# Preparing the inputs x and t (meshed as X, T) for predictions in one single array, as X_star
X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
U_pred = griddata(X_star, Exact_u.flatten(), (X, T), method='cubic')
    
####### Row 0: u(t,x) ##################    
gs0 = gridspec.GridSpec(1, 2)
gs0.update(top=1-0.06, bottom=1-1.0/3.0+0.06, left=0.15, right=0.85, wspace=0)
ax = plt.subplot(gs0[:, :])

h = ax.imshow(U_pred.T, interpolation='nearest', cmap='rainbow', 
            extent=[t.min(), t.max(), x.min(), x.max()], 
                  origin='lower', aspect='auto')
plt.show()

data = scipy.io.loadmat("1d-burgers/data/burgers.mat")

# Flatten makes [[]] into [], [:,None] makes it a column vector
t = data['t'].flatten()[:,None] # T x 1
x = data['x'].flatten()[:,None] # N x 1

# Keeping the 2D data for the solution data (real() is maybe to make it float by default, in case of zeroes)
Exact_u = np.real(data['usol']).T # T x N

X, T = np.meshgrid(x,t)

# Preparing the inputs x and t (meshed as X, T) for predictions in one single array, as X_star
X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
U_pred = griddata(X_star, Exact_u.flatten(), (X, T), method='cubic')
    
####### Row 0: u(t,x) ##################    
gs0 = gridspec.GridSpec(1, 2)
gs0.update(top=1-0.06, bottom=1-1.0/3.0+0.06, left=0.15, right=0.85, wspace=0)
ax = plt.subplot(gs0[:, :])

h = ax.imshow(U_pred.T, interpolation='nearest', cmap='rainbow', 
            extent=[t.min(), t.max(), x.min(), x.max()], 
                  origin='lower', aspect='auto')
plt.show()

data = scipy.io.loadmat("1d-burgers/data/burgers_mathematica.mat")

# Flatten makes [[]] into [], [:,None] makes it a column vector
t = data['t'].flatten()[:,None] # T x 1
x = data['x'].flatten()[:,None] # N x 1

# Keeping the 2D data for the solution data (real() is maybe to make it float by default, in case of zeroes)
Exact_u = np.real(data['usol']).T # T x N

X, T = np.meshgrid(x,t)

# Preparing the inputs x and t (meshed as X, T) for predictions in one single array, as X_star
X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
U_pred = griddata(X_star, Exact_u.flatten(), (X, T), method='cubic')
    
####### Row 0: u(t,x) ##################    
gs0 = gridspec.GridSpec(1, 2)
gs0.update(top=1-0.06, bottom=1-1.0/3.0+0.06, left=0.15, right=0.85, wspace=0)
ax = plt.subplot(gs0[:, :])

h = ax.imshow(U_pred.T, interpolation='nearest', cmap='rainbow', 
            extent=[t.min(), t.max(), x.min(), x.max()], 
                  origin='lower', aspect='auto')
plt.show()

#%%
