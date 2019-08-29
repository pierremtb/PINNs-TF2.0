#%%
import sympy as sp
import numpy as np

# Setting up the symbolic variables
x, nu, t = sp.symbols('x nu t')
phi = sp.exp(-(x-4*t)**2/(4*nu*(t+1))) + sp.exp(-(x-4*t-2*np.pi)**2/(4*nu*(t+1)))

# Evaluate the partial derivative dphi/dx using sympy
phiprime = phi.diff(x)
#print phiprime

# Create the initial conditions function
u = -2*nu*(phiprime/phi)+4
#print u

# Transform the symbolic equation into a function using lambdify
from sympy.utilities.lambdify import lambdify
ufunc = lambdify ((t, x, nu), u)

nu = 0.01/np.pi
vxn = 256
vtn = 100
x = np.linspace(-1.0, 1, vxn)
t = np.linspace(0, 1, vtn)
u = np.zeros((vxn, vtn))
for i, x_i in enumerate(x):
    for j, t_j in enumerate(t):
        u[i, j] = ufunc(t_j, x_i, nu)

print(u)

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