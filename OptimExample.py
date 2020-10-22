# This is a sample Python script.
# author: Fabrizio Miorelli
# Licence: AGPL 3.0



import scipy.optimize as opt
import numpy as np
from scipy.optimize import minimize



####################################################################
# 1) minimizing the Rosenbrock function
####################################################################

# Defining the Rosenbrock function
def obj(x):
    f = sum( 100*(x[1:]-x[:-1]**2)**2 + (1-x[:-1])**2)
    return f

# Defining the initial values
x0 = np.array([-20, 10, 23, 62, 31])


# 1) using the Nelder-Mead alghoritm: slow convergence (simplex evaluation)
res = minimize(obj, x0=x0, method='Nelder-Mead', options={'xatol': 1e-8, 'disp': True, 'maxiter': 9999})

# 2) using the BFGS alghoritm: uses the gradient of the objective function to speed up the convergence
#    with gradient
def rosen_der(x):
    xm = x[1:-1]
    xm_m1 = x[:-2]
    xm_p1 = x[2:]
    der = np.zeros_like(x)
    der[1:-1] = 200*(xm-xm_m1**2) - 400*(xm_p1 - xm**2)*xm - 2*(1-xm)
    der[0] = -400*x[0]*(x[1]-x[0]**2) - 2*(1-x[0])
    der[-1] = 200*(x[-1]-x[-2]**2)
    return der

res = minimize(obj, x0=x0, method='BFGS', options={'disp': True}, jac=rosen_der)

#    without gradient
res = minimize(obj, x0=x0, method='BFGS', options={'disp': True})


# 3) using the Newton-CG method

#    with Hessian and Jacobian
def hessian_ros(x):
    x = np.asarray(x)
    H = np.diag(-400*x[:-1],1) - np.diag(400*x[:-1],-1)
    diagonal = np.zeros_like(x)
    diagonal[0] = 1200*x[0]**2-400*x[1]+2
    diagonal[-1] = 200
    diagonal[1:-1] = 202 + 1200*x[1:-1]**2-400*x[2:]
    H = H + np.diag(diagonal)
    return H

res = minimize(obj, x0=x0, method='Newton-CG', jac=rosen_der, hess=hessian_ros, options={'disp': True, 'xtol': 1e-08})






