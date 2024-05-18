import numpy as np
from scipy.optimize import minimize

from EconModel import EconModelClass, jit

from consav.grids import nonlinspace
from consav.linear_interp import interp_1d
from consav.linear_interp import interp_2d
from consav.linear_interp import interp_4d
from consav.quadrature import normal_gauss_hermite

def log_likelihood(theta, model,est_par,data):

    par = model.par
    sol = model.sol

    # Update the model parameters
    par.update(dict(zip(est_par, theta)))
    model.setup()

    # Solve the model
    sch = np.array(data.sch)
    w = np.array(data.w)
    enter = np.array(data.enter)

    
    d_predict = interp_1d()