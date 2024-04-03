import numpy as np
from scipy.optimize import minimize

from EconModel import EconModelClass, jit

from consav.grids import nonlinspace
from consav.linear_interp import interp_1d
from consav.quadrature import log_normal_gauss_hermite

class EducationModel(EconModelClass):
    def settings(self):
        """ fundamental settings """
        pass

    def setup(self):
        """ set baseline parameters """

        # unpack
        par = self.par

        par.T = 6 # perioder
        par.zeta = 0 # prob. of being interrupted
        par.beta = 0.5 # discount rate 
        par.Nfix = 6 # number of types

        # utility of attending school correlation with background (in it is also ability in education)
        par.delta0 = 0 # Father's education
        par.delta1 = 0 # Mother's education 
        par.delta2 = 0 # Household income 
        par.delta3 = 0 # No. siblings 
        par.delta4 = 0 # Nuclear family 
        par.delta5 = 0 # Rural 
        par.delta6 = 0 # South

        par.simgaxi = 0 # std. of shock to utility of going to school

        # wages ability's correlation with background
        par.gamma0 = 0 # Father's education 
        par.gamma1 = 0 # Mother's education 
        par.gamma2 = 0 # Household income 
        par.gamma3 = 0 # No. siblings 
        par.gamma4 = 0 # Nuclear family 
        par.gamma5 = 0 # Rural 
        par.gamma6 = 0 # South

        # employment ability's correlation with background
        par.gamma0 = 0 # Father's education 
        par.gamma1 = 0 # Mother's education 
        par.gamma2 = 0 # Household income 
        par.gamma3 = 0 # No. siblings 
        par.gamma4 = 0 # Nuclear family 
        par.gamma5 = 0 # Rural 
        par.gamma6 = 0 # South

        # utility of attending school correlation with school time (splines)
        # (lige nu arbejder vi bare med 1 lineær sammenhæng)
        par.delta7 = -0.1

        # employment return to schooling 
        par.kappa1 = -0.0258 # employment return to schooling 
        par.kappa2 = -0.0146 # employment return to work experience 
        par.kappa3 = 0.0001 # employment return to work experience squared

        par.sigmae = 0 # std. of shock of being employed

        # wage 
        par.phi2 = 0.0877 # wage return to experience
        par.phi3 = -0.0030 # wage return to experience squared

        par.sigmaw = 0 # std. of shock to wage

        # wage return to schooling (splines)
        # (lige nu arbejder vi bare med 1 andengradssammenhæng)
        par.phi1 = 0.05

        par.tol_simulate = 1e-12 # tolerance when simulating household problem
    
    def allocate(self):
        """ allocate model """

        # unpack 
        par = self.par
        sol = self.sol 
        sim = self.sim 

        

