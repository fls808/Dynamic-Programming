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

        par.T = 65 # perioder
        par.school_max = 22
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
        par.gamma0_w = 0 # Father's education 
        par.gamma1_w = 0 # Mother's education 
        par.gamma2_w = 0 # Household income 
        par.gamma3_w = 0 # No. siblings 
        par.gamma4_w = 0 # Nuclear family 
        par.gamma5_w = 0 # Rural 
        par.gamma6_w = 0 # South

        # employment ability's correlation with background
        par.gamma0_e = 0 # Father's education 
        par.gamma1_e = 0 # Mother's education 
        par.gamma2_e = 0 # Household income 
        par.gamma3_e = 0 # No. siblings 
        par.gamma4_e = 0 # Nuclear family 
        par.gamma5_e = 0 # Rural 
        par.gamma6_e = 0 # South

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

        # a. father education


    
    def bellman(self,ev0,output=1):
        
        # unpack 
        par = self.par

        # Value of options 
        
        # pstop = 
        pinterup = par.zeta
    
    def solve(self):

        # unpack
        par = self.par 
        sol = self.sol

        # b. solve last period 

        # c. loop backwards (over all periods)
        for t in reversed(range(par.T)):

            # loop over state variables: family background, school time, experience, and shocks
            for i_f in enumerate(par.father_grid):
                for i_m in enumerate(par.mother_grid):
                    for i_inc in enumerate(par.income_grid):
                        for i_sib in enumerate(par.siblings_grid):
                            for i_nuc in enumerate(par.nuclear_grid):
                                for i_rur in enumerate(par.rural_grid):
                                    for i_sou in enumerate(par.south_grid):
                                        for i_s in enumerate(par.school_time_grid):
                                            for i_e in enumerate(par.experience_grid):
                                               for i_eps_xi in enumerate():



      
    
    def utility_school(self, father, mother, income, siblings, nuclear, rural, south, school_time):
        """ utility of attending school """
        # unpack
        par = self.par
        familiy_util = par.delta0*father + par.delta1*mother + par.delta2*income + par.delta3*siblings + par.delta4*nuclear + par.delta5*rural + par.delta6*south
        school_time_util = par.delta7*school_time
        utility_school = familiy_util + school_time_util

        return utility_school
    
    def utility_work(self, e,school_time, experience):
        """ utility of working """
        par = self.par
        wage = np.exp(self.logwage(school_time, experience))
        e = 1/np.exp(np.exp(self.logestar(school_time, experience)))
        return np.log(e*wage)
        
    def logwage(self, school_time, experience):
        """ log wage """
        par = self.par
        return par.phi1*school_time + par.phi2*experience + par.phi3*experience**2
    
    def logestar(self, school_time, experience):
        par = self.par
        return par.kappa1*school_time + par.kappa2*experience + par.kappa3*experience**2
