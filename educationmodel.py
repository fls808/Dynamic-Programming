import numpy as np
from scipy.optimize import minimize

from EconModel import EconModelClass, jit

from consav.grids import nonlinspace
from consav.linear_interp import interp_1d
from consav.quadrature import normal_gauss_hermite

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

        # wage 
        par.phi2 = 0.0877 # wage return to experience
        par.phi3 = -0.0030 # wage return to experience squared


        # wage return to schooling (splines)
        # (lige nu arbejder vi bare med 1 andengradssammenhæng)
        par.phi1 = 0.05


        # grids
        par.dad_educ_min = 0
        par.dad_educ_max = 20
        par.Nd = 20 # number of dad education grid points

        par.mom_educ_min = 0
        par.mom_educ_max = 20
        par.Nm = 20 # number of mom education grid points

        par.num_sib_min = 0
        par.num_sib_max = 15
        par.Ns = 15 # number of siblings grid points

        par.income_min = 0
        par.income_max = 155000
        par.Ni = 155 # number of income grid points

        par.school_time_min = 6
        par.school_time_max = 20
        par.Ns = 15 # number of school time grid points

        par.experience_min = 0
        par.experience_max = 17
        par.Ne = 34 # number of experience grid points

        par.wage_min = 2
        par.wage_max = 40 
        par.Nw = 38 # number of wage grid points

        # shocks 
        par.Nepsxi = 5
        par.sigmaxi = 0 # std. of shock to utility of going to school 
        par.Nepsw = 5
        par.sigmaw = 0 # std. of shock to wage
        par.Nepse = 5 
        par.sigmae = 0 # std. of shock of being employed

        # ability
        par.Nnuxi = 5
        par.Nnue = 5
        par.Nnuw = 5

        par.tol_simulate = 1e-12 # tolerance when simulating household problem
    
    def allocate(self):
        """ allocate model """

        # unpack 
        par = self.par
        sol = self.sol 
        sim = self.sim 

        # Types 

        # a. father education
        par.father_grid = np.linspace(par.dad_educ_min,par.dad_educ_max,par.Nd)
        
        # b. mother education
        par.mother_grid = np.linspace(par.mom_educ_min,par.mom_educ_max,par.Nm)

        # c. income grid 
        par.income_grid = nonlinspace(par.income_min,par.income_max,par.Ni,1.1)

        # d. siblings grid
        par.siblings_grid = np.linspace(par.num_sib_min,par.num_sib_max,par.Ns)

        # e. nuclear family
        par.nuclear_grid = np.array([0,1])

        # f. rural
        par.rural_grid = np.array([0,1])

        # g. south
        par.south_grid = np.array([0,1])

        #_____________
        # h. school time grid
        par.school_time_grid = np.linspace(par.school_time_min,par.school_time_max,par.Ns)

        # i. experience grid
        par.experience_grid = nonlinspace(par.experience_min,par.experience_max,par.Ne,1.1)

        # j. wage grid
        par.wage_grid = nonlinspace(par.wage_min,par.wage_max,par.Nw,1.1)

        # k. shocks grid
        par.epsxi_grid, par.epsxi_weight = normal_gauss_hermite(par.sigma_xi,par.Nepsxi)
        par.epsw_grid, par.epsw_weight = normal_gauss_hermite(par.sigma_w,par.Nepsw)
        par.epse_grid, par.epse_weight = normal_gauss_hermite(par.sigma_e,par.Nepse)
    
    def solve(self):

        # unpack
        par = self.par 
        sol = self.sol

        # b. solve last period 

        # c. loop backwards (over all periods)
        for t in reversed(range(par.T)):

            # loop over state variables: family background, school time, experience, and shocks
            for i_f, father in enumerate(par.father_grid):
                for i_m, mother in enumerate(par.mother_grid):
                    for i_i, income in enumerate(par.income_grid):
                        for i_s, siblings in enumerate(par.siblings_grid):
                            for i_n, nuclear in enumerate(par.nuclear_grid):
                                for i_r, rural in enumerate(par.rural_grid):
                                    for i_s, south in enumerate(par.south_grid):
                                        for i_st, school_time in enumerate(par.school_time_grid):
                                            for i_e, experience in enumerate(par.experience_grid):
                                                for i_epsxi, epsxi in enumerate(par.epsxi_grid):
                                                    for i_epsw, epsw in enumerate(par.epsw_grid):
                                                        for i_epse, epse in enumerate(par.epse_grid):
                                                            pass

    def bellman_school(self,t, father, mother, income, siblings, nuclear, rural, south, school_time, epsxi, nuxi):
        """ bellman equation for school """
        par = self.par
        sol = self.sol

        # flow utility
        util = self.utility_school(father, mother, income, siblings, nuclear, rural, south, school_time, nuxi, epsxi)

        # expected value
        V_next = sol.V[t+1,father,mother,income,siblings,nuclear,rural,south,school_time]
        EV_next = 0 

        for i_epsxi, epsxi_next in enumerate(par.epsxi_grid):
            for i_epsw, epsw_next in enumerate(par.epsw_grid):
                for i_epse, epse_next in enumerate(par.epse_grid):
                    V_next = 
                    EV_next += 
        
        
                                                            
    
    def ability_school(self,nuxi):
        """ ability of attending school """
        return nuxi
    
    def ability_wage(self,nuw,father, mother, income, siblings, nuclear, rural, south):
        """ ability wage """
        par = self.par
        return nuw + par.gamma0_w*father + par.gamma1_w*mother + par.gamma2_w*income + par.gamma3_w*siblings + par.gamma4_w*nuclear + par.gamma5_w*rural + par.gamma6_w*south
    
    def ability_employment(self,nue,father, mother, income, siblings, nuclear, rural, south):
        """ ability employment """
        par = self.par
        return nue + par.gamma0_e*father + par.gamma1_e*mother + par.gamma2_e*income + par.gamma3_e*siblings + par.gamma4_e*nuclear + par.gamma5_e*rural + par.gamma6_e*south
    
    
    def utility_school(self, father, mother, income, siblings, nuclear, rural, south, school_time, nuxi, epsxi):
        """ utility of attending school """
        # unpack
        par = self.par
        familiy_util = par.delta0*father + par.delta1*mother + par.delta2*income + par.delta3*siblings + par.delta4*nuclear + par.delta5*rural + par.delta6*south
        school_time_util = par.delta7*school_time
        ability_school = self.ability_school(nuxi)

        utility_school = familiy_util + school_time_util + ability_school + epsxi
        return utility_school
    
    def utility_work(self, school_time, experience,epsw, father, mother, income, siblings, nuclear, rural, south):
        """ utility of working """
        par = self.par
        wage = np.exp(self.logwage(school_time, experience, epsw, father, mother, income, siblings, nuclear, rural, south))
        e = 1/np.exp(np.exp(self.logestar(school_time, experience)))
        return np.log(e*wage)
        
    def logwage(self, school_time, experience, epsw, father, mother, income, siblings, nuclear, rural, south):
        """ log wage """
        par = self.par
        ability_wage = self.ability_wage(father, mother, income, siblings, nuclear, rural, south)
        return par.phi1*school_time + par.phi2*experience + par.phi3*experience**2 + ability_wage + epsw
    
    def logestar(self, school_time, experience, epse, nue, father, mother, income, siblings, nuclear, rural, south):
        par = self.par
        ability_employment = self.ability_employment(nue,father, mother, income, siblings, nuclear, rural, south)
        return par.kappa1*school_time + par.kappa2*experience + par.kappa3*experience**2 + ability_employment + epse
