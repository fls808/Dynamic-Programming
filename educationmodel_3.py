import numpy as np
from scipy.optimize import minimize

from EconModel import EconModelClass, jit

from consav.grids import nonlinspace
from consav.linear_interp import interp_1d
from consav.linear_interp import interp_2d
from consav.linear_interp import interp_4d
from consav.quadrature import normal_gauss_hermite

class EducationModel(EconModelClass):
    def settings(self):
        """ fundamental settings """
        pass

    def setup(self):
        """ set baseline parameters """

        # unpack
        par = self.par

        par.T = 11 # 1979-1990
        par.simT = par.T 
        # par.zeta = 0 # prob. of being interrupted
        par.beta = 0.97 # discount rate 
        par.Nfix = 6 # number of types
        par.simN = par.Nfix*50 # number of households. Should be something dividable with number of types

        # utility of attending school correlation with school time (splines)
        # (lige nu arbejder vi bare med 1 lineær sammenhæng)
        par.delta7 = -0.00001

        # employment return to schooling 
        par.kappa1 = -0.0258 # employment return to schooling 
        par.kappa2 = -0.0146 # employment return to work experience 
        par.kappa3 = 0.0001 # employment return to work experience squared

        # wage 
        par.phi2 = 0.0877 # wage return to experience
        par.phi3 = -0.0030 # wage return to experience squared


        # wage return to schooling (splines)
        # (lige nu arbejder vi bare med en lineær sammenhæng)
        par.phi1 = 1 # wage return to schooling
        # 5.5 

        # grids 
        par.school_time_min = 6
        par.school_time_max = 20
        par.Nst = 15 # number of school time grid points

        par.experience_min = 0
        par.experience_max = 17
        par.Ne = 18 # number of experience grid points

        # shocks 
        par.Nepsxi = 5
        par.sigma_xi = 1 # std. of shock to utility of going to school 
        par.Nepsw = 5
        par.sigma_w = 0.2966 # std. of shock to wage
        par.Nepse = 5 
        par.sigma_e = 1.3160 # std. of shock of being employed

        # ability
        par.nuxi_1 = -2.9693 
        par.nuxi_2 = -2.7838
        par.nuxi_3 = -3.2766
        par.nuxi_4 = -3.3891
        par.nuxi_5 = -2.3878
        par.nuxi_6 = -2.7010

        par.nuw_1 = 1.5374 
        par.nuw_2 = 1.8672
        par.nuw_3 = 1.1951
        par.nuw_4 = 1.5055
        par.nuw_5 = 2.1162
        par.nuw_6 = 1.8016

        par.nue_1 = -3.4537
        par.nue_2 = -2.4784
        par.nue_3 = -3.3351
        par.nue_4 = -1.5840
        par.nue_5 = -3.6242
        par.nue_6 = -3.7365
 
        par.util_sch_fix_0 = 0 +4
        par.util_sch_fix_1 = 0.75+4
        par.util_sch_fix_2 = 1.5+4
        par.util_sch_fix_3 = 2.25+4
        par.util_sch_fix_4 = 3.0+4
        par.util_sch_fix_5 = 3.75+4
        par.util_sch_fix_6 = 4.5+4

    
    def allocate(self):
        """ allocate model """

        # unpack 
        par = self.par
        sol = self.sol 
        sim = self.sim 

        # e. school time grid
        par.school_time_grid = np.linspace(par.school_time_min,par.school_time_max,par.Nst)

        # i. experience grid
        par.experience_grid = np.linspace(par.experience_min,par.experience_max,par.Ne)

        # k. shocks grid
        par.epsxi_grid, par.epsxi_weight = normal_gauss_hermite(par.sigma_xi,par.Nepsxi)
        par.epsw_grid, par.epsw_weight = normal_gauss_hermite(par.sigma_w,par.Nepsw)
        par.epse_grid, par.epse_weight = normal_gauss_hermite(par.sigma_e,par.Nepse)

        # l. types  grid
        par.nuxi_grid = np.array([par.nuxi_1,par.nuxi_2,par.nuxi_3,par.nuxi_4,par.nuxi_5,par.nuxi_6])
        par.nuw_grid = np.array([par.nuw_1,par.nuw_2,par.nuw_3,par.nuw_4,par.nuw_5,par.nuw_6])
        par.nue_grid = np.array([par.nue_1,par.nue_2,par.nue_3,par.nue_4,par.nue_5,par.nue_6])
        par.util_sch_fix_grid = np.array([par.util_sch_fix_0,par.util_sch_fix_1,par.util_sch_fix_2,par.util_sch_fix_3,par.util_sch_fix_4,par.util_sch_fix_5,par.util_sch_fix_6])

        # m. solution arrays 
        shape = (par.T,par.Nfix,par.Nst,par.Ne,par.Nepsxi,par.Nepsw,par.Nepse)

        sol.d = np.nan + np.zeros(shape)
        sol.V = np.nan + np.zeros(shape)
        sol.school_time = np.nan + np.zeros(shape)
        sol.experience = np.nan + np.zeros(shape)
        sol.wage = np.nan + np.zeros(shape)

        # e. simulation arrays
        shape = (par.simN,par.simT)
        sim.d = np.nan + np.zeros(shape)
        sim.school_time = np.nan + np.zeros(shape)
        sim.experience = np.nan + np.zeros(shape)
        sim.type = np.zeros(shape) + np.nan
        sim.wage = np.nan + np.zeros(shape)

        # f. draws used to simulate shocks 
        np.random.seed(9210)
        sim.draws_epsxi = np.random.normal(scale=par.sigma_xi,size=shape)
        sim.draws_epsw = np.random.normal(scale=par.sigma_w,size=shape)
        sim.draws_epse = np.random.normal(scale=par.sigma_e,size=shape)

        # g. initialization
        par.block_length = par.simN // par.Nfix 
        sim.type_init = np.repeat(np.arange(par.Nfix),par.block_length)
        sim.school_time_init = np.ones(par.simN)*6
        sim.experience_init = np.zeros(par.simN)
    

    
    def solve(self):

        # unpack
        par = self.par 
        sol = self.sol


        # c. loop backwards (over all periods)
        for t in reversed(range(par.T)):
            for i_fix in range(par.Nfix):
                for i_st, school_time in enumerate(par.school_time_grid):
                    for i_e, experience in enumerate(par.experience_grid):
                            for i_epsxi, epsxi in enumerate(par.epsxi_grid):
                                for i_epsw, epsw in enumerate(par.epsw_grid):
                                    for i_epse, epse in enumerate(par.epse_grid):
                                        nuxi = par.nuxi_grid[i_fix]
                                        nue = par.nue_grid[i_fix]
                                        nuw = par.nuw_grid[i_fix]
                                        util_school_fix = par.util_sch_fix_grid[i_fix]
            
                                        # solve last period 
                                        if t == par.T-1:
                                            utility_work  = self.utility_work(school_time, experience, nue, epse, nuw,epsw)
                                            utility_school = self.utility_school(school_time, util_school_fix, nuxi, epsxi)
                                            sol.V[t,i_fix,i_st,i_e,i_epsxi,i_epsw,i_epse] = np.maximum(utility_school,utility_work)
                                            sol.d[t,i_fix,i_st,i_e,i_epsxi,i_epsw,i_epse] = utility_school > utility_work
                                                
                                            if sol.d[t,i_fix,i_st,i_e,i_epsxi,i_epsw,i_epse]== 0 :
                                                sol.wage[t,i_fix,i_st,i_e,i_epsxi,i_epsw,i_epse] = self.wage(school_time,experience,nuw,epsw)

                                            sol.school_time[t,i_fix,i_st,i_e,i_epsxi,i_epsw,i_epse] = school_time
                                            sol.experience[t,i_fix,i_st,i_e,i_epsxi,i_epsw,i_epse] = experience
                                            
                                    
                                        else:

                                            # b. bellman work
                                            bellman_work  = self.bellman_work(t, school_time, experience, i_fix, nue, epse, nuw,epsw)

                                            # a. bellman school
                                            bellman_school = self.bellman_school(t,school_time, i_fix, epsxi, nuxi)
                                        
                                            sol.V[t,i_fix,i_st,i_e,i_epsxi,i_epsw,i_epse] = np.maximum(bellman_school,bellman_work)
                                            sol.d[t,i_fix,i_st,i_e,i_epsxi,i_epsw,i_epse] = bellman_school > bellman_work

                                            if sol.d[t,i_fix,i_st,i_e,i_epsxi,i_epsw,i_epse]== 0 :
                                                sol.wage[t,i_fix,i_st,i_e,i_epsxi,i_epsw,i_epse] = self.wage(school_time,experience,nuw,epsw)
                                            

                                            sol.school_time[t,i_fix,i_st,i_e,i_epsxi,i_epsw,i_epse] = school_time
                                            sol.experience[t,i_fix,i_st,i_e,i_epsxi,i_epsw,i_epse] = experience
                                            

    def bellman_school(self,t, school_time, i_fix, epsxi, nuxi):
        """ bellman equation for school """
        par = self.par
        sol = self.sol

        # flow utility
        util_school_fix = par.util_sch_fix_grid[i_fix] 
        utility_school = self.utility_school(school_time, util_school_fix, nuxi, epsxi)

        # expected value
        EV_next_school = 0 
        school_next = school_time + 1
        experience_next = 0.0

        for i_epsxi, epsxi_next in enumerate(par.epsxi_grid):
            for i_epsw, epsw_next in enumerate(par.epsw_grid):
                for i_epse, epse_next in enumerate(par.epse_grid):
                    V_next = sol.V[t+1,i_fix,:,:,i_epsxi,i_epsw,i_epse]
                    V_next_interp = interp_2d(par.school_time_grid,par.experience_grid,V_next,school_next,experience_next)
                    EV_next_school += par.epsxi_weight[i_epsxi]*par.epsw_weight[i_epsw]*par.epse_weight[i_epse] * V_next_interp

        bellman_school = utility_school + par.beta* EV_next_school
        return bellman_school

    def bellman_work(self,t, school_time, experience, i_fix, nue, epse, nuw,epsw):
        """ bellman equation for work """
        par = self.par
        sol = self.sol

        # flow utility
        utility_work = self.utility_work(school_time, experience, nue, epse, nuw,epsw)

        experience_next = experience + 1 
        school_next = school_time

        EV_next_work = 0

        # det er lige meget med shock til skole 
        for i_epsw, epsw_next in enumerate(par.epsw_grid):
            for i_epse, epse_next in enumerate(par.epse_grid):
                V_next = sol.V[t+1,i_fix,:,:,0,i_epsw,i_epse]
                V_next_interp = interp_2d(par.school_time_grid,par.experience_grid,V_next,school_next,experience_next)
                EV_next_work += par.epsw_weight[i_epsw]*par.epse_weight[i_epse] * V_next_interp
        bellman_work = utility_work + par.beta* EV_next_work
        return bellman_work

    
    def utility_school(self,school_time, util_school_fix, nuxi, epsxi):
        """ utility of attending school """
        # unpack
        par = self.par
        school_time_util = par.delta7*school_time
        utility_school = util_school_fix + school_time_util + nuxi + epsxi
        return utility_school    
    
    def utility_work(self, school_time, experience, nue, epse, nuw,epsw):
        """ utility of working """
        par = self.par
        wage = self.wage(school_time, experience, nuw,epsw)
        e = 1/np.exp(np.exp(self.logestar(school_time, experience, epse, nue)))
        return np.log(e*wage)
        
    def wage(self, school_time, experience, nuw,epsw):
        return np.exp(self.logwage(school_time, experience, nuw, epsw))

    def logwage(self, school_time, experience, nuw, epsw):
        """ log wage """
        par = self.par
        return par.phi1*school_time + par.phi2*experience + par.phi3*experience**2 + nuw + epsw
    
    def logestar(self, school_time, experience, epse, nue):
        par = self.par
        return par.kappa1*school_time + par.kappa2*experience + par.kappa3*experience**2 + nue + epse

    def simulate(self):
        """ simulate model """
        par = self.par
        sol = self.sol
        sim = self.sim

        for i in range(par.simN):
            # i. initialize states
            sim.school_time[i,0] = sim.school_time_init[i]
            sim.experience[i,0] = sim.experience_init[i]
            sim.type[i,0] = sim.type_init[i]
           
            for t in range(par.simT):
                i_fix = int(sim.type[i,t])
 
                school_time_index =int(sim.school_time[i,t]-6)
                #print i and school time

                sol_d = sol.d[t,i_fix,school_time_index,:,:,:,:]

                if sim.d[i,t-1] == 0:
                    sim.d[i,t] = 0
                
                else:
                    sim.d[i,t] = interp_4d(par.experience_grid,par.epsxi_grid,par.epsw_grid,par.epse_grid,
                                           sol_d,sim.experience[i,t],sim.draws_epsxi[i,t],sim.draws_epsw[i,t],sim.draws_epse[i,t])


                if sim.d[i,t] == 0:
                    sim.wage[i,t] = self.wage(school_time_index,sim.experience[i,t],par.nuw_grid[i_fix],sim.draws_epsw[i,t])
               

                if t < par.simT-1:
                    sim.experience[i,t+1] = sim.experience[i,t]+ (1-sim.d[i,t])

                    sim.school_time[i,t+1] = sim.school_time[i,t] + sim.d[i,t]
                    

                    sim.type[i,t+1] = sim.type[i,t]
                

                





