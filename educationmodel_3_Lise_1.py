import numpy as np
from scipy.optimize import minimize

import pandas as pd 

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
        par.zeta = 0 # prob. of being interrupted
        par.beta = 0.97 # discount rate 
        par.Nfix = 6 # number of types
        par.simN = par.Nfix*50 # number of households. Should be something dividable with number of types
        par.N = 1710 # number of observations

        par.delta0 = 0.0205 # Father's education
        par.delta1 = -0.0080 # Mother's education 
        par.delta2 = 0.0017 # Household income 
        par.delta3 = -0.0156 # No. siblings 
        par.delta4 = 0.0387 # Nuclear family 
        par.delta5 = -0.0618 # Rural 
        par.delta6 = -0.0412 # South

        # wages ability's correlation with background
        par.gamma0_w = 0.0106 # Father's education 
        par.gamma1_w = -0.0144 # Mother's education 
        par.gamma2_w = 0.0012 # Household income 
        par.gamma3_w = -0.0084 # No. siblings 
        par.gamma4_w = 0.0225 # Nuclear family 
        par.gamma5_w = -0.0591 # Rural 
        par.gamma6_w = -0.0363 # South

        # employment ability's correlation with background
        par.gamma0_e = -0.0221 # Father's education 
        par.gamma1_e = -0.0031 # Mother's education 
        par.gamma2_e = -0.0006 # Household income 
        par.gamma3_e = 0.0123 # No. siblings 
        par.gamma4_e = -0.0100 # Nuclear family 
        par.gamma5_e = 0.0559 # Rural 
        par.gamma6_e = -0.0986 # South

        # utility of attending school correlation with school time (splines)
        # (lige nu arbejder vi bare med 1 lineær sammenhæng)
        par.delta7 = -0.00001 #Forstår ikke denne værdi....

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
        par.school_time_max = 20 #Er det ikke 22?
        par.Nst = 15 # number of school time grid points

        par.experience_min = 0
        par.experience_max = 17
        par.Ne = 18 # number of experience grid points

        #Probability of interrupting school
        par.zeta = 0.0749 #Fra paper indtil videre

        # shocks 
        # par.Nepsxi = 5
        # par.sigma_xi = 1 # std. of shock to utility of going to school 
        # par.Nepsw = 5
        # par.sigma_w = 0.2966 # std. of shock to wage
        # par.Nepse = 5 
        # par.sigma_e = 1.3160 # std. of shock of being employed

        # ability
        par.nuxi_tilde_1 = -2.9693 
        par.nuxi_tilde_2 = -2.7838
        par.nuxi_tilde_3 = -3.2766
        par.nuxi_tilde_4 = -3.3891
        par.nuxi_tilde_5 = -2.3878
        par.nuxi_tilde_6 = -2.7010

        par.nuw_tilde_1 = 1.5374 
        par.nuw_tilde_2 = 1.8672
        par.nuw_tilde_3 = 1.1951
        par.nuw_tilde_4 = 1.5055
        par.nuw_tilde_5 = 2.1162
        par.nuw_tilde_6 = 1.8016

        par.nue_tilde_1 = -3.4537
        par.nue_tilde_2 = -2.4784
        par.nue_tilde_3 = -3.3351
        par.nue_tilde_4 = -1.5840
        par.nue_tilde_5 = -3.6242
        par.nue_tilde_6 = -3.7365

        #par.nuxi_fix_min = -4
        #par.nuxi_fix_max = 4
        #par.Nnuxi_fix = 9

        par.nuw_fix_min = 0.13
        par.nuw_fix_max = 181
        par.Nnuw_fix = 15

        par.nue_fix_min = -0.91
        par.nue_fix_max = -0.20
        par.Nnue_fix = 3

        par.util_sch_fix_min = 0.04
        par.util_sch_fix_max = 256
        par.Nutil_sch_fix = 15

        par.q1 = 0.6286
        par.q2 = -0.3823
        par.q3 = -0.4227
        par.q4 = -0.4513
        par.q5 = -0.0776
        par.q6 = 0



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

        # l. types  grid
        par.nuxi_tilde_grid = np.array([par.nuxi_tilde_1,par.nuxi_tilde_2,par.nuxi_tilde_3,par.nuxi_tilde_4,par.nuxi_tilde_5,par.nuxi_tilde_6])
        par.nuw_tilde_grid = np.array([par.nuw_tilde_1,par.nuw_tilde_2,par.nuw_tilde_3,par.nuw_tilde_4,par.nuw_tilde_5,par.nuw_tilde_6])
        par.nue_tilde_grid = np.array([par.nue_tilde_1,par.nue_tilde_2,par.nue_tilde_3,par.nue_tilde_4,par.nue_tilde_5,par.nue_tilde_6])

        #par.nuxi_fix_grid = np.linspace(par.nuxi_fix_min,par.nuxi_fix_max,par.Nnuxi_fix)
        par.nuw_fix_grid = np.linspace(par.nuw_fix_min,par.nuw_fix_max,par.Nnuw_fix)
        par.nue_fix_grid = np.linspace(par.nue_fix_min,par.nue_fix_max,par.Nnue_fix)
        par.util_sch_fix_grid = np.linspace(par.util_sch_fix_min,par.util_sch_fix_max,par.Nutil_sch_fix)

        # m. solution arrays 
        shape = (par.T,par.Nfix,par.Nnuw_fix,par.Nnue_fix,par.Nutil_sch_fix,par.Nst,par.Ne)

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
        sim.u_d = np.random.rand(par.simN,par.simT) # choice of school
        #sim.draws_epsxi = np.random.normal(scale=par.sigma_xi,size=shape)
        #sim.draws_epsw = np.random.normal(scale=par.sigma_w,size=shape)
        #sim.draws_epse = np.random.normal(scale=par.sigma_e,size=shape)

        # g. initialization
        par.block_length = par.simN // par.Nfix 
        sim.type_init = np.repeat(np.arange(par.Nfix),par.block_length)
        sim.school_time_init = np.ones(par.simN)*6
        sim.experience_init = np.zeros(par.simN)

        par.p1 = np.exp(par.q1)/(np.exp(par.q1)+np.exp(par.q2)+np.exp(par.q3)+np.exp(par.q4)+np.exp(par.q5)+np.exp(par.q6))
        par.p2 = np.exp(par.q2)/(np.exp(par.q1)+np.exp(par.q2)+np.exp(par.q3)+np.exp(par.q4)+np.exp(par.q5)+np.exp(par.q6))
        par.p3 = np.exp(par.q3)/(np.exp(par.q1)+np.exp(par.q2)+np.exp(par.q3)+np.exp(par.q4)+np.exp(par.q5)+np.exp(par.q6))
        par.p4 = np.exp(par.q4)/(np.exp(par.q1)+np.exp(par.q2)+np.exp(par.q3)+np.exp(par.q4)+np.exp(par.q5)+np.exp(par.q6))
        par.p5 = np.exp(par.q5)/(np.exp(par.q1)+np.exp(par.q2)+np.exp(par.q3)+np.exp(par.q4)+np.exp(par.q5)+np.exp(par.q6))
        par.p6 = np.exp(par.q6)/(np.exp(par.q1)+np.exp(par.q2)+np.exp(par.q3)+np.exp(par.q4)+np.exp(par.q5)+np.exp(par.q6))
    
    
    def solve(self):

        # unpack
        par = self.par 
        sol = self.sol


        # c. loop backwards (over all periods)
        for t in reversed(range(par.T)):
            for i_fix in range(par.Nfix):
                for i_nuw_fix, nuw_fix in enumerate(par.nuw_fix_grid):
                    for i_nue_fix, nue_fix in enumerate(par.nue_fix_grid):
                        for i_util_sch_fix, util_sch_fix in enumerate(par.util_sch_fix_grid):
                            for i_st, school_time in enumerate(par.school_time_grid):
                                for i_e, experience in enumerate(par.experience_grid):
                                    nuxi = par.nuxi_tilde_grid[i_fix] 
                                    nue = par.nue_tilde_grid[i_fix] + nue_fix
                                    nuw = par.nuw_tilde_grid[i_fix] + nuw_fix

                                    # solve last period 
                                    if t == par.T-1:
                                        utility_work  = self.utility_work(school_time, experience, nue, nuw)
                                        utility_school = self.utility_school(school_time, util_sch_fix, nuxi)

                                        maxV = np.maximum(utility_school,utility_work) 
                                        sol.V[t,i_fix,i_nuw_fix,i_nue_fix,i_util_sch_fix,i_st,i_e] = (maxV + np.log(np.exp(utility_school-maxV) + np.exp(utility_work-maxV)))
                                        sol.d[t,i_fix,i_nuw_fix,i_nue_fix,i_util_sch_fix,i_st,i_e] = 1/(1+np.exp(utility_school-utility_work))
                                        
                                        sol.wage[t,i_fix,i_nuw_fix,i_nue_fix,i_util_sch_fix,i_st,i_e] = self.wage(school_time,experience,nuw)

                                        sol.school_time[t,i_fix,i_nuw_fix,i_nue_fix,i_util_sch_fix,i_st,i_e] = school_time
                                        sol.experience[t,i_fix,i_nuw_fix,i_nue_fix,i_util_sch_fix,i_st,i_e] = experience

                                
                                    else:

                                        # b. bellman work
                                        bellman_work  = self.bellman_work(t, school_time, experience, i_fix, nue, nuw,i_nuw_fix,i_nue_fix,i_util_sch_fix)

                                        # a. bellman school
                                        bellman_school = self.bellman_school(t,school_time,i_fix,util_sch_fix, nuxi,i_nuw_fix,i_nue_fix,i_util_sch_fix)

                                        maxV = np.maximum(bellman_school,bellman_work)
                                        sol.V[t,i_fix,i_nuw_fix,i_nue_fix,i_util_sch_fix,i_st,i_e] = (maxV + np.log(np.exp(bellman_school-maxV) + np.exp(bellman_work-maxV)))
                                        sol.d[t,i_fix,i_nuw_fix,i_nue_fix,i_util_sch_fix,i_st,i_e] = 1/(1+np.exp(bellman_school-bellman_work))

                                        sol.wage[t,i_fix,i_nuw_fix,i_nue_fix,i_util_sch_fix,i_st,i_e] = self.wage(school_time,experience,nuw)

                                        sol.school_time[t,i_fix,i_nuw_fix,i_nue_fix,i_util_sch_fix,i_st,i_e] = school_time
                                        sol.experience[t,i_fix,i_nuw_fix,i_nue_fix,i_util_sch_fix,i_st,i_e] = experience

                                        

    def bellman_school(self,t, school_time,i_fix,util_sch_fix, nuxi,i_nuw_fix,i_nue_fix,i_util_sch_fix):
        """ bellman equation for school """
        par = self.par
        sol = self.sol

        # flow utility
        utility_school = self.utility_school(school_time, util_sch_fix, nuxi)

        # expected value
        EV_next_school = 0 
        school_next = school_time + 1

        # Faktisk overflødigt at interpolere, da vi jo ikke rammer udenfor grid point. (Men så ikke alligevel, da der jo ikke er nok points)
        V_next = sol.V[t+1,i_fix,i_nuw_fix,i_nue_fix,i_util_sch_fix,:,:]
        EV_next_school = interp_2d(par.school_time_grid,par.experience_grid,V_next,school_next,0)
        EV_next_interupt = interp_2d(par.school_time_grid,par.experience_grid,V_next,school_time,0)

        bellman_school = utility_school + par.beta * (par.zeta * EV_next_interupt + (1-par.zeta) * EV_next_school)
        return bellman_school

    def bellman_work(self,t, school_time, experience, i_fix, nue, nuw,i_nuw_fix,i_nue_fix,i_util_sch_fix):
        """ bellman equation for work """
        par = self.par
        sol = self.sol

        # flow utility
        utility_work = self.utility_work(school_time, experience, nue, nuw)

        experience_next = experience + 1 

        V_next = sol.V[t+1,i_fix,i_nuw_fix,i_nue_fix,i_util_sch_fix,:,:]
        # Kan vi bare gøre pga. extreme value. 
        EV_next_work = interp_2d(par.school_time_grid,par.experience_grid,V_next,school_time,experience_next)

        bellman_work = utility_work + par.beta* EV_next_work
        return bellman_work

    
    def utility_school(self,school_time, util_school_fix, nuxi):
        """ utility of attending school """
        # unpack
        par = self.par
        school_time_util = par.delta7*school_time
        utility_school = util_school_fix + school_time_util + nuxi
        return utility_school    
    
    def utility_work(self, school_time, experience, nue, nuw):
        """ utility of working """
        par = self.par
        wage = self.wage(school_time, experience, nuw)
        e = 1/np.exp(np.exp(self.logestar(school_time, experience, nue)))
        return np.log(e*wage)
        
    def wage(self, school_time, experience, nuw):
        logwage_value = self.logwage(school_time, experience, nuw)
        max_exp_value = 709
        clamped_logwage_value = np.clip(logwage_value, -max_exp_value, max_exp_value)
        return np.exp(clamped_logwage_value)

    def logwage(self, school_time, experience, nuw):
        """ log wage """
        par = self.par
        return par.phi1*school_time + par.phi2*experience + par.phi3*experience**2 + nuw
    
    def logestar(self, school_time, experience, nue):
        par = self.par
        return par.kappa1*school_time + par.kappa2*experience + par.kappa3*experience**2 + nue
    


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
                sol_d = sol.d[t,i_fix,school_time_index,:]

                # Hvis du var på arbejde i sidste periode, så er du det også i denne periode
                if sim.d[i,t-1] == 0:
                    sim.d[i,t] = 0
                
                else:
                    sim.d[i,t] = sim.u_d[i,t] < interp_1d(par.experience_grid,sol_d,sim.experience[i,t])
                    
                if sim.d[i,t] == 0:
                    sim.wage[i,t] = self.wage(school_time_index,sim.experience[i,t],par.nuw_grid[i_fix])
               
                if t < par.simT-1:
                    sim.experience[i,t+1] = sim.experience[i,t]+ (1-sim.d[i,t])

                    sim.school_time[i,t+1] = sim.school_time[i,t] + sim.d[i,t]

                    sim.type[i,t+1] = sim.type[i,t]
                

                





