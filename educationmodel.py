import numpy as np

from EconModel import EconModelClass, jit

from consav.linear_interp import interp_2d

class EducationModel(EconModelClass):
    def settings(self):
        """ fundamental settings """
        pass

    def setup(self):
        """ set baseline parameters """

        # unpack
        par = self.par

        # Basis parameters
        par.T = 10 # 1980-1989
        par.simT = par.T 
        par.beta = 0.97 # Discount rate 
        par.Nfix = 4 # Number of types
        par.simN = par.Nfix*500 # Number of households
        par.N = 1710 # Number of observations

        # Utility of attending school correlation with background
        par.delta0 = 0.0205 # Father's education
        par.delta1 = -0.0080 # Mother's education 
        par.delta2 = 0.0017 # Household income 
        par.delta3 = -0.0156 # No. siblings 
        par.delta4 = 0.0387 # Nuclear family 
        par.delta5 = -0.0618 # Rural 
        par.delta6 = -0.0412 # South

        # Wage ability's correlation with background
        par.gamma0_w = 0.0106 # Father's education 
        par.gamma1_w = -0.0144 # Mother's education 
        par.gamma2_w = 0.0012 # Household income 
        par.gamma3_w = -0.0084 # No. siblings 
        par.gamma4_w = 0.0225 # Nuclear family 
        par.gamma5_w = -0.0591 # Rural 
        par.gamma6_w = -0.0363 # South

        # Utility of attending school correlation with school time 
        par.delta7 = -0.00001 # School time  
        par.delta8 = -0.002 # School time squared

        # Employment return to schooling 
        par.kappa1 = -0.0258 # employment return to schooling 
        par.kappa2 = -0.002 # employment return to schooling squared

        par.kappa3 = -0.0146 # employment return to work experience 
        par.kappa4 = -0.002 # employment return to work experience squared

        # Wage return to schooling 
        par.phi1 = 0.15 # wage return to schooling
        par.phi2 = -0.004 #wage return to schooling squared

        # Wage 
        par.phi3 = 0.0877 # wage return to experience
        par.phi4 = -0.0030 # wage return to experience squared

        # Grids 
        par.school_time_min = 6 #Minimum amount of years in school
        par.school_time_max = 20  #Maxium amount of years in school
        par.Nst = 15 # number of school time grid points

        par.experience_min = 0 # Minimum amount of experience
        par.experience_max = 17 # Maximum amount of experience
        par.Ne = 18 # number of experience grid points

        #Probability of interrupting school
        par.zeta = 0.0749 #Fra paper indtil videre

        # Estimated parameters from Belzil and Hansen (2003)
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

        par.q1 = 0.6286
        par.q2 = -0.3823
        par.q3 = -0.4227
        par.q4 = 0 #Normalized to 0

        # Probability of seeing the different types
        par.p1 = np.exp(par.q1)/(np.exp(par.q1)+np.exp(par.q2)+np.exp(par.q3)+np.exp(par.q4))
        par.p2 = np.exp(par.q2)/(np.exp(par.q1)+np.exp(par.q2)+np.exp(par.q3)+np.exp(par.q4))
        par.p3 = np.exp(par.q3)/(np.exp(par.q1)+np.exp(par.q2)+np.exp(par.q3)+np.exp(par.q4))
        par.p4 = np.exp(par.q4)/(np.exp(par.q1)+np.exp(par.q2)+np.exp(par.q3)+np.exp(par.q4))

        par.nuw_fix_min = 0.13 # Minimum value, chosen from observing data
        par.nuw_fix_max = 181 # Maximum value, chosen from observing data
        par.Nnuw_fix = 15 # Number of grid points

        par.util_sch_fix_min = 0.04 # Minimum value, chosen from observing data
        par.util_sch_fix_max = 256 # Maximum value, chosen from observing data
        par.Nutil_sch_fix = 15 # Number of grid points


    def allocate(self):
        """ allocate model """

        # unpack 
        par = self.par
        sol = self.sol 
        sim = self.sim 

        # School time grid
        par.school_time_grid = np.linspace(par.school_time_min,par.school_time_max,par.Nst)

        # Experience grid
        par.experience_grid = np.linspace(par.experience_min,par.experience_max,par.Ne) 

        # Types grid
        par.nuxi_tilde_grid = np.array([par.nuxi_tilde_1,par.nuxi_tilde_2,par.nuxi_tilde_3,par.nuxi_tilde_4]) # Orthogonal school ability
        par.nuw_tilde_grid = np.array([par.nuw_tilde_1,par.nuw_tilde_2,par.nuw_tilde_3,par.nuw_tilde_4]) # Orthogonal work ability
        par.nuw_fix_grid = np.linspace(par.nuw_fix_min,par.nuw_fix_max,par.Nnuw_fix) # Work ability from family characteristics
        par.util_sch_fix_grid = np.linspace(par.util_sch_fix_min,par.util_sch_fix_max,par.Nutil_sch_fix) # Utility school from family characteristics

        # Solution arrays 
        shape = (par.T,par.Nfix,par.Nnuw_fix,par.Nutil_sch_fix,par.Nst,par.Ne)
        sol.d = np.nan + np.zeros(shape)
        sol.V = np.nan + np.zeros(shape)
        sol.school_time = np.nan + np.zeros(shape)
        sol.experience = np.nan + np.zeros(shape)
        sol.wage = np.nan + np.zeros(shape)

        # Simulation arrays
        shape = (par.simN,par.simT)
        sim.school = np.nan + np.zeros(shape)
        sim.work = np.nan + np.zeros(shape)
        sim.interrupt = np.nan + np.zeros(shape)
        sim.school_time = np.nan + np.zeros(shape)
        sim.experience = np.nan + np.zeros(shape)
        sim.wage = np.nan + np.zeros(shape)
        sim.util_school_fix = np.nan + np.zeros(par.simN)
        sim.abil_work_fix = np.nan + np.zeros(par.simN)

        values_school = [50,200,50,200]  # Four different values for utility school for simulation 2
        values_work = [30,30,120,120] # Four different values for ability work for simulation 2

        num_occurrences = par.simN // len(values_school) # Number of occurences for the 4 different types

        sim.util_school_fix_2 = np.tile(values_school, num_occurrences) # Creates a vector with the chosen values for utility school 
        sim.abil_work_fix_2 = np.tile(values_work, num_occurrences) # Creates a vector with the chosen values for ability work

        # In case par.simN is not perfectly divisible by 4, handle the remainder
        remainder = par.simN % len(values_school)
        if remainder != 0:
            sim.util_school_fix_2 = np.concatenate([sim.util_school_fix_2, np.array(values_school[:remainder])])
            sim.abil_work_fix_2 = np.concatenate([sim.abil_work_fix_2, np.array(values_work[:remainder])])


        # Draws used to simulate shocks 
        np.random.seed(9210)
        sim.u_d = np.random.rand(par.simN,par.simT) # choice of school
        sim.interrupt_d = np.random.rand(par.simN,par.simT) #choice of interruption

        # Initialization for simulation
        par.block_length = par.simN // par.Nfix # Number of observations of each type
        sim.type = np.repeat(np.arange(par.Nfix),par.block_length) # Creates a vector that repeats the different types.

        # Randomly drawn values for the simulation. The intervals are chosen on the basis of the observed data.
        sim.school_time_init = np.random.randint(6,16, size=par.simN)
        sim.experience_init = np.zeros(par.simN)
        sim.Dad_educ = np.random.randint(0, 21, size=par.simN)
        sim.Mom_educ = np.random.randint(0, 21, size=par.simN)
        sim.Num_siblings = np.random.randint(0, 16, size=par.simN)
        sim.Urban = np.random.randint(0,2,size=par.simN)
        sim.Nuclear = np.random.randint(0,2,size=par.simN)
        sim.Family_income = np.random.uniform(0,150500,size=par.simN)
        sim.South = np.random.randint(0,2,size=par.simN)

    
    def solve(self):

        # Unpack
        par = self.par 
        sol = self.sol


        # Loop backwards (over all periods)
        for t in reversed(range(par.T)):
            for i_fix in range(par.Nfix):
                for i_nuw_fix, nuw_fix in enumerate(par.nuw_fix_grid):
                    for i_util_sch_fix, util_sch_fix in enumerate(par.util_sch_fix_grid):
                        for i_st, school_time in enumerate(par.school_time_grid):
                            for i_e, experience in enumerate(par.experience_grid):
                                nuxi = par.nuxi_tilde_grid[i_fix] 
                                nuw = par.nuw_tilde_grid[i_fix] + nuw_fix

                                # solve last period 
                                if t == par.T-1:
                                    utility_work  = self.utility_work(school_time, experience, nuw)
                                    utility_school = self.utility_school(school_time, util_sch_fix, nuxi)

                                    maxV = np.maximum(utility_school,utility_work) 

                                    # Expected maximum value
                                    sol.V[t,i_fix,i_nuw_fix,i_util_sch_fix,i_st,i_e] = (maxV + np.log(np.exp(utility_school-maxV) + np.exp(utility_work-maxV)))
                                    
                                    # Probability of school being the optimal choice
                                    sol.d[t,i_fix,i_nuw_fix,i_util_sch_fix,i_st,i_e] = 1/(1+np.exp(utility_work-utility_school))
                                    
                                else: # If we're not in the last period

                                    # Bellman work
                                    bellman_work  = self.bellman_work(t, school_time, experience, i_fix, nuw,i_nuw_fix,i_util_sch_fix)

                                    # Bellman school
                                    bellman_school = self.bellman_school(t,school_time,experience,i_fix,util_sch_fix, nuxi,i_nuw_fix,i_util_sch_fix)

                                    maxV = np.maximum(bellman_school,bellman_work)

                                    # Expected maximum value
                                    sol.V[t,i_fix,i_nuw_fix,i_util_sch_fix,i_st,i_e] = (maxV + np.log(np.exp(bellman_school-maxV) + np.exp(bellman_work-maxV)))
                                    
                                    # Probability of school being the optimal choice
                                    sol.d[t,i_fix,i_nuw_fix,i_util_sch_fix,i_st,i_e] = 1/(1+np.exp(bellman_work-bellman_school))

                                        

    def bellman_school(self,t, school_time,experience,i_fix,util_sch_fix, nuxi,i_nuw_fix,i_util_sch_fix):
        """ bellman equation for school """
        # Unpack
        par = self.par
        sol = self.sol

        # Flow utility
        utility_school = self.utility_school(school_time, util_sch_fix, nuxi)

        # Expected value
        EV_next_school = 0 
        school_next = school_time + 1
        school_next_index = int(min(school_next-6,14))
        school_time_index = int(school_time-6)
        experience_index = int(experience)

        EV_next_school = sol.V[t+1,i_fix,i_nuw_fix,i_util_sch_fix,school_next_index,experience_index]
        EV_next_interupt = sol.V[t+1,i_fix,i_nuw_fix,i_util_sch_fix,school_time_index,experience_index]

        bellman_school = utility_school + par.beta * (par.zeta * EV_next_interupt + (1-par.zeta) * EV_next_school)
        return bellman_school

    def bellman_work(self,t, school_time, experience, i_fix, nuw,i_nuw_fix,i_util_sch_fix):
        """ bellman equation for work """

        # Unpack
        par = self.par
        sol = self.sol

        # Flow utility
        utility_work = self.utility_work(school_time, experience, nuw)

        experience_next = int(min(experience + 1,17))
        school_next_index = int(school_time-6)

        EV_next_work = sol.V[t+1,i_fix,i_nuw_fix,i_util_sch_fix,school_next_index,experience_next]
  
        bellman_work = utility_work + par.beta* EV_next_work
        return bellman_work

    def utility_school(self,school_time, util_school_fix, nuxi):
        """ utility of attending school """
        # Unpack
        par = self.par
        school_time_util = par.delta7*school_time + par.delta8*school_time**2 
        utility_school = util_school_fix + school_time_util + nuxi
        return utility_school    
    
    def utility_work(self, school_time, experience, nuw):
        """ utility of working """
        wage = self.wage(school_time, experience)
        e = 1/np.exp(np.exp(self.logestar(school_time, experience)))
        return np.log(e*wage) + nuw
    
    def wage(self, school_time, experience):
        return np.exp(self.logwage(school_time, experience))

    def logwage(self, school_time, experience):
        """ log wage """
        # Unpack
        par = self.par
        return par.phi1*school_time + par.phi2*school_time**2 + par.phi3*experience + par.phi4*experience**2
    
    def logestar(self, school_time, experience):
        """" log employment """
        # Unpack
        par = self.par
        return par.kappa1*school_time + par.kappa2*school_time**2+ par.kappa3*experience + par.kappa4*experience**2

    def simulate(self):
        """ simulate model """
        par = self.par
        sol = self.sol
        sim = self.sim

        for i in range(par.simN):

            # Initialize states
            sim.school_time[i,0] = sim.school_time_init[i]
            sim.experience[i,0] = sim.experience_init[i]

            # Utility school and ability work from the chosen values
            sim.util_school_fix[i] = sim.util_school_fix_2[i]
            sim.abil_work_fix[i] = sim.abil_work_fix_2[i]
            
        
            for t in range(par.simT):
                i_fix = int(sim.type[i])

                school_time_index =int(sim.school_time[i,t]-6)
            
                sol_d = sol.d[t,i_fix,:,:,int(min(school_time_index,14)),int(sim.experience[i,t])]

                # If the individual is interrupted
                if sim.interrupt_d[i,t] < par.zeta:
                    sim.interrupt[i,t] = 1
                    sim.school[i,t] = 0 
                    sim.work[i,t] = 0 
                    
                # If not interrupted
                else:
                    sim.school[i,t] = sim.u_d[i,t]< interp_2d(par.nuw_fix_grid,par.util_sch_fix_grid,sol_d,sim.abil_work_fix[i],sim.util_school_fix[i])
                    sim.work[i,t] = 1-sim.school[i,t]
                    sim.interrupt[i,t] = 0
                
                # If the individual works
                if sim.work[i,t] == 1:
                    sim.wage[i,t] = self.wage(school_time_index,int(sim.experience[i,t]))

                # Transition to next period
                if t < par.simT-1:
                    if sim.interrupt[i,t] == 1:
                        sim.experience[i,t+1] = sim.experience[i,t]
                        sim.school_time[i,t+1] = sim.school_time[i,t]
                    
                    elif sim.work[i,t]==1:
                        sim.experience[i,t+1] = sim.experience[i,t] +1
                        sim.school_time[i,t+1] = sim.school_time[i,t]
                    
                    elif sim.school[i,t]==1: 
                        sim.experience[i,t+1] = sim.experience[i,t]
                        sim.school_time[i,t+1] = sim.school_time[i,t] +1

                





