import numpy as np
from scipy.optimize import minimize

from EconModel import EconModelClass, jit

import scipy.optimize as optimize


import pandas as pd

from consav.grids import nonlinspace
from consav.linear_interp import interp_1d
from consav.linear_interp import interp_2d
from consav.linear_interp import interp_3d
from consav.linear_interp import interp_4d
from consav.quadrature import normal_gauss_hermite

#s 
class estimate_class():

    def estimate(self,model,family_data,decision_data,pnames,theta0):

        res = optimize.minimize(self.obj,theta0,args=(model, family_data,decision_data, pnames),options={'disp':True})
        self.updatepar(model.par,pnames,res.x)

    # Vi vil gerne have indkomporeret løn i denne model. 
    def obj(self,theta, model, family_data,decision_data,pnames):
        return -self.ll(theta, model, family_data,decision_data,pnames)

    def ll(self,theta, model, family_data,decision_data,pnames,output_exp = False):

        self.updatepar(model.par,pnames,theta)

        for parname in pnames:
            print(parname, getattr(model.par,parname))


        model.allocate()
        model.solve()
        print('model solved')

        sol = model.sol
        par = model.par

        idx = family_data.idx


        # give ability
        Dad_educ = family_data.Dad_educ
        Mom_educ = family_data.Mom_educ
        Family_income = family_data.Family_income
        Num_siblings = family_data.Num_siblings
        Nuclear = family_data.Nuclear
        Urban = family_data.Urban
        South = family_data.South


        experience_going_in = decision_data.exp_going_in 
        school_time_going_in = decision_data.edu_going_in


        school = decision_data.dummy_school 
        interrupt = decision_data.dummy_interrup
        work = decision_data.dummy_work

        
        # family related part of utility and ability. 
        util_school  = par.delta0*Dad_educ + par.delta1*Mom_educ + par.delta2*Family_income + par.delta3*Num_siblings + par.delta4*Nuclear + par.delta5*Urban + par.delta6*South
        ability_job = par.gamma0_w*Dad_educ + par.gamma1_w*Mom_educ + par.gamma2_w*Family_income + par.gamma3_w*Num_siblings + par.gamma4_w*Nuclear + par.gamma5_w*Urban + par.gamma6_w*South
        

        school_time_index =school_time_going_in-6

        log_likelihood = 0

        epsilon = 1e-10

        probabilities = {0: par.p1, 1: par.p2, 2: par.p3, 3: par.p4}
        # Iterate over N and T
        for i in range(par.N):
            for t in range(par.T):
                
                experience_i = experience_going_in[i + par.N * t]
                
                if np.isnan(experience_i):
                    for r in range(t):
                        #if experience_i equal to nan 
                        if np.isnan(experience_i):
                            if output_exp==True:
                                print('experience_i is nan, going ', r+1,' periods back to find a non-nan')
                            experience_i = experience_going_in[i + par.N *(t-r-1)]
                
                if np.isnan(experience_i):
                    if output_exp==True:
                        print('experience_i is still nan')
                    experience_i = 0

                for x in range(1, 4):
                    sol_x = sol.d[t, x, :, :, school_time_index[i + par.N * t],:]
                    school_optimal_x = interp_3d(
                        par.nuw_fix_grid,
                        par.util_sch_fix_grid,
                        par.experience_grid,
                        sol_x,
                        ability_job[i],
                        util_school[i],
                        experience_i
                    )
                    clamped_school_optimal_x = np.clip(school_optimal_x, 0+epsilon, 1-epsilon)

                    # Incorporate the probability of being interupted etc. 
                    choice_prob_x = par.zeta*interrupt[i+par.N*t] + (1-par.zeta)*(1-clamped_school_optimal_x)*work[i+par.N*t] + (1-par.zeta)*clamped_school_optimal_x*school[i+par.N*t]

                    if choice_prob_x == 0: 
                        print('idx = ', idx[i], 't =', t)
                        print('choice_prob_x = 0', 'school t-1=' , school_time_going_in[i + par.N * t], 'interrupt t-1 =', interrupt[i + par.N * (t-1)], 'work t-1 =', work[i + par.N * (t-1)])
                        print('school t =', school[i + par.N * t], 'interrupt t =', interrupt[i + par.N * t], 'work t =', work[i + par.N * t])
                    
                    if x in probabilities:
                        log_likelihood += probabilities[x] * np.log(choice_prob_x)
        
        print('log_likelihood =', log_likelihood)
        return log_likelihood
        

    def updatepar(self,par,parnames, parvals):
        """ Update parameters """
        for i,parname in enumerate(parnames):
            parval = parvals[i]
            setattr(par,parname,parval)
        return par

    def read_data(self): 

            # Load data into a DataFrame
            family = pd.read_csv("famb.data", delim_whitespace=True, header=None)
            merged_data = pd.read_csv('merged_data.data', sep='\t')

            # Extract columns
            idx = family.iloc[:, 0]            # individual id
            Mom_educ = family.iloc[:, 1]       # Mother's education
            Dad_educ = family.iloc[:, 2]       # Dad's education
            Num_siblings = family.iloc[:, 3]
            Urban = family.iloc[:, 4]
            Nuclear = family.iloc[:, 5] 
            Family_income = family.iloc[:, 6]
            South = family.iloc[:, 7]
            AFQT_score = family.iloc[:, 8]

            # Extract columns from merged data
            idx_merged = merged_data.iloc[:,0]
            year = merged_data.iloc[:,1] 
            dummy_school = merged_data.iloc[:,2]
            edu_going_in = merged_data.iloc[:,3]
            dummy_interrup = merged_data.iloc[:,4]
            dummy_work = merged_data.iloc[:,5]
            # Change work to 0, when both work=1 and school=1 
            dummy_work = np.where((dummy_work == 1) & (dummy_school == 1), 0, dummy_work)

            
            exp_going_in = merged_data.iloc[:,6]
            wage = merged_data.iloc[:,7]

            # Collect in a dataframe
            #remove_first_row_index=idx-np.append(0,idx[:-1])
            family = {'idx': idx,'Mom_educ':Mom_educ, 'Dad_educ': Dad_educ, 'Num_siblings': Num_siblings, 'Urban': Urban, 'Nuclear': Nuclear, 'Family_income': Family_income, 'South': South, 'AFQT_score': AFQT_score}
            dta_family = pd.DataFrame(family) 

            merged = {'id': idx_merged, 'year': year, 'dummy_school': dummy_school, 'edu_going_in': edu_going_in, 'dummy_interrup': dummy_interrup, 'dummy_work': dummy_work, 'exp_going_in': exp_going_in, 'wage': wage}
            dta_merged = pd.DataFrame(merged)
            # Remove observations with dummy == 0 in all years
            # df = df.drop(df[df.boolean!=0].index)

            # save data
            #dta = df.drop(['id','boolean'],axis=1)
            
            return dta_family, dta_merged