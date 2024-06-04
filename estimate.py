import numpy as np

import scipy.optimize as optimize


import pandas as pd

from consav.linear_interp import interp_3d
from numpy import linalg as la
 
class estimate_class():
 
    def estimate(self,model,family_data,decision_data,pnames,theta0):
        result = optimize.minimize(-np.mean(self.obj), theta0, args=(model, family_data, decision_data, pnames), method='L-BFGS-B', options={'disp': True}, tol= 1e-4)
        self.updatepar(model.par,pnames,result.x)

        cov, se = self.variance(model,pnames,family_data,decision_data,result.x)

        res = {
        'theta': result.x,
        'se':       se,
        't': result.x / se,
        'cov':      cov,
        'success':  result.success, # bool, whether convergence was succesful 
        'nit':      result.nit, # no. algorithm iterations 
        'nfev':     result.nfev, # no. function evaluations 
        'fun':      result.fun # function value at termination 
        }   
        return res


    def obj(self,theta, model, family_data,decision_data,pnames):
        return -self.ll(theta, model, family_data,decision_data,pnames)

    def ll(self,theta, model, family_data,decision_data,pnames,output_exp = False):
        """ log likelihood """
        self.updatepar(model.par,pnames,theta)

        for parname in pnames:
            print(parname, getattr(model.par,parname))


        model.allocate()
        model.solve()

        sol = model.sol
        par = model.par

        idx = family_data.idx


        # Give family-related part of utility and ability. 
        Dad_educ = family_data.Dad_educ
        Mom_educ = family_data.Mom_educ
        Family_income = family_data.Family_income
        Num_siblings = family_data.Num_siblings
        Nuclear = family_data.Nuclear
        Urban = family_data.Urban
        South = family_data.South
     
        util_school  = par.delta0*Dad_educ + par.delta1*Mom_educ + par.delta2*Family_income + par.delta3*Num_siblings + par.delta4*Nuclear + par.delta5*Urban + par.delta6*South
        ability_job = par.gamma0_w*Dad_educ + par.gamma1_w*Mom_educ + par.gamma2_w*Family_income + par.gamma3_w*Num_siblings + par.gamma4_w*Nuclear + par.gamma5_w*Urban + par.gamma6_w*South

        # Other state variables
        experience_going_in = decision_data.exp_going_in 
        school_time_going_in = decision_data.edu_going_in

        # "Decision" (including interruption)
        school = decision_data.dummy_school 
        interrupt = decision_data.dummy_interrup
        work = decision_data.dummy_work


        school_time_index =school_time_going_in-6

        log_likelihood = 0

        epsilon = 1e-10

        probabilities = {0: par.p1, 1: par.p2, 2: par.p3, 3: par.p4} # Probability of being a specific type
        # Iterate over N and T
        for i in range(par.N):
            for t in range(par.T):
                
                # experience going in is a panel data set, where the observations are sorted by time and individual.
                experience_i = experience_going_in[i + par.N * t]
                
                # If experience_i is nan, we go back in time to find a non-nan value. If we do not find a non-nan value, we set experience_i to 0.
                if np.isnan(experience_i):
                    for r in range(t):
                        #If experience_i equal to nan 
                        if np.isnan(experience_i):
                            if output_exp==True:
                                print('experience_i is nan, going ', r+1,' periods back to find a non-nan')
                            experience_i = experience_going_in[i + par.N *(t-r-1)]
                
                if np.isnan(experience_i):
                    if output_exp==True:
                        print('experience_i is still nan')
                    experience_i = 0

                # Finding the solutions for the different types of individuals
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
                    # Securing that the school_optimal_x is within the bounds of 0 and 1
                    clamped_school_optimal_x = np.clip(school_optimal_x, 0+epsilon, 1-epsilon)

                    # Choice probability, incorporating the probability of being interupted etc. 
                    choice_prob_x = par.zeta*interrupt[i+par.N*t] + (1-par.zeta)*(1-clamped_school_optimal_x)*work[i+par.N*t] + (1-par.zeta)*clamped_school_optimal_x*school[i+par.N*t]
                    
                    # Making the "weighted average" of the likelihood taking into account the probability of being a specific type
                    if x in probabilities:
                        log_likelihood += probabilities[x] * np.log(choice_prob_x)
        
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
            idx = family.iloc[:, 0]            # Individual id
            Mom_educ = family.iloc[:, 1]       # Mother's education
            Dad_educ = family.iloc[:, 2]       # Dad's education
            Num_siblings = family.iloc[:, 3]   # Number of siblings 
            Urban = family.iloc[:, 4]          # Urban dummy
            Nuclear = family.iloc[:, 5]        # Nuclear family
            Family_income = family.iloc[:, 6]  # Family income
            South = family.iloc[:, 7]          # South dummy

            # Extract columns from merged data
            idx_merged = merged_data.iloc[:,0]
            year = merged_data.iloc[:,1] 
            edu_going_in = merged_data.iloc[:,3] # Education going into the year
            exp_going_in = merged_data.iloc[:,6] 
            wage = merged_data.iloc[:,7]

            # Dummies
            dummy_school = merged_data.iloc[:,2] # Dummy for going to school 
            dummy_interrup = merged_data.iloc[:,4] # Dummy for being interrupted
            dummy_work = merged_data.iloc[:,5] # Dummy for working

            # Change work to 0, when both work=1 and school=1 
            dummy_work = np.where((dummy_work == 1) & (dummy_school == 1), 0, dummy_work)

            # Collect in a dataframe
            family = {'idx': idx,'Mom_educ':Mom_educ, 'Dad_educ': Dad_educ, 'Num_siblings': Num_siblings, 'Urban': Urban, 'Nuclear': Nuclear, 'Family_income': Family_income, 'South': South}
            dta_family = pd.DataFrame(family) 

            merged = {'id': idx_merged, 'year': year, 'dummy_school': dummy_school, 'edu_going_in': edu_going_in, 'dummy_interrup': dummy_interrup, 'dummy_work': dummy_work, 'exp_going_in': exp_going_in, 'wage': wage}
            dta_merged = pd.DataFrame(merged)
            
            return dta_family, dta_merged
    
    def variance(self,model,pnames,family_data,decision_data, thetahat):
        """Calculates the variance for the likelihood function."""
  
        # Take number of rows in the decision data as the number of observations
        N = decision_data.shape[0]
        #P = thetahat.size

        # numerical Hessian
        f_q = lambda theta: self.obj(theta,model,family_data,decision_data,pnames) # setting the negative of the log-likelihood as the function to differentiate
        H = self.hessian(thetahat,f_q)/N
        H_inv = la.inv(H)  
        
        cov = 1/N * H_inv

        # se: P-vector of std.errs. 
        se = np.sqrt(np.diag(cov))

        return cov, se



    def hessian(self,basis, fhandle, h=1e-4): 
        # Computes the hessian of the input function at the point x0 
        print('basis', basis)

        print('fhandle', fhandle)
        assert basis.ndim == 1 , f'x0 must be 1-dimensional'
        assert callable(fhandle), 'fhandle must be a callable function handle'

        # aggregate rows with a raw sum (as opposed to the mean)
        agg_fun = np.sum

        # Initialization
        K = basis.size
        f2 = np.zeros((K,K)) # double step
        f1 = np.zeros((K,))  # single step
        h_rel = h # optimal step size is smaller than for gradient
                    
        # Step size 
        dh = np.empty((K,))
        for k in range(K): 
            if basis[k] == 0.0: # use absolute step when relative is impossible 
                dh[k] = h_rel 
            else: # use a relative step 
                dh[k] = h_rel*basis[k]

        # Initial point 
        f0 = agg_fun(fhandle(basis)) 

        # Evaluate single forward steps
        for k in range(K): 
            x1 = np.copy(basis) 
            x1[k] = basis[k] + dh[k] 
            f1[k] = agg_fun(fhandle(x1))

        # Double forward steps
        for k in range(K): 
            for j in range(k+1): # only loop to the diagonal!! This is imposing symmetry to save computations
                
                # 1. find the new point (after double-stepping) 
                x2 = np.copy(basis) 
                if k==j: # diagonal steps: only k'th entry is changed, taking two steps 
                    x2[k] = basis[k] + dh[k] + dh[k] 
                else:  # we have taken both a step in the k'th and one in the j'th directions 
                    x2[k] = basis[k] + dh[k] 
                    x2[j] = basis[j] + dh[j]  

                # 2. compute function value 
                f2[k,j] = agg_fun(fhandle(x2))
                
                # 3. fill out above the diagonal ()
                if j < k: # impose symmetry  
                    f2[j,k] = f2[k,j]

        hess = np.empty((K,K))
        for k in range(K): 
            for j in range(K): 
                hess[k,j] = ((f2[k,j] - f1[k]) - (f1[j] - f0)) / (dh[k] * dh[j])
        
        # Regularization to ensure positive definiteness
        #hess += 1e-4 * np.eye(K)
        return hess