import numpy as np

import scipy.optimize as optimize


import pandas as pd

from consav.linear_interp import interp_3d
 
class estimate_class():

    def estimate(self,model,family_data,decision_data,pnames,theta0):
        res = optimize.minimize(self.obj,theta0,args=(model, family_data,decision_data, pnames),options={'disp':True})
        self.updatepar(model.par,pnames,res.x)

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


        # Give ability
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

        
        # Family related part of utility and ability. 
        util_school  = par.delta0*Dad_educ + par.delta1*Mom_educ + par.delta2*Family_income + par.delta3*Num_siblings + par.delta4*Nuclear + par.delta5*Urban + par.delta6*South
        ability_job = par.gamma0_w*Dad_educ + par.gamma1_w*Mom_educ + par.gamma2_w*Family_income + par.gamma3_w*Num_siblings + par.gamma4_w*Nuclear + par.gamma5_w*Urban + par.gamma6_w*South
        

        school_time_index =school_time_going_in-6

        log_likelihood = 0

        epsilon = 1e-10

        probabilities = {0: par.p1, 1: par.p2, 2: par.p3, 3: par.p4} # Probability of being a specific type
        # Iterate over N and T
        for i in range(par.N):
            for t in range(par.T):
                
                experience_i = experience_going_in[i + par.N * t]
                
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
            dummy_school = merged_data.iloc[:,2]
            edu_going_in = merged_data.iloc[:,3]
            dummy_interrup = merged_data.iloc[:,4]
            dummy_work = merged_data.iloc[:,5]

            # Change work to 0, when both work=1 and school=1 
            dummy_work = np.where((dummy_work == 1) & (dummy_school == 1), 0, dummy_work)

            exp_going_in = merged_data.iloc[:,6]
            wage = merged_data.iloc[:,7]

            # Collect in a dataframe
            family = {'idx': idx,'Mom_educ':Mom_educ, 'Dad_educ': Dad_educ, 'Num_siblings': Num_siblings, 'Urban': Urban, 'Nuclear': Nuclear, 'Family_income': Family_income, 'South': South}
            dta_family = pd.DataFrame(family) 

            merged = {'id': idx_merged, 'year': year, 'dummy_school': dummy_school, 'edu_going_in': edu_going_in, 'dummy_interrup': dummy_interrup, 'dummy_work': dummy_work, 'exp_going_in': exp_going_in, 'wage': wage}
            dta_merged = pd.DataFrame(merged)
            
            return dta_family, dta_merged
    
    def variance(self,model,pnames,family_data,decision_data, result):
        """Calculates the variance for the likelihood function."""
  
        # Take number of rows in the decision data as the number of observations
        N = decision_data.shape[0]
        thetahat = result.x
        P = thetahat.size

        # numerical Hessian
        f_q = lambda theta : self.ll(theta,model,family_data,decision_data,pnames) 
        H = self.hessian(f_q, thetahat)/N
        H_inv = la.inv(H)  
        
        cov = 1/N * H_inv

        # se: P-vector of std.errs. 
        se = np.sqrt(np.diag(cov))

        return cov, se

    
    def hessian( fhandle , x0 , h=1e-5 ) -> np.ndarray: 
        '''hessian(): computes the (K,K) matrix of 2nd partial derivatives
            using the aggregation "sum" (i.e. consider dividing by N)
        Returns: 
            hess: (K,K) matrix of second partial derivatives 
        '''

        # Computes the hessian of the input function at the point x0 
        assert x0.ndim == 1 , f'x0 must be 1-dimensional'
        assert callable(fhandle), 'fhandle must be a callable function handle'

        # aggregate rows with a raw sum (as opposed to the mean)
        agg_fun = np.sum

        # Initialization
        K = x0.size
        f2 = np.zeros((K,K)) # double step
        f1 = np.zeros((K,))  # single step
        h_rel = h # optimal step size is smaller than for gradient
                    
        # Step size 
        dh = np.empty((K,))
        for k in range(K): 
            if x0[k] == 0.0: # use absolute step when relative is impossible 
                dh[k] = h_rel 
            else: # use a relative step 
                dh[k] = h_rel*x0[k]

        # Initial point 
        f0 = agg_fun(fhandle(x0)) 

        # Evaluate single forward steps
        for k in range(K): 
            x1 = np.copy(x0) 
            x1[k] = x0[k] + dh[k] 
            f1[k] = agg_fun(fhandle(x1))

        # Double forward steps
        for k in range(K): 
            for j in range(k+1): # only loop to the diagonal!! This is imposing symmetry to save computations
                
                # 1. find the new point (after double-stepping) 
                x2 = np.copy(x0) 
                if k==j: # diagonal steps: only k'th entry is changed, taking two steps 
                    x2[k] = x0[k] + dh[k] + dh[k] 
                else:  # we have taken both a step in the k'th and one in the j'th directions 
                    x2[k] = x0[k] + dh[k] 
                    x2[j] = x0[j] + dh[j]  

                # 2. compute function value 
                f2[k,j] = agg_fun(fhandle(x2))
                
                # 3. fill out above the diagonal ()
                if j < k: # impose symmetry  
                    f2[j,k] = f2[k,j]

        hess = np.empty((K,K))
        for k in range(K): 
            for j in range(K): 
                hess[k,j] = ((f2[k,j] - f1[k]) - (f1[j] - f0)) / (dh[k] * dh[j])

        return hess