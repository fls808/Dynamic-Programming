import numpy as np
from scipy.optimize import minimize

from EconModel import EconModelClass, jit

import scipy.optimize as optimize

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


def estimate():

    pnames = ['delta7']

    res = optimize.minimize(ll,theta0,args=(model, data, pnames),method='Nelder-Mead',options={'disp':True})
    model = updatepar(model,pnames,res.x)

def ll(theta, model, data,pnames):

    model = updatepar(model,pnames,theta)
   
    model.setup()
    model.solve()

    lik_pr =  



def updatepar(par,parnames, parvals):
    """ Update parameters """
    for i,parname in enumerate(parnames):
        parval = parvals[i]
        setattr(par,parname,parval)
    return par

def read_data_old(self): 

        # Load data into a DataFrame
        family = pd.read_csv("famb.data", delimiter=",")
        exp = pd.read_csv('exp.data', delim_whitespace=True, header=None, index_col=0)
        edu = pd.read_csv('ed.data', delim_whitespace=True, header=None, index_col=0)
        wage = pd.read_csv('wage.data', delim_whitespace=True, header=None, index_col=0)

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
        # Calculate the dummy variable
        dummy = (family.diff(axis=1).iloc[:, 1:] == 0).astype(int)

        new_column_names = [str(year) for year in range(1979, 1991)]

        # Education data 
        edu.index.name = 'id'
        edu.columns = new_column_names
        edu = edu.replace(-9999, np.nan)

        # Experience data
        exp.columns = new_column_names
        exp = exp.replace(-9999, np.nan)

        # Wage data
        wage.columns = new_column_names
        wage = wage.replace(-9999, np.nan)

        # Collect in a dataframe
        remove_first_row_index=idx-np.append(0,idx[:-1])
        family = {'id': idx,'Mother education':Mom_educ, 'Father education': Dad_educ, 'Number of siblings': Num_siblings, 'Urban': Urban, 'Nuclear': Nuclear, 'Family_income': Family_income, 'South': South, 'AFQT_score': AFQT_score, 'Dummy': dummy,  'boolean': remove_first_row_index}
        df = pd.DataFrame(family) 

        # Remove observations with dummy == 0 in all years
        df = df.drop(df[df.boolean!=0].index)

        # save data
        dta = df.drop(['id','boolean'],axis=1)
        
        return dta

def read_data(self): 

        # Load data into a DataFrame
        family = pd.read_csv("famb.data", delimiter=",")
        merged_data = pd.read_csv('merged_data.data', delim_whitespace=True, header=None, index_col=0)
        

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
        dummy = merged_data[:,2]
        edu = merged_data[:,3]
        exp = merged_data[:,4]
        wage = merged_data[:,5]

        # Collect in a dataframe
        remove_first_row_index=idx-np.append(0,idx[:-1])
        family = {'id': idx,'Mother education':Mom_educ, 'Father education': Dad_educ, 'Number of siblings': Num_siblings, 'Urban': Urban, 'Nuclear': Nuclear, 'Family_income': Family_income, 'South': South, 'AFQT_score': AFQT_score, 'Dummy': dummy,  'boolean': remove_first_row_index}
        df = pd.DataFrame(family) 

        # Remove observations with dummy == 0 in all years
        df = df.drop(df[df.boolean!=0].index)

        # save data
        dta = df.drop(['id','boolean'],axis=1)
        
        return dta