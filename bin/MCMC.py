import numpy as np
import pandas as pd
from scipy.integrate import odeint
import scipy.optimize as op
from scipy.optimize import Bounds
import emcee
import matplotlib.pyplot as plt
import sys

class Obs_data:
    def __init__(self):
        self.obs_time = np.array([0.0,15.0,20.0,35.0,40.0],dtype='float64')
        self.DsRedBreastMean = np.array([2e6,7472538.433,13812038.59,28917066.91,27536532.68],dtype='float64')
        self.DsRedBreastStd = np.array([0.0,2741175.349,3754008.889,17941627.8,9029738.531],dtype='float64')
        self.GFPBreastMean = np.array([0.0,1674828.135,3651197.537,12024145.7,25267190.8],dtype='float64')
        self.GFPBreastStd = np.array([0.0,627791.9714,1260460.54,3426210.465,11367021.6],dtype='float64')

        self.DsRedBloodStMean = np.array([0.0,11000.0],dtype='float64')
        self.DsRedBloodStStd = np.array([0.0,10000.0],dtype='float64')
        self.GFPBloodStMean = np.array([0.0,25000.0],dtype='float64')
        self.GFPBloodStStd = np.array([0.0,10000.0],dtype='float64')

        self.DsRedLungMean = np.array([0.0, 11965.48722, 57514.73141, 2438531.594, 5534042.826],dtype='float64')
        self.DsRedLungStd = np.array([0.0,3170.335905, 85294.21074, 2294754.963, 3649237.296],dtype='float64')
        self.GFPLungMean = np.array([0.0, 60413.64349, 92990.94792, 2372466.215, 4465957.174],dtype='float64')
        self.GFPLungStd = np.array([0.0,70574.67943, 186123.8559, 2829820.057, 2253428.623],dtype='float64')

obs_data = Obs_data()
smooth_time = np.arange(0,45,0.1)
# Breast
parameters_breast_labels = ['alpha_br','alpha_bg','K_b','lamb','beta_r','beta_g','tau']
parameters_breast_units = ['1/day','1/day','1/day','1/day','1/day','1/day','day']
parameters_breast_descs = ['growth rate of red cells on breast','growth rate of green cells on breast', 'carrying capacity of cells on breast','hypoxia rate','intravasation rate of red cells','intravasation rate of green cells', 'meantime for cells to start intravasation']
parameters_breast_bounds_lb = [np.log(2)/10,np.log(2)/10,1e8,0,0.004,0.004,8]
parameters_breast_bounds_ub = [np.log(2)/2,np.log(2)/2,1e9,1.0,0.04,0.04,12]
parameters_breast_df = pd.DataFrame(data={'Parameter': parameters_breast_labels, 'Unit': parameters_breast_units, 'Description': parameters_breast_descs, 'Lower bound': parameters_breast_bounds_lb, 'Upper bound': parameters_breast_bounds_ub})
constraints_breast = [ ['K_b = inf','tau = 0'], ['K_b = inf','tau = 0','alpha_bg = alpha_br'], ['K_b = inf','tau = 0','beta_g = beta_r'], ['K_b = inf','tau = 0','alpha_bg = alpha_br','beta_g = beta_r'], ['tau = 0'], ['tau = 0','alpha_bg = alpha_br'], ['tau = 0','beta_g = beta_r'], ['tau = 0', 'alpha_bg = alpha_br','beta_g = beta_r'], ['K_b = inf'], ['K_b = inf','alpha_bg = alpha_br'], ['K_b = inf','beta_g = beta_r'], ['K_b = inf','alpha_bg = alpha_br','beta_g = beta_r'], [], ['alpha_bg = alpha_br'], ['beta_g = beta_r'], ['alpha_bg = alpha_br', 'beta_g = beta_r'], ['K_b = inf','tau = 0','beta_g = 1.5*beta_r'], ['K_b = inf','tau = 0','alpha_bg = alpha_br','beta_g = 1.5*beta_r'], ['tau = 0','beta_g = 1.5*beta_r'], ['tau = 0', 'alpha_bg = alpha_br','beta_g = 1.5*beta_r'], ['K_b = inf','beta_g = 1.5*beta_r'], ['K_b = inf','alpha_bg = alpha_br','beta_g = 1.5*beta_r'], ['beta_g = 1.5*beta_r'], ['alpha_bg = alpha_br', 'beta_g = 1.5*beta_r'] ]
# Bloodstream and Lung
parameters_mets_labels = ['alpha_lr','alpha_lg','K_l','phi_r','phi_g','f_Er','f_Eg']
parameters_mets_units = ['1/day','1/day','1/day','1/day','1/day','None','None']
parameters_mets_descs = ['growth rate of red cells on lung','growth rate of green cells on lung','carrying capacity of cells on lung', 'clearance rate of red cells on blood stream','clearance rate of green cells on blood stream','extravasating fraction of red cells','extravasating fraction of green cells']
parameters_mets_bounds_lb = [np.log(2)/10,np.log(2)/10,1e8,1.0,1.0,0.0005,0.0005]
parameters_mets_bounds_ub = [np.log(2)/2,np.log(2)/2,1e9,500,500,0.1,0.1]
parameters_mets_df = pd.DataFrame(data={'Parameter': parameters_mets_labels, 'Unit': parameters_mets_units, 'Description': parameters_mets_descs, 'Lower bound': parameters_mets_bounds_lb, 'Upper bound': parameters_mets_bounds_ub})
constraints_mets = [ ['K_l = inf'], ['K_l = inf', 'alpha_lg = alpha_lr'], ['K_l = inf', 'phi_r = phi_g'], ['K_l = inf', 'f_Eg = 1.0*phi_r/phi_g*f_Er'], ['K_l = inf', 'alpha_lg = alpha_lr', 'phi_r = phi_g'], ['K_l = inf', 'alpha_lg = alpha_lr', 'f_Eg = 1.0*phi_r/phi_g*f_Er'], ['K_l = inf', 'phi_r = phi_g', 'f_Eg = 1.0*phi_r/phi_g*f_Er'], ['K_l = inf', 'alpha_lg = alpha_lr', 'phi_r = phi_g', 'f_Eg = 1.0*phi_r/phi_g*f_Er'], [], ['alpha_lg = alpha_lr'], ['phi_r = phi_g'], ['f_Eg = 1.0*phi_r/phi_g*f_Er'], ['alpha_lg = alpha_lr', 'phi_r = phi_g'], ['alpha_lg = alpha_lr', 'f_Eg = 1.0*phi_r/phi_g*f_Er'], ['phi_r = phi_g', 'f_Eg = 1.0*phi_r/phi_g*f_Er'], ['alpha_lg = alpha_lr', 'phi_r = phi_g', 'f_Eg = 1.0*phi_r/phi_g*f_Er'], ['K_l = inf', 'phi_r = 3*phi_g'], ['K_l = inf', 'alpha_lg = alpha_lr', 'phi_r = 3*phi_g'], ['K_l = inf', 'phi_r = 3*phi_g', 'f_Eg = 1.0*phi_r/phi_g*f_Er'], ['K_l = inf', 'alpha_lg = alpha_lr', 'phi_r = 3*phi_g', 'f_Eg = 1.0*phi_r/phi_g*f_Er'], ['phi_r = 3*phi_g'], ['alpha_lg = alpha_lr', 'phi_r = 3*phi_g'], ['phi_r = 3*phi_g', 'f_Eg = 1.0*phi_r/phi_g*f_Er'], ['alpha_lg = alpha_lr', 'phi_r = 3*phi_g', 'f_Eg = 1.0*phi_r/phi_g*f_Er'], ['K_l = inf', 'phi_r = 6*phi_g'], ['K_l = inf', 'alpha_lg = alpha_lr', 'phi_r = 6*phi_g'], ['K_l = inf', 'phi_r = 6*phi_g', 'f_Eg = 1.0*phi_r/phi_g*f_Er'], ['K_l = inf', 'alpha_lg = alpha_lr', 'phi_r = 6*phi_g', 'f_Eg = 1.0*phi_r/phi_g*f_Er'], ['phi_r = 6*phi_g'], ['alpha_lg = alpha_lr', 'phi_r = 6*phi_g'], ['phi_r = 6*phi_g', 'f_Eg = 1.0*phi_r/phi_g*f_Er'], ['alpha_lg = alpha_lr', 'phi_r = 6*phi_g', 'f_Eg = 1.0*phi_r/phi_g*f_Er'], ['K_l = inf', 'f_Eg = 1.5*phi_r/phi_g*f_Er'], ['K_l = inf', 'alpha_lg = alpha_lr', 'f_Eg = 1.5*phi_r/phi_g*f_Er'], ['K_l = inf', 'phi_r = phi_g', 'f_Eg = 1.5*phi_r/phi_g*f_Er'], ['K_l = inf', 'alpha_lg = alpha_lr', 'phi_r = phi_g', 'f_Eg = 1.5*phi_r/phi_g*f_Er'], ['f_Eg = 1.5*phi_r/phi_g*f_Er'], ['alpha_lg = alpha_lr', 'f_Eg = 1.5*phi_r/phi_g*f_Er'], ['phi_r = phi_g', 'f_Eg = 1.5*phi_r/phi_g*f_Er'], ['alpha_lg = alpha_lr', 'phi_r = phi_g', 'f_Eg = 1.5*phi_r/phi_g*f_Er'], ['K_l = inf', 'phi_r = 3*phi_g', 'f_Eg = 1.5*phi_r/phi_g*f_Er'], ['K_l = inf', 'alpha_lg = alpha_lr', 'phi_r = 3*phi_g', 'f_Eg = 1.5*phi_r/phi_g*f_Er'], ['phi_r = 3*phi_g', 'f_Eg = 1.5*phi_r/phi_g*f_Er'], ['alpha_lg = alpha_lr', 'phi_r = 3*phi_g', 'f_Eg = 1.5*phi_r/phi_g*f_Er'], ['K_l = inf', 'phi_r = 6*phi_g', 'f_Eg = 1.5*phi_r/phi_g*f_Er'], ['K_l = inf', 'alpha_lg = alpha_lr', 'phi_r = 6*phi_g', 'f_Eg = 1.5*phi_r/phi_g*f_Er'], ['phi_r = 6*phi_g', 'f_Eg = 1.5*phi_r/phi_g*f_Er'], ['alpha_lg = alpha_lr', 'phi_r = 6*phi_g', 'f_Eg = 1.5*phi_r/phi_g*f_Er'] ]

def Apply_Constraints(constrain, ParameterDataFrame, ValueParConst = None):
    indx_remain = ParameterDataFrame.index.tolist()
    indx_delete = []
    ValueParComplete = np.zeros(len(indx_remain),dtype='float64')
    ConstraintSliced = []
    for const in constrain:
        Pars = const.split() # remove spaces
        Pars = "".join(Pars) # join with no spaces
        Pars = Pars.split('=') # both sides of constrain
        # Checking error
        if (len(Pars) != 2):
            print(f"Error reading the constraint: {const}!")
            exit(1)
        LeftSide = Pars[0] # name of parameter (left side of constrain)
        RightSide = Pars[1].split("*") # read and split if have a multiplication by scalar (right side of constraint)
        indx_leftside = ParameterDataFrame.index[ParameterDataFrame["Parameter"]==LeftSide].tolist() # will remove the parameter on left side of constraint
        # Checking error
        if (len(indx_leftside) != 1):
            print(f"Error reading the constraint: {const}! Parameter: {LeftSide} not found!")
            exit(1)
        indx_delete = indx_delete + indx_leftside
        # Checking error
        if (len(RightSide) < 1 or len(RightSide) > 3):
            print(f"Error reading the constraint: {const}! Constraint must be [ParameterName1 = scalar] or [ParameterName1 = ParameterName2] or [ParameterName1 = scalar * ParameterName2] or [ParameterName1 = scalar * AddPar1/AddPar2 * ParameterName2]")
            exit(1)
        # Manipulating right side
        if (len(RightSide) == 1): # no multiplication
            indx_rightSide = ParameterDataFrame.index[ParameterDataFrame["Parameter"]==RightSide[0]].tolist() # indx to apply constraint
            if (len(indx_rightSide) == 1): TempList = indx_leftside+indx_rightSide+[1.0] # [ParameterName1 = ParameterName2]
            else:
                if (RightSide[0] == 'inf'): TempList = indx_leftside+[np.inf] # [ParameterName1 = inf]
                else: TempList = indx_leftside+[float(RightSide[0])] # [ParameterName1 = scalar]
            ConstraintSliced.append(TempList)
        if (len(RightSide) == 2): # with multiplication [ParameterName1 = scalar * ParameterName2]
            indx_rightSide = ParameterDataFrame.index[ParameterDataFrame["Parameter"]==RightSide[1]].tolist() # indx to apply constraint
            TempList = indx_leftside+indx_rightSide+[float(RightSide[0])]
            ConstraintSliced.append(TempList)
        if (len(RightSide) == 3): # with multiplication [ParameterName1 = scalar * AddPar1/AddPar2 * ParameterName2]
            indx_rightSide = ParameterDataFrame.index[ParameterDataFrame["Parameter"]==RightSide[2]].tolist() # indx to apply constraint
            Add_rightSide = RightSide[1].split("/")
            indx_rightSideAdd1 = ParameterDataFrame.index[ParameterDataFrame["Parameter"]==Add_rightSide[0]].tolist() # indx to apply constraint (add parameter1)
            indx_rightSideAdd2 = ParameterDataFrame.index[ParameterDataFrame["Parameter"]==Add_rightSide[1]].tolist() # indx to apply constraint (add parameter1)
            TempList = indx_leftside+indx_rightSide+[float(RightSide[0])]+indx_rightSideAdd1+indx_rightSideAdd2
            ConstraintSliced.append(TempList)
    indx_remain = list(set(indx_remain) - set(indx_delete)) # remove constrain indexes from remain indexes
    if ( ValueParConst is None ):
        return indx_remain
    else:
        # Mapping parameter to complete vector
        ValueParComplete[indx_remain] = ValueParConst
        # Applying constraint
        for const in ConstraintSliced:
            if( len(const) == 2 ):
                ValueParComplete[const[0]] = const[1] # [ParameterName1 = scalar]
            elif ( len(const) == 3 ):
                ValueParComplete[const[0]] = ValueParComplete[const[1]]*const[2] # [ParameterName1 = ParameterName2] and [ParameterName1 = scalar * ParameterName2]
            elif ( len(const) == 5 ):
                ValueParComplete[const[0]] = ValueParComplete[const[1]]*const[2]*(ValueParComplete[const[3]]/ValueParComplete[const[4]]) # [ParameterName1 = scalar * AddPar1/AddPar2 * ParameterName2]
            else:
                print(f"Error: Sliced constraint {const} is not readable")
                exit(1)

        return ValueParComplete

def model_breast(y, t, par):
    alpha_br,alpha_bg,K_b,lamb,beta_r,beta_g,tau = par
    # after tau days starts intravasation
    if (t < tau):
        H = 0
    else:
        H = 1
    B_r = alpha_br*y[0]*(1.0 - (y[0]/(K_b - y[1])) ) - lamb*y[0] - H*beta_r*y[0]
    B_g = alpha_bg*y[1]*(1.0 - (y[1]/(K_b - y[0])) ) + lamb*y[0] - H*beta_g*y[1]

    return [B_r,B_g]

def model_mets(t, y, par, par_breast):
    alpha_br,alpha_bg,K_b,lamb,beta_r,beta_g,tau = par_breast
    alpha_lr,alpha_lg,K_l,phi_r,phi_g,f_Er,f_Eg = par
    # after tau days starts intravasation
    if (t < tau):
        H = 0
    else:
        H = 1
    B_r = alpha_br*y[0]*(1.0 - (y[0]/(K_b - y[1])) ) - lamb*y[0] - H*beta_r*y[0]
    B_g = alpha_bg*y[1]*(1.0 - (y[1]/(K_b - y[0])) ) + lamb*y[0] - H*beta_g*y[1]
    BS_r = H*beta_r*y[0] - phi_r*y[2]
    BS_g = H*beta_g*y[1] - phi_g*y[3]
    L_r = alpha_lr*y[4]*(1.0 - (y[4]/(K_l - y[5])) ) + f_Er*phi_r*y[2]
    L_g = alpha_lg*y[5]*(1.0 - (y[5]/(K_l - y[4])) ) + f_Eg*phi_g*y[3]
    return [B_r,B_g,BS_r,BS_g,L_r,L_g]

def forward_model_breast(times,theta, ID_model):
    initial_cond = np.array([obs_data.DsRedBreastMean[0],obs_data.GFPBreastMean[0]],dtype='float64')
    parameter = Apply_Constraints(constraints_breast[ID_model-1],parameters_breast_df,theta)
    output = odeint(model_breast, t=times, y0=initial_cond,args=tuple([parameter]))
    return output

def forward_model_mets(times,theta, ID_model, theta_breast, modelID_breast):
    initial_cond = np.array([obs_data.DsRedBreastMean[0],obs_data.GFPBreastMean[0],obs_data.DsRedBloodStMean[0],obs_data.GFPBloodStMean[0],obs_data.DsRedLungMean[0],obs_data.GFPLungMean[0]],dtype='float64')
    parameters_breast = Apply_Constraints(constraints_breast[modelID_breast-1],parameters_breast_df,theta_breast)
    parameter = Apply_Constraints(constraints_mets[ID_model-1],parameters_mets_df,theta)
    output = odeint(model_mets, t=times, y0=initial_cond,args=tuple([parameter,parameters_breast]),tfirst=True)
    return output

def log_likelihood_breast(theta, ID_model):
    model = forward_model_breast(obs_data.obs_time,theta, ID_model)
    log_BreastRed = np.sum( np.log(2*np.pi)+np.log(obs_data.DsRedBreastStd[1:]**2) + ((obs_data.DsRedBreastMean[1:] - model[1:,0]) / obs_data.DsRedBreastStd[1:])**2 )
    log_BreastGreen = np.sum( np.log(2*np.pi)+np.log(obs_data.GFPBreastStd[1:]**2) + ((obs_data.GFPBreastMean[1:] - model[1:,1]) /  obs_data.GFPBreastStd[1:])**2 )
    return -0.5*(log_BreastRed+log_BreastGreen)

def log_likelihood_mets(theta, ID_model, theta_breast, modelID_breast):
    model = forward_model_mets(obs_data.obs_time,theta, ID_model, theta_breast, modelID_breast)
    log_BloodStreamRed = np.sum( np.log(2*np.pi)+np.log(obs_data.DsRedBloodStStd[1:]**2) + ((obs_data.DsRedBloodStMean[-1] - model[-1,2]) / obs_data.DsRedBloodStStd[-1])**2 )
    log_BloodStreamGreen = np.sum( np.log(2*np.pi)+np.log(obs_data.GFPBloodStStd[-1]**2) + ((obs_data.GFPBloodStMean[-1] - model[-1,3]) /  obs_data.GFPBloodStStd[-1])**2 )
    log_LungRed = np.sum( np.log(2*np.pi)+np.log(obs_data.DsRedLungStd[1:]**2) + ((obs_data.DsRedLungMean[1:] - model[1:,4]) / obs_data.DsRedLungStd[1:])**2 )
    log_LungGreen = np.sum( np.log(2*np.pi)+np.log(obs_data.GFPLungStd[1:]**2) + ((obs_data.GFPLungMean[1:] - model[1:,5]) /  obs_data.GFPLungStd[1:])**2 )
    return -0.5*(log_BloodStreamRed+log_BloodStreamGreen+log_LungRed+log_LungGreen)

def RSS_breast(theta, ID_model): #Residual Sum of Squares
    model = forward_model_breast(obs_data.obs_time,theta, ID_model)
    x_obs = np.concatenate((obs_data.DsRedBreastMean[1:],obs_data.GFPBreastMean[1:]),axis=None)
    x_model = np.concatenate((model[1:,0],model[1:,1]),axis=None)
    return np.sum((x_model-x_obs)**2)

def RSS_mets(theta, ID_model, theta_breast, modelID_breast): #Residual Sum of Squares
    model = forward_model_mets(obs_data.obs_time,theta,ID_model,theta_breast,modelID_breast)
    x_obs = np.concatenate((obs_data.DsRedBreastMean[1:],obs_data.GFPBreastMean[1:],obs_data.DsRedBloodStMean[-1],obs_data.GFPBloodStMean[-1],obs_data.DsRedLungMean[1:],obs_data.GFPLungMean[1:]),axis=None)
    x_model = np.concatenate((model[1:,0],model[1:,1],model[-1,2],model[-1,3],model[1:,4],model[1:,5]),axis=None)
    return np.sum((x_model-x_obs)**2)

def log_prior(theta, lowerBounds, upperBounds): # log of prior distribution [uniform distribution]
    if ( (theta < lowerBounds).any() | (theta > upperBounds).any() ):
        return -np.inf
    else: return 0.0

def log_probability(theta, lowerBounds, upperBounds, ID_model_breast, ID_model_mets = None, theta_breast = None):
    logPrior = log_prior(theta, lowerBounds, upperBounds)
    if not np.isfinite(logPrior): return -np.inf
    else:
        if not (ID_model_mets): return logPrior + log_likelihood_breast(theta, ID_model_breast)
        else: return logPrior + log_likelihood_mets(theta, ID_model_mets, theta_breast, ID_model_breast)

def MaxLikEst_breast(ID_model, tol_fun, plot=False): #breast tol_fun = 135
    idx = Apply_Constraints(constraints_breast[ID_model-1],parameters_breast_df)
    new_lowerBounds = np.asarray(parameters_breast_df["Lower bound"].loc[idx].tolist())
    new_upperBounds = np.asarray(parameters_breast_df["Upper bound"].loc[idx].tolist())
    MinLik_breast = lambda *args: -log_likelihood_breast(*args)
    final_result = None
    for it in range(50):
        initial_guess = np.random.uniform(new_lowerBounds,new_upperBounds)
        result = op.minimize(MinLik_breast, initial_guess, method = 'SLSQP', args=(ID_model), bounds=Bounds(new_lowerBounds,new_upperBounds), options={'ftol': 1e-10, 'disp':False})
        if (result['success'] and result['fun'] < tol_fun): # convergence check and tolerance
            if ( final_result == None or result['fun'] < final_result['fun']):
                final_result = result
    if not plot:
        if(final_result): return final_result
        else:
            print("Error: Optimization didn't work. Increase the number of restarts or the tolerance.\n")
            exit(-1)
    # Plot the approx
    model_mle = forward_model_breast(obs_data.obs_time, final_result['x'], ID_model)
    fig, ax = plt.subplots(figsize=(9, 4))
    fig.suptitle(f"Params: {final_result['x']} Noise: {final_result['fun']}")
    ax.scatter(obs_data.obs_time,obs_data.DsRedBreastMean, marker='o', color='r',label='RFP')
    ax.scatter(obs_data.obs_time,obs_data.GFPBreastMean, marker='o', color='g',label='GFP')
    ax.errorbar(obs_data.obs_time[1:],obs_data.DsRedBreastMean[1:],yerr=obs_data.DsRedBreastStd[1:],ls='none',ecolor='red',capsize=10)
    ax.errorbar(obs_data.obs_time[1:],obs_data.GFPBreastMean[1:],yerr=obs_data.GFPBreastStd[1:],ls='none',ecolor='green',capsize=10)
    ax.plot(obs_data.obs_time, model_mle[:,0], color='r',label='DsRed')
    ax.plot(obs_data.obs_time, model_mle[:,1], color='g',label='GFP')
    ax.set(xlabel = "Time (days)", ylabel = "Number of cells", title = "Breast (RFP and GFP cells)")
    plt.show()

def MaxLikEst_mets(ID_model, theta_breast, modelID_breast, tol_fun, plot=False): #breast tol_fun = 135
    idx = Apply_Constraints(constraints_mets[ID_model-1],parameters_mets_df)
    new_lowerBounds = np.asarray(parameters_mets_df["Lower bound"].loc[idx].tolist())
    new_upperBounds = np.asarray(parameters_mets_df["Upper bound"].loc[idx].tolist())
    MinLik_breast = lambda *args: -log_likelihood_mets(*args)
    final_result = None
    for it in range(50):
        initial_guess = np.random.uniform(new_lowerBounds,new_upperBounds)
        result = op.minimize(MinLik_breast, initial_guess, method = 'SLSQP', args=(ID_model,theta_breast, modelID_breast), bounds=Bounds(new_lowerBounds,new_upperBounds), options={'ftol': 1e-10, 'disp':False})
        if (result['success'] and result['fun'] < tol_fun): # convergence check and tolerance
            if ( final_result == None or result['fun'] < final_result['fun']):
                final_result = result
    if not plot:
        if(final_result): return final_result
        else:
            print("Error: Optimization didn't work. Increase the number of restarts or the tolerance.\n")
            exit(-1)
    # Plot the approx
    model_mle = forward_model_mets(obs_data.obs_time, final_result['x'], ID_model, theta_breast, modelID_breast)
    fig, ax = plt.subplots(1,2,figsize=(12, 4))
    fig.suptitle(f"Params: {final_result['x']} Noise: {final_result['fun']}")
    # Plot Bloodstream
    ax[0].scatter([obs_data.obs_time[0],obs_data.obs_time[-1]], obs_data.DsRedBloodStMean, marker='o', color='r',label='DsRed')
    ax[0].scatter([obs_data.obs_time[0],obs_data.obs_time[-1]], obs_data.GFPBloodStMean, marker='o', color='g',label='GFP')
    ax[0].errorbar(obs_data.obs_time[-1],obs_data.DsRedBloodStMean[-1], yerr=obs_data.DsRedBloodStStd[-1],ls='none',ecolor='red',capsize=10)
    ax[0].errorbar(obs_data.obs_time[-1],obs_data.GFPBloodStMean[-1], yerr=obs_data.GFPBloodStStd[-1],ls='none',ecolor='green',capsize=10)
    ax[0].plot(obs_data.obs_time, model_mle[:,2], color='r',label='RFP')
    ax[0].plot(obs_data.obs_time, model_mle[:,3], color='g',label='GFP')
    ax[0].set(xlabel = "Time (days)", ylabel = "Number of cells", title="Bloodstream (RFP and GFP cells)")
    # Plot Lung
    ax[1].scatter(obs_data.obs_time,obs_data.DsRedLungMean, marker='o', color='r',label='RFP')
    ax[1].scatter(obs_data.obs_time,obs_data.GFPLungMean, marker='o', color='g',label='GFP')
    ax[1].errorbar(obs_data.obs_time[1:],obs_data.DsRedLungMean[1:],yerr=obs_data.DsRedLungStd[1:],ls='none',ecolor='red',capsize=10)
    ax[1].errorbar(obs_data.obs_time[1:],obs_data.GFPLungMean[1:],yerr=obs_data.GFPLungStd[1:],ls='none',ecolor='green',capsize=10)
    ax[1].plot(obs_data.obs_time, model_mle[:,4], color='r',label='RFP')
    ax[1].plot(obs_data.obs_time, model_mle[:,5], color='g',label='GFP')
    ax[1].set(xlabel = "Time (days)", ylabel = "Number of cells", title="Lung (RFP and GFP cells)")
    plt.show()

# Bayes Inference
def BayesInference_MCMC_breast(ID_model, tol_fun, fileName, Num_Walkers = 30, Max_Iterations = 1000, mle_range = None):
    idx = Apply_Constraints(constraints_breast[ID_model-1],parameters_breast_df)
    old_lowerBounds = np.asarray(parameters_breast_df["Lower bound"].loc[idx].tolist())
    old_upperBounds = np.asarray(parameters_breast_df["Upper bound"].loc[idx].tolist())
    if (mle_range):
        mle_param = MaxLikEst_breast(modelID, tol_fun)['x']
        new_lowerBounds = mle_param - mle_range*mle_param
        new_upperBounds = mle_param + mle_range*mle_param
        # Check if inside of old prior distribution
        new_lowerBounds = np.asarray([new if old < new else old for old, new in zip(old_lowerBounds, new_lowerBounds)])
        new_upperBounds = np.asarray([new if old > new else old for old, new in zip(old_upperBounds, new_upperBounds)])
    else:
        new_lowerBounds = old_lowerBounds
        new_upperBounds = old_upperBounds

    Num_Dimension = len(new_lowerBounds)
    initial_state = np.random.uniform(new_lowerBounds,new_upperBounds, size=(Num_Walkers,Num_Dimension))

    backend = emcee.backends.HDFBackend(fileName)
    backend.reset(Num_Walkers, Num_Dimension)
    sampler = emcee.EnsembleSampler(Num_Walkers, Num_Dimension, log_probability, threads=10,backend=backend, args=[new_lowerBounds,new_upperBounds,ID_model])
    sampler.run_mcmc(initial_state, Max_Iterations, progress=True);

def BayesInference_MCMC_mets(ID_model, theta_breast, modelID_breast, tol_fun, fileName, Num_Walkers = 30, Max_Iterations = 1000, mle_range = None):
    idx = Apply_Constraints(constraints_mets[ID_model-1],parameters_mets_df)
    old_lowerBounds = np.asarray(parameters_mets_df["Lower bound"].loc[idx].tolist())
    old_upperBounds = np.asarray(parameters_mets_df["Upper bound"].loc[idx].tolist())
    if (mle_range):
        mle_param = MaxLikEst_mets(modelID, theta_breast, modelID_breast, tol_fun)['x']
        new_lowerBounds = mle_param - mle_range*mle_param
        new_upperBounds = mle_param + mle_range*mle_param
        # Check if inside of old prior distribution
        new_lowerBounds = np.asarray([new if old < new else old for old, new in zip(old_lowerBounds, new_lowerBounds)])
        new_upperBounds = np.asarray([new if old > new else old for old, new in zip(old_upperBounds, new_upperBounds)])
    else:
        new_lowerBounds = old_lowerBounds
        new_upperBounds = old_upperBounds

    Num_Dimension = len(new_lowerBounds)
    initial_state = np.random.uniform(new_lowerBounds,new_upperBounds, size=(Num_Walkers,Num_Dimension))

    backend = emcee.backends.HDFBackend(fileName)
    backend.reset(Num_Walkers, Num_Dimension)
    sampler = emcee.EnsembleSampler(Num_Walkers, Num_Dimension, log_probability, threads=10,backend=backend, args=[new_lowerBounds,new_upperBounds,modelID_breast,ID_model,theta_breast])
    sampler.run_mcmc(initial_state, Max_Iterations, progress=True);

def Loading_MCMC_breast(folder, Measure='MLE',burn_in = 100, modelID_specific=None):
    NumParameters = []
    MLE_Parameters = []
    Median_Parameters = []
    Mean_Parameters = []
    LL_value = []
    LL_Occam_value = []
    RSS_value = []
    MLE_Solution = []
    FlatSamples = []
    Samples_Chains = []
    Samples_Chains_loglik = []
    if (modelID_specific): ModelsID = [modelID_specific]
    else: ModelsID = range(1,len(constraints_breast)+1)
    for modelID in ModelsID:
        idx = Apply_Constraints(constraints_breast[modelID-1],parameters_breast_df)
        reader = emcee.backends.HDFBackend(folder+"MCMC_Breast%03d.h5"%modelID)
        thin = int(0.1*burn_in)
        samples = reader.get_chain()
        flat_samples = reader.get_chain(discard=burn_in, flat=True, thin=thin)
        samples_loglik = reader.get_log_prob(discard=burn_in, flat=True, thin=thin)
        idx_ML = np.argwhere(samples_loglik == samples_loglik.max())
        ML_Par = flat_samples[idx_ML[0],:]
        Mean_Par = np.mean(flat_samples,axis=0)
        Median_Par = np.median(flat_samples,axis=0)
        MLE_Parameters.append(ML_Par)
        Median_Parameters.append(Median_Par)
        Mean_Parameters.append(Mean_Par)
        NumParameters.append(len(idx))
        # Choose Solution: Mean, Mode, Median
        if (Measure == 'MLE'): CentralMeasure = ML_Par
        if (Measure == 'Mean'): CentralMeasure = Mean_Par
        if (Measure == 'Median'): CentralMeasure = Median_Par
        log_likelihood_value = log_likelihood_breast(CentralMeasure, modelID)
        LL_value.append(log_likelihood_value)
        RSS_value.append(RSS_breast(CentralMeasure,modelID))
        MLE_Solution.append(forward_model_breast(smooth_time,CentralMeasure,modelID))
        FlatSamples.append(flat_samples)
        Samples_Chains.append(samples)
        Samples_Chains_loglik.append(samples_loglik)
    return NumParameters, MLE_Parameters, MLE_Solution, Median_Parameters, Mean_Parameters, LL_value, RSS_value, FlatSamples, Samples_Chains, Samples_Chains_loglik

def Loading_MCMC_mets(folder,theta_breast, modelID_breast, Measure='MLE',burn_in = 100, modelID_specific=None):
    NumParameters = []
    MLE_Parameters = []
    Median_Parameters = []
    Mean_Parameters = []
    LL_value = []
    LL_Occam_value = []
    RSS_value = []
    MLE_Solution = []
    FlatSamples = []
    Samples_Chains = []
    Samples_Chains_loglik = []
    if (modelID_specific): ModelsID = [modelID_specific]
    else: ModelsID = range(1,len(constraints_mets)+1)
    for modelID in ModelsID:
        idx = Apply_Constraints(constraints_mets[modelID-1],parameters_mets_df)
        reader = emcee.backends.HDFBackend(folder+"MCMC_Breast%03d_Mets%03d.h5"%(modelID_breast,modelID))
        thin = int(0.1*burn_in)
        samples = reader.get_chain()
        flat_samples = reader.get_chain(discard=burn_in, flat=True, thin=thin)
        samples_loglik = reader.get_log_prob(discard=burn_in, flat=True, thin=thin)
        idx_ML = np.argwhere(samples_loglik == samples_loglik.max())
        ML_Par = flat_samples[idx_ML[0],:]
        Mean_Par = np.mean(flat_samples,axis=0)
        Median_Par = np.median(flat_samples,axis=0)
        MLE_Parameters.append(ML_Par)
        Median_Parameters.append(Median_Par)
        Mean_Parameters.append(Mean_Par)
        NumParameters.append(len(idx))
        # Choose Solution: Mean, Mode, Median
        if (Measure == 'MLE'): CentralMeasure = ML_Par
        if (Measure == 'Mean'): CentralMeasure = Mean_Par
        if (Measure == 'Median'): CentralMeasure = Median_Par
        log_likelihood_value = log_likelihood_mets(CentralMeasure, modelID, theta_breast, modelID_breast)
        LL_value.append(log_likelihood_value)
        RSS_value.append(RSS_mets(CentralMeasure,modelID,theta_breast,modelID_breast))
        MLE_Solution.append(forward_model_mets(smooth_time,CentralMeasure,modelID, theta_breast, modelID_breast))
        FlatSamples.append(flat_samples)
        Samples_Chains.append(samples)
        Samples_Chains_loglik.append(samples_loglik)
    return NumParameters, MLE_Parameters, MLE_Solution, Median_Parameters, Mean_Parameters, LL_value, RSS_value, FlatSamples, Samples_Chains, Samples_Chains_loglik

def Constraints_to_hypothesis(constraints):
    Hypos = []
    for hyp in constraints:
        hypothesis = []
        if (('K_b = inf' in hyp) or ('K_l = inf' in hyp)): hypothesis.append('EG')
        else: hypothesis.append('LG')
        if('tau = 0' in hyp): hypothesis.append('NDI')
        elif(['tau = 0'] in constraints): hypothesis.append('DI')
        if('alpha_bg = alpha_br' in hyp): hypothesis.append('H1')
        if('beta_g = beta_r' in hyp): hypothesis.append('H2a')
        elif('beta_g = 1.5*beta_r' in hyp): hypothesis.append('H2b')
        if('phi_r = phi_g' in hyp): hypothesis.append('H3a')
        elif('phi_r = 3*phi_g' in hyp): hypothesis.append('H3b')
        elif('phi_r = 6*phi_g' in hyp): hypothesis.append('H3c')
        if('f_Eg = 1.0*phi_r/phi_g*f_Er' in hyp): hypothesis.append('H4a')
        elif('f_Eg = 1.5*phi_r/phi_g*f_Er' in hyp): hypothesis.append('H4b')
        if('alpha_lg = alpha_lr' in hyp): hypothesis.append('H5')
        Hypos.append(hypothesis)
    return Hypos

if __name__ == '__main__':
    folder = "MCMC_Files/Model_Candidates/"
    # Breast
    tol_fun_breast = 135
    for modelID in range(int(sys.argv[1]),int(sys.argv[2])+1):
        print('Model breast: ',modelID)
        BayesInference_MCMC_breast(modelID, tol_fun_breast, fileName=folder+"MCMC_Breast%03d.h5"%modelID, mle_range=0.2) # MCMC - Bayes Inference
        # MaxLikEst_breast(modelID, tol_fun_breast, plot=True)

    Model_Parameters_Breast = {18: np.array([[0.12788638, 0.01795128, 0.03986062]]), 22: np.array([[ 0.11320129,  0.01563299,  0.03987029, 11.76371081]])}
    # Bloodstream and Lung
    tol_fun_mets = 133
    ModelID_breast = int(sys.argv[3])
    theta_breast = Model_Parameters_Breast[ModelID_breast]
    for modelID in range(int(sys.argv[1]),int(sys.argv[2])+1):
        print('Model breast: ',ModelID_breast,'  Model mets: ', modelID)
        BayesInference_MCMC_mets(modelID,theta_breast,ModelID_breast,tol_fun_mets, fileName=folder+"MCMC_Breast%03d_Mets%03d.h5"%(ModelID_breast,modelID), mle_range=0.2) # MCMC - Bayes Inference
        # MaxLikEst_mets(modelID,theta_breast,ModelID_breast,tol_fun_mets, plot=True)
