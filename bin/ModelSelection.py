import numpy as np

def AkaikeInfCriterion(loglikelihood,numParameters): # AIC
    return -2*loglikelihood + 2*numParameters

def AkaikeInfCriterion_C(loglikelihood,numParameters,dataSize): # second-order AIC
    return AkaikeInfCriterion(loglikelihood,numParameters) + (2*numParameters*(numParameters+1))/(dataSize-numParameters-1)

def BayesInfCriterion(loglikelihood,numParameters,dataSize): # BIC
    return -2*loglikelihood + numParameters*np.log(dataSize)

def weight_InfCriterion(ValuesInfCriterion):
    Delta_IC = np.array(ValuesInfCriterion) - min(ValuesInfCriterion)
    denominator = np.sum(np.exp(-0.5*Delta_IC))
    weight = np.exp(-0.5*Delta_IC)/denominator
    return weight, Delta_IC

def evidence_ratio(Delta_IC):
    Delta_IC_best = min(Delta_IC)
    ER = np.exp(-0.5*Delta_IC_best)/np.exp(-0.5*Delta_IC)
    return ER

def Model_Selection(Parameters, max_log_likelihood, dataSize, Constraints):
    AIC = []; AIC_c = []; BIC = []; AIC_RSS = [];
    for modelID in range(len(Constraints)):
        numParameters = len(Parameters[modelID])
        AIC.append(AkaikeInfCriterion(max_log_likelihood[modelID],numParameters))
        AIC_c.append(AkaikeInfCriterion_C(max_log_likelihood[modelID],numParameters,dataSize))
        BIC.append(BayesInfCriterion(max_log_likelihood[modelID],numParameters,dataSize))
    return AIC,AIC_c,BIC
