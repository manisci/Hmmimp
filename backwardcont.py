import numpy as np,numpy.random
from init_gaussian import hmmgaussian
from scipy import stats


# def normalize(u):
#     Z = np.sum(u)
#     if Z==0:
#         return (u,1.0)
#     else:
#         v = u / Z
#     return (v,Z)

def clipvalues_prevoverflowfw(vector):
    eps = 2.22044604925e-16
    minpie = np.min(vector)
    maxpie = np.max(vector)
    vector[np.argmin(vector)] = eps
    vector[np.argmax(vector)] = 1.0
    for i in range(np.shape(vector)[0]):
        if vector[i] < eps:
            vector[i] = eps +( vector[i] - minpie)
        if vector[i] == 0:
            vector[i] =eps 
       
    return vector

def backwardcont(transmtrx,obsmtrx,pie,observations):
    ''' Input : Transition matrix, pie, state_observation probs, observations
    Output: betas,  Probablities of observing the rest of the observations from that point on, given that we are at a given state at a give timepoint for each sample'''
    # initialization
    eps = 2.22044605e-16
    if len(np.shape(observations)) == 1:
        numstates = np.shape(transmtrx)[0]
        timelength = np.shape(observations)[0]
        betas = eps * np.ones((timelength,numstates))
        betas[timelength-1,:] = np.ones((numstates))
        # print betas[timelength-1,:]
        for t in range(timelength-1,0,-1):
            phi_t = eps * np.ones(numstates)
            for state in range(numstates):
                probeps = abs((0.01 *  obsmtrx[state,1]))
                distr = stats.norm(obsmtrx[state,0], obsmtrx[state,1])
                phi_t[state] = distr.cdf(observations[t]+probeps) - distr.cdf(observations[t]- probeps)
            interm_result = np.multiply(phi_t , (betas[t,:]))
            betas[t-1,:] = np.matmul(transmtrx,interm_result)
            # betas[t-1,:] = clipvalues_prevoverflowfw(betas[t-1,:])

            
    else:
        # multiple samples
        numstates = np.shape(transmtrx)[0]
        numsamples = np.shape(observations)[0]
        timelength = np.shape(observations)[1]
        betas = eps * np.ones((numsamples,timelength,numstates))
        for sample in range(numsamples):
            betas[sample,timelength-1,:] = np.ones((1,numstates))
            # print betas[timelength-1,:]
            for t in range(timelength-1,0,-1):
                phi_t = eps * np.ones(numstates)
                for state in range(numstates):
                    probeps = abs((0.01 *  obsmtrx[state,1]))
                    distr = stats.norm(obsmtrx[state,0], obsmtrx[state,1])
                    phi_t[state] = distr.cdf(observations[sample,t]+probeps) - distr.cdf(observations[sample,t]- probeps)+ eps
                betas[sample,t-1,:]= np.matmul(transmtrx,np.multiply(phi_t , (betas[sample,t,:])))
                # betas[sample,t-1,:] = clipvalues_prevoverflowfw(betas[sample,t-1,:])


    return betas

# def main():
#     exmodel = hmmgaussian(3,1,50,1)
#     observations = exmodel.observations
#     pie = exmodel.pie
#     transmtrx = exmodel.transitionmtrx
#     obsmtrx = exmodel.obsmtrx
#     seqofstates = exmodel.seqofstates
#     betas = backward(transmtrx,obsmtrx,pie,observations)
#     print "just stay here"
# main()