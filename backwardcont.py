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

def backward(transmtrx,obsmtrx,pie,observations):
    ''' Input : Transition matrix, pie, state_observation probs, observations
    Output: betas,  Probablities of observing the rest of the observations from that point on, given that we are at a given state at a give timepoint for each sample'''
    # initialization
    eps = 2.22044605e-16
    if len(np.shape(observations)) == 1:
        numstates = np.shape(transmtrx)[0]
        timelength = np.shape(observations)[0]
        betas = np.empty((timelength,numstates))
        betas[timelength-1,:] = np.ones((1,numstates))
        # print betas[timelength-1,:]
        for t in range(timelength-1,0,-1):
            phi_t = eps * np.empty(numstates)
            for state in range(numstates):
                distr = stats.norm(obsmtrx[state,0], obsmtrx[state,1])
                phi_t[state] = distr.pdf(observations[t])
            betas[t-1,:] = np.matmul(transmtrx,np.multiply(phi_t , (betas[t,:])))
    else:
        # multiple samples
        numstates = np.shape(transmtrx)[0]
        numsamples = np.shape(observations)[0]
        timelength = np.shape(observations)[1]
        betas = np.empty((numsamples,timelength,numstates))
        for sample in range(numsamples):
            betas[sample,timelength-1,:] = np.ones((1,numstates))
            # print betas[timelength-1,:]
            for t in range(timelength-1,0,-1):
                phi_t = eps * np.empty(numstates)
                for state in range(numstates):
                    distr = stats.norm(obsmtrx[state,0], obsmtrx[state,1])
                    phi_t[state] = distr.pdf(observations[sample,t])
                betas[sample,t-1,:]= np.matmul(transmtrx,np.multiply(phi_t , (betas[sample,t,:])))

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