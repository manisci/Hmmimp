import numpy as np,numpy.random
from init_forward import hmmforward
from scipy import stats

def normalize(u):
    Z = np.sum(u)
    v = u / Z
    return (v,Z)


def backward(transmtrx,obsmtrx,pie,observations):
    # initialization
    numstates = np.shape(transmtrx)[0]
    timelength = np.shape(observations)[0]
    betas = np.empty((timelength,numstates))
    betas[timelength-1,:] = np.ones((1,numstates))
    # print betas[timelength-1,:]
    for t in range(timelength-1,0,-1):
        phi_t = obsmtrx[:,int(observations[t])]
        betas[t-1,:] = np.matmul(transmtrx,np.multiply(phi_t , (betas[t,:]))) 
    return (betas)

# def main():
#     exmodel = hmmforward(5,10,1,10)
#     observations = exmodel.observations
#     pie = exmodel.pie
#     transmtrx = exmodel.transitionmtrx
#     obsmtrx = exmodel.obsmtrx
#     seqofstates = exmodel.seqofstates
#     betas = backward(transmtrx,obsmtrx,pie,observations)
# main()