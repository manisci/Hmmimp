import numpy as np,numpy.random
from init_forward import hmmforward
from scipy import stats

def normalize(u):
    Z = np.sum(u)
    if Z==0:
        return (u,1.0)
    else:
        v = u / Z
    return (v,Z)

def backward(transmtrx,obsmtrx,pie,observations):
    # initialization
    numstates = np.shape(transmtrx)[0]
    timelength = np.shape(observations)[0]
    betas = np.zeros((timelength,numstates))
    (betas[timelength-1,:], dumm) = normalize(np.ones((1,numstates)))
    # (betas[timelength-1,:],dummy) = normalize(betas[timelength-1,:])
    # print betas[timelength-1,:]
    for t in range(timelength-1,0,-1):
        phi_t = obsmtrx[:,int(observations[t])]
        (betas[t-1,:], dumm) = normalize(np.matmul(transmtrx,np.multiply(phi_t , (betas[t,:])))) 
        # (betas[t-1,:],dummy) = normalize(betas[t-1,:])
    (betas[0,:], doosh) = normalize(betas[0,:])
    return (betas)

# def main():
#     exmodel = hmmforward(5,10,1,500)
#     observations = exmodel.observations
#     pie = exmodel.pie
#     transmtrx = exmodel.transitionmtrx
#     obsmtrx = exmodel.obsmtrx
#     seqofstates = exmodel.seqofstates
#     betas = backward(transmtrx,obsmtrx,pie,observations)
# main()