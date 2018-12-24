import numpy as np,numpy.random
from init_forward import hmmforward
from scipy import stats

def normalize(u):
    Z = np.sum(u)
    v = u / Z
    return (v,Z)


def foward(transmtrx,obsmtrx,pie,observations):
    # initialization
    numstates = np.shape(transmtrx)[0]
    timelength = np.shape(observations)[0]
    Zis = np.empty((timelength,1))
    most_likely_seq = np.empty((timelength,1))
    alphas = np.empty((timelength,numstates))
    phi0 = obsmtrx[:,int(observations[0])]
    (alphas[0,:],Zis[0]) = normalize(np.multiply(phi0,pie)) 
    most_likely_seq[0] = np.argmax(alphas[0,:])
    print np.shape(pie)
    for t in range(1,timelength):
        phi_t = obsmtrx[:,int(observations[t])]
        (alphas[t,:],Zis[t]) = normalize(np.multiply(phi_t,np.matmul(np.transpose(transmtrx) , np.transpose(alphas[t-1,:])))) 
        most_likely_seq[t] = np.argmax(alphas[t,:])
    log_prob_most_likely_seq = np.sum(np.log(Zis))
    return (alphas,log_prob_most_likely_seq,most_likely_seq)



def main():
    exmodel = hmmforward(5,10,1,5000)
    observations = exmodel.observations
    pie = exmodel.pie
    transmtrx = exmodel.transitionmtrx
    obsmtrx = exmodel.obsmtrx
    seqofstates = exmodel.seqofstates
    (alphas,log_prob_most_likely_seq,most_likely_seq) = foward(transmtrx,obsmtrx,pie,observations)
    # print np.sum(seqofstates==most_likely_seq) / float(exmodel.obserlength)
    # print stats.mode(seqofstates)
    # print stats.mode(most_likely_seq)
main()