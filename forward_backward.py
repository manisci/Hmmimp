import numpy as np,numpy.random
from init_forward import hmmforward
from scipy import stats
from forward import foward
from backward import backward
from datashape.coretypes import float32

def normalize(u):
    Z = np.sum(u)
    v = u / Z
    return (v,Z)


def foward_backward(transmtrx,obsmtrx,pie,observations):
    # initialization
    numstates = np.shape(transmtrx)[0]
    timelength = np.shape(observations)[0]
    gammas = np.empty((timelength,numstates))
    (alphas,forward_log_prob_most_likely_seq,forward_most_likely_seq) = foward(transmtrx,obsmtrx,pie,observations)
    betas = backward(transmtrx,obsmtrx,pie,observations)
    Zis = np.zeros((timelength,1))
    most_likely_seq = np.empty((timelength,1))
    for t in range(timelength):
        (gammas[t,:],Zis[t]) =  normalize(np.multiply(alphas[t,:],betas[t,:]))
        if Zis[t] == 0 :
            Zis[t] = 2.22044604925e-16 
        most_likely_seq[t] = np.argmax(gammas[t,:])
    log_prob_most_likely_seq = np.sum(np.log(Zis) + 2.22044604925e-16)
    return (gammas,alphas,log_prob_most_likely_seq,most_likely_seq,forward_most_likely_seq,forward_log_prob_most_likely_seq)



def main():
    exmodel = hmmforward(5,10,1,20)
    observations = exmodel.observations
    pie = exmodel.pie
    transmtrx = exmodel.transitionmtrx
    obsmtrx = exmodel.obsmtrx
    seqofstates = exmodel.seqofstates
    (gammas,alphas,log_prob_most_likely_seq,most_likely_seq,forward_most_likely_seq,forward_log_prob_most_likely_seq) = foward_backward(transmtrx,obsmtrx,pie,observations)
    print "forward_backward acc"
    print np.sum(seqofstates==most_likely_seq) / float(exmodel.obserlength)
    print "forward acc"
    print np.sum(seqofstates==forward_most_likely_seq) / float(exmodel.obserlength)
    print "forward_backward prob"
    print log_prob_most_likely_seq
    print "forward prob"
    print forward_log_prob_most_likely_seq
    print "forward_backward is more certain at each time point"
    numwins = 0
    for i in range(exmodel.obserlength):
        if max(alphas[i,:]) <= max(gammas[i,:]):
            numwins +=1
    print numwins / float(exmodel.obserlength)

    

    # print stats.mode(seqofstates)
    # print stats.mode(most_likely_seq)
main()