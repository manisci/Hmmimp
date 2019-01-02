import numpy as np,numpy.random
from init_forward import hmmforward
from scipy import stats
from forward import forward
from backward import backward

def normalize(u):
    Z = np.sum(u)
    if Z==0:
        return (u,1.0)
    else:
        v = u / Z
    return (v,Z)

def clipvalues_prevunderflowfw(alphas,betas,gammas,Ziis):
    gammas = np.clip(gammas,2.22044604925e-16,1.0)
    alphas = np.clip(alphas,2.22044604925e-16,1.0)
    betas = np.clip(betas,2.22044604925e-16,1.0)
    Ziis = np.clip(Ziis,2.22044604925e-16,1.0)
    return (alphas,betas,gammas,Ziis)

def forward_backward(transmtrx,obsmtrx,pie,observations):
    # initialization
    numstates = np.shape(transmtrx)[0]
    timelength = np.shape(observations)[0]
    gammas = np.empty((timelength,numstates))
    (alphas,forward_log_prob_most_likely_seq,forward_most_likely_seq,Ziis) = forward(transmtrx,obsmtrx,pie,observations)
    betas = backward(transmtrx,obsmtrx,pie,observations)
    Zis = np.zeros((timelength,1))
    most_likely_seq = np.empty((timelength,1))
    # for i in range(timelength):
    #     betas[i,:] /= float(Ziis[i])

    for t in range(timelength):
        (gammas[t,:],Zis[t]) =  normalize(np.multiply(alphas[t,:],betas[t,:]))
        most_likely_seq[t] = np.argmax(gammas[t,:])
    (alphas,betas,gammas,Ziis) = clipvalues_prevunderflowfw(alphas,betas,gammas,Ziis)
    log_prob_most_likely_seq = np.sum(np.log(Zis) + 2.22044604925e-16)

    return (gammas,betas,alphas,log_prob_most_likely_seq,most_likely_seq,forward_most_likely_seq,forward_log_prob_most_likely_seq,Ziis)



# def main():
#     exmodel = hmmforward(5,10,2,50)
#     observations = exmodel.observations
#     pie = exmodel.pie
#     transmtrx = exmodel.transitionmtrx
#     obsmtrx = exmodel.obsmtrx
#     seqofstates = exmodel.seqofstates
#     (gammas,betas,alphas,log_prob_most_likely_seq,most_likely_seq,forward_most_likely_seq,forward_log_prob_most_likely_seq,,Zis) = forward_backward(transmtrx,obsmtrx,pie,observations)
#     print "forward_backward acc"
#     print np.sum(seqofstates==most_likely_seq) / float(exmodel.obserlength)
#     print "forward acc"
#     print np.sum(seqofstates==forward_most_likely_seq) / float(exmodel.obserlength)
#     print "forward_backward prob"
#     print log_prob_most_likely_seq
#     print "forward prob"
#     print forward_log_prob_most_likely_seq
#     print "forward_backward is more certain at each time point"
#     numwins = 0
#     for i in range(exmodel.obserlength):
#         if max(alphas[i,:]) <= max(gammas[i,:]):
#             numwins +=1
#     print numwins / float(exmodel.obserlength)

    

#     # print stats.mode(seqofstates)
#     # print stats.mode(most_likely_seq)
# main()