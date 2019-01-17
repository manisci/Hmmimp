import numpy as np,numpy.random
from init_forward import hmmforward
from scipy import stats
from forward import forward
from backward import backward
from forward_backward import forward_backward
from sklearn.preprocessing import normalize

# def normalize(u):
#     Z = np.sum(u)
#     if Z==0:
#         return (u,1.0)
#     else:
#         v = u / Z
#     return (v,Z)

def viterbi(transmtrx,obsmtrx,pie,observations):
    ''' Input : Transition matrix, pie, state_observation probs, observations
    Output : The most likely sequence of states (and also the probabilite for each time point) and its probabilite for the given observations
    Unlike forward backward, considers the most probable sequence given the state for all time points
    not just the optimum state individually for each time point'''
    # initialization
    eps = 2.22044604925e-16
    if len(np.shape(observations)) == 1:
        numstates = np.shape(transmtrx)[0]
        timelength = np.shape(observations)[0]
        deltas = eps * np.ones((timelength,numstates))
        optzis = eps * np.ones((timelength))
        As = eps * np.ones((timelength,numstates))
        deltas[0,:] = normalize((np.multiply(obsmtrx[:,int(observations[0])],pie)).reshape(-1,1),norm = 'l1')
        otherzero = np.argmax(deltas[0,:])
        for t in range(1,timelength):
            # set A here
            for j in range(numstates):
                # print deltas[t-1,:] * transmtrx[:,j] * obsmtrx[j,int(observations[t])]
                normed = normalize((deltas[t-1,:] * transmtrx[:,j] * obsmtrx[j,int(observations[t])]).reshape(-1,1),norm = 'l1')
                # print normed
                As[t,j] = int(np.argmax(normed))
                cands = eps * np.ones((numstates))
                for i in range(numstates):
                    cands[i] = deltas[t-1,i] *(transmtrx[i,j]) *(obsmtrx[j,int(observations[t])])
                deltas[t,j] = max(cands)
            (deltas[t,:],Z) = normalize(deltas[t,:].reshape(-1,1),norm = 'l1')
        optzis[timelength-1] = int(np.argmax(deltas[timelength-1,:]))
        for k in range(timelength-2,-1,-1):
            optzis[k] = As[k+1,int(optzis[k+1])]
    else:
        numstates = np.shape(transmtrx)[0]
        timelength = np.shape(observations)[1]
        numsamples = np.shape(observations)[0]
        deltas = eps * np.ones((numsamples,timelength,numstates))
        optzis = eps *np.ones((numsamples,timelength))
        As = eps * np.ones((timelength,numstates))
        for sample in range(numsamples):
            deltas[sample,0,:]  = normalize((np.multiply(obsmtrx[:,int(observations[sample,0])],pie)).reshape(-1,1),norm = 'l1')
            # otherzero = np.argmax(deltas[sample,0,:])
            for t in range(1,timelength):
                # set A here
                for j in range(numstates):
                    # print deltas[t-1,:] * transmtrx[:,j] * obsmtrx[j,int(observations[t])]
                    normed = normalize((deltas[sample,t-1,:] * transmtrx[:,j] * obsmtrx[j,int(observations[sample,t])]).reshape(-1,1),norm = 'l1')
                    # print normed
                    As[t,j] = int(np.argmax(normed))
                    cands = eps * np.ones((numstates))
                    for i in range(numstates):
                        cands[i] = deltas[sample,t-1,i] *(transmtrx[i,j]) *(obsmtrx[j,int(observations[sample,t])])
                    deltas[sample,t,j] = max(cands)
                deltas[sample,t,:] = normalize(deltas[sample,t,:].reshape(-1,1),norm = 'l1')
            optzis[sample,timelength-1] = int(np.argmax(deltas[sample,timelength-1,:]))
            for k in range(timelength-2,-1,-1):
                optzis[sample,k] = As[k+1,int(optzis[k+1])]       

    return (optzis,deltas)

# def main():
#     exmodel = hmmforward(5,10,500,500)
#     observations = exmodel.observations
#     pie = exmodel.pie
#     transmtrx = exmodel.transitionmtrx
#     obsmtrx = exmodel.obsmtrx
#     seqofstates = exmodel.seqofstates
#     (gammas,alphas,log_prob_most_likely_seq,most_likely_seq,forward_most_likely_seq,forward_log_prob_most_likely_seq) = forward_backward(transmtrx,obsmtrx,pie,observations)
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

#     mlpath = viterbi(transmtrx,obsmtrx,pie,observations)
#     # print mlpath
#     print "viterbi similar to reality"
#     print np.sum(seqofstates==mlpath) / float(exmodel.obserlength)
#     print "vitervi similarity to forward_backward"
#     print np.sum(mlpath==most_likely_seq) / float(exmodel.obserlength)
#     print "viterbi similarity to forward seq"
#     print np.sum(mlpath==forward_most_likely_seq) / float(exmodel.obserlength)

# main()