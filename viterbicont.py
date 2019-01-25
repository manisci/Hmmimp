import numpy as np,numpy.random
from init_gaussian import hmmgaussian
from scipy import stats
from forwardcont import forwardcont
from backwardcont import backwardcont
from forward_backward_cont import forward_backwardcont
from sklearn.preprocessing import normalize


def viterbicont(transmtrx,obsmtrx,pie,observations):
    ''' Input : Transition matrix, pie, state_observation probs, observations
    Output : The most likely sequence of states (and also the probabilite for each time point) and its probabilite for the given observations
    Unlike forward backward, considers the most probable sequence given the state for all time points
    not just the optimum state individually for each time point'''
    # initialization
    eps = 2.22044604925e-16
    probeps = 0.1
    if len(np.shape(observations)) == 1:
        numstates = np.shape(transmtrx)[0]
        timelength = np.shape(observations)[0]
        deltas = eps * np.ones((timelength,numstates))
        optzis = eps * np.ones((timelength))
        As = eps * np.ones((timelength,numstates))
        # .reshape(-1,1)
        probs = eps * np.ones(numstates)
        for state in range(numstates):
            distr = stats.norm(obsmtrx[state,0], obsmtrx[state,1])
            probs[state] = distr.cdf(observations[0]+probeps) - distr.cdf(observations[0]- probeps)
        deltas[0,:] = normalize((np.multiply(probs,pie)).reshape(1, -1),norm = 'l1')
        for t in range(1,timelength):
            # set A here
            for j in range(numstates):
                # print deltas[t-1,:] * transmtrx[:,j] * obsmtrx[j,int(observations[t])]
                distr = stats.norm(obsmtrx[j,0], obsmtrx[j,1])
                prob = distr.cdf(observations[t]+probeps) - distr.cdf(observations[t]- probeps)
                normed = normalize((deltas[t-1,:] * transmtrx[:,j] * prob).reshape(1, -1),norm = 'l1')
                # print normed
                As[t,j] = int(np.argmax(normed))
                cands = eps * np.ones((numstates))
                for i in range(numstates):
                    cands[i] = deltas[t-1,i] *(transmtrx[i,j]) *(prob)
                deltas[t,j] = max(cands)
            deltas[t,:] = normalize(deltas[t,:].reshape(1, -1),norm = 'l1')
        optzis[timelength-1] = int(np.argmax(deltas[timelength-1,:]))
        for k in range(timelength-2,-1,-1):
            optzis[k] = As[k+1,int(optzis[k+1])]
    else:
        numstates = np.shape(transmtrx)[0]
        timelength = np.shape(observations)[1]
        numsamples = np.shape(observations)[0]
        deltas = eps * np.ones((numsamples,timelength,numstates))
        optzis = eps *np.ones((numsamples,timelength))
        As = eps * np.ones((numsamples,timelength,numstates))
        for sample in range(numsamples):
            probs = eps * np.ones(numstates)
            for state in range(numstates):
                distr = stats.norm(obsmtrx[state,0], obsmtrx[state,1])
                probs[state] = distr.cdf(observations[sample,0]+probeps) - distr.cdf(observations[sample,0]- probeps)
            deltas[sample,0,:]  = normalize((np.multiply(probs,pie)).reshape(1, -1),norm = 'l1')
            # otherzero = np.argmax(deltas[sample,0,:])
            for t in range(1,timelength):
                # set A here
                for j in range(numstates):
                    # print deltas[t-1,:] * transmtrx[:,j] * obsmtrx[j,int(observations[t])]
                    distr = stats.norm(obsmtrx[j,0], obsmtrx[j,1])
                    prob = distr.cdf(observations[sample,t]+probeps) - distr.cdf(observations[sample,t]- probeps)
                    normed = normalize((deltas[sample,t-1,:] * transmtrx[:,j] * prob).reshape(1, -1),norm = 'l1')
                    # print normed
                    As[sample,t,j] = int(np.argmax(normed))
                    cands = eps * np.ones((numstates))
                    for i in range(numstates):
                        cands[i] = deltas[sample,t-1,i] *(transmtrx[i,j]) *(prob)
                    deltas[sample,t,j] = max(cands)
                deltas[sample,t,:] = normalize(deltas[sample,t,:].reshape(1, -1),norm = 'l1')
            optzis[sample,timelength-1] = int(np.argmax(deltas[sample,timelength-1,:]))
            for k in range(timelength-2,-1,-1):
                optzis[sample,k] = As[sample,k+1,int(optzis[sample,k+1])]       
    return (optzis,deltas)

# def main():
#     exmodel = hmmgaussian(3,1,50,3)
#     observations = exmodel.observations
#     pie = exmodel.pie
#     transmtrx = exmodel.transitionmtrx
#     obsmtrx = exmodel.obsmtrx
#     seqofstates = exmodel.seqofstates
#     (gammas,betas,alphas,log_prob_most_likely_seq,most_likely_seq,forward_most_likely_seq,forward_log_prob_most_likely_seq,Ziis,logobservations) = forward_backwardcont(transmtrx,obsmtrx,pie,observations)
#     print "forward_backward acc"
#     print np.sum(seqofstates==most_likely_seq) / float(exmodel.obserlength)
#     print "forward acc"
#     print np.sum(seqofstates==forward_most_likely_seq) / float(exmodel.obserlength)
#     print "forward_backward prob"
#     print log_prob_most_likely_seq
#     print "forward prob"
#     print forward_log_prob_most_likely_seq
#     (mlpath,deltas) = viterbicont(transmtrx,obsmtrx,pie,observations)
#     print "forward_backward is more certain at each time point"
#     # for i in range(exmodel.obserlength):
#     #     if max(alphas[i,:]) <= max(gammas[i,:]):
#     #         numwins +=1
#     # print numwins / float(exmodel.obserlength)

#     # # print mlpath
#     # print "viterbi similar to reality"
#     # print np.sum(seqofstates==mlpath) / float(exmodel.obserlength)
#     # print "vitervi similarity to forward_backward"
#     # print np.sum(mlpath==most_likely_seq) / float(exmodel.obserlength)
#     # print "viterbi similarity to forward seq"
#     # print np.sum(mlpath==forward_most_likely_seq) / float(exmodel.obserlength)

# main()