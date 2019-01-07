import pytest
import numpy as np,numpy.random
from init_forward import hmmforward
from scipy import stats
from forward import forward
from backward import backward
from forward_backward import forward_backward
from scipy import stats

def normalize(u):
    Z = np.sum(u)
    if Z==0:
        v = u / (Z + 2.22044604925e-16)
    else:
        v = u / Z
    return (v,Z)

@pytest.mark.parametrize("nums,numobscase,piequality,numobservs",[(5,10,2,500),(2,4,1,500),(8,20,1,1000),(4,16,1,50)])

def test_viterbi(nums,numobscase,piequality,numobservs):
    ''' you can convert the observations and obsmtrx all together into one matrix
    called soft evidence which is a K * T matrix by using the corresponding
    distribution across all the states for each time point and use that instead'''
    # initialization
    hmmexample = hmmforward(nums,numobscase,piequality,numobservs,1)
    numstates = np.shape(hmmexample.transitionmtrx)[0]
    timelength = np.shape(hmmexample.observations)[0]
    deltas = np.empty((timelength,numstates))
    optzis = np.empty((timelength,1))
    As = np.empty((timelength,numstates))
    (deltas[0,:] ,Z) = normalize((np.multiply(hmmexample.obsmtrx[:,int(hmmexample.observations[0])],hmmexample.pie)))
    for t in range(1,timelength):
        # set A here
        for j in range(numstates): 
            # print deltas[t-1,:] * transmtrx[:,j] * obsmtrx[j,int(observations[t])]
            (normed,Z) = normalize(deltas[t-1,:] * hmmexample.transitionmtrx[:,j] * hmmexample.obsmtrx[j,int(hmmexample.observations[t])])
            # print normed
            As[t,j] = int(np.argmax(normed))
            cands = np.empty((numstates,1))
            for i in range(numstates):
                cands[i] = deltas[t-1,i] *(hmmexample.transitionmtrx[i,j]) *(hmmexample.obsmtrx[j,int(hmmexample.observations[t])])
            deltas[t,j] = max(cands)
        (deltas[t,:],Z) = normalize(deltas[t,:])
    optzis[timelength-1] = int(np.argmax(deltas[timelength-1,:]))
    for k in range(timelength-2,-1,-1):
        optzis[k] = As[k+1,int(optzis[k+1])]
    # assert stats.mode(hmmexample.seqofstates).mode == stats.mode(optzis).mode
    # assert stats.mode(optzis).mode == np.argmax(hmmexample.pie)
    assert abs(np.sum(deltas[1,:]) - 1 )< 0.01
    assert abs(np.sum(normed) -1) < 0.01
    assert np.sum(np.isinf(deltas)) == 0
    assert np.sum(np.isinf(cands)) == 0
    assert np.sum(np.isinf(As)) == 0


# def main():
#     exmodel = hmmforward(5,10,500,500,1)
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