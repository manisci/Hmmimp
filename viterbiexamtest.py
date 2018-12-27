import numpy as np,numpy.random
from init_forward import hmmforward
from scipy import stats
from forward import forward
from backward import backward
from forward_backward import forward_backward

def normalize(u):
    Z = np.sum(u)
    if Z==0:
        v = u / (Z+ 2.22044604925e-16)
    else:
        v = u / Z
    return (v,Z)

def viterbi(transmtrx,obsmtrx,pie,observations):
    ''' you can convert the observations and obsmtrx all together into one matrix
    called soft evidence which is a K * T matrix by using the corresponding
    distribution across all the states for each time point and use that instead'''
    # initialization
    numstates = np.shape(transmtrx)[0]
    timelength = np.shape(observations)[0]
    deltas = np.empty((timelength,numstates))
    optzis = np.empty((timelength,1))
    As = np.empty((timelength,numstates))
    (deltas[0,:] ,Z) = normalize((np.multiply(obsmtrx[:,int(observations[0])],pie)))
    otherzero = np.argmax(deltas[0,:])
    for t in range(1,timelength):
        # set A here
        for j in range(numstates):
            # print deltas[t-1,:] * transmtrx[:,j] * obsmtrx[j,int(observations[t])]
            (normed,Z) = normalize(deltas[t-1,:] * transmtrx[:,j] * obsmtrx[j,int(observations[t])])
            # print normed
            As[t,j] = int(np.argmax(normed))
            cands = np.empty((numstates,1))
            for i in range(numstates):
                cands[i] = deltas[t-1,i] *(transmtrx[i,j]) *(obsmtrx[j,int(observations[t])])
            deltas[t,j] = max(cands)
        (deltas[t,:],Z) = normalize(deltas[t,:])
    optzis[timelength-1] = int(np.argmax(deltas[timelength-1,:]))
    for k in range(timelength-2,-1,-1):
        optzis[k] = As[k+1,int(optzis[k+1])]
    return optzis

def main():
    # this is a toy example from slide 5 of 
    # https://ocw.mit.edu/courses/aeronautics-and-astronautics/16-410-principles-of-autonomy-and-decision-making-fall-2010/lecture-notes/MIT16_410F10_lec21.pdf
    observations = np.array([1,2,2,1,1,1,2,1,2])
    pie = np.array([1,0,0])
    transmtrx = np.array([[0,0.5,0.5],[0,0.9,0.1],[0,0,1]])
    obsmtrx = np.array([[0,0.5,0.5],[0,0.9,0.1],[0,0.1,0.9]])
    (gammas,alphas,log_prob_most_likely_seq,most_likely_seq,forward_most_likely_seq,forward_log_prob_most_likely_seq) = forward_backward(transmtrx,obsmtrx,pie,observations)
    print "forward_backward "
    print most_likely_seq
    print "forward acc"
    print forward_most_likely_seq
    print "forward_backward prob"
    print log_prob_most_likely_seq
    print "forward prob"
    print forward_log_prob_most_likely_seq
    print "forward_backward is more certain at each time point"

    mlpath = viterbi(transmtrx,obsmtrx,pie,observations)
    # print mlpath
    print "viterbi similar to reality"
    print mlpath

main()