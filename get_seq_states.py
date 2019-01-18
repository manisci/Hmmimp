import numpy as np,numpy.random
from init_forward import hmmforward
from scipy import stats
from forward import forward
from backward import backward
from forward_backward import forward_backward
from Baumwelch import Baumwelch
from viterbi import viterbi
 

def getseqofstates(numstates,numobscases,numsamples,observations,exmodel):
    ''' returns most probable seq of states and probability of 
    being in each state for each timestep, as a second argument of size numtimesteps * numstates
    Input : numstates,numobscases,numsamples,observations,exmodel
    Output: Most likely sequence of state, Optimal Z's and probabilites of being in each state for 
    each time step. 
    Exmodel may only be used for the close to reality initialization and is optional

    '''
    # print gammas
    # print most_likely_seq
    (pie,transmtrx,obsmtrx) = Baumwelch(observations,numstates,numobscases,numsamples,exmodel)
    (seq_states ,deltas)= viterbi(transmtrx,obsmtrx,pie,observations)
    return (seq_states,deltas)
def main():
    numstates = 2
    numobscases = 3
    numsamples = 1
    numbofobsrv = 50
    exmodel = hmmforward(numstates,numobscases,1,numbofobsrv,numsamples)
    observations = exmodel.observations
    (seqofstates,deltas) = getseqofstates(numstates,numobscases,numsamples,observations,exmodel)
    realseqofstates = exmodel.seqofstates
    # print np.sum(seqofstates[0,:] == realseqofstates[0,:])
    print "I got this much of the sequence right!"
    NoCorstates = 0
    if numsamples > 1:
        for sample in range(numsamples):
            NoCorstates += float(np.sum(seqofstates[sample,:] == realseqofstates[sample,:])) 
    else:
        NoCorstates += float(np.sum(seqofstates == realseqofstates))         
    print float(NoCorstates) /float(numsamples * numbofobsrv)
    # print deltas
    # print np.shape(seqofstates)
    # print np.shape(realseqofstates)
    # print np.shape(deltas)
    # print np.concatenate((seqofstates,realseqofstates,deltas),axis = 1)
main()