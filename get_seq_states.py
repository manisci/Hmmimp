import numpy as np,numpy.random
from init_forward import hmmforward
from scipy import stats
from forward import forward
from backward import backward
from forward_backward import forward_backward
from Baumwelch import Baumwelch
from viterbi import viterbi
 

def getseqofstates(numstates,numobscases,numsamples,observations):
    ''' returns most probable seq of states and probability of 
    being in each state for each timestep, as a second argument of size numtimesteps * numstates'''
    # print gammas
    # print most_likely_seq
    exmodel = hmmforward(numstates,numobscases,1,len(observations))
    (pie,transmtrx,obsmtrx) = Baumwelch(observations,numstates,numobscases,numsamples,exmodel)
    (seq_states ,deltas)= viterbi(transmtrx,obsmtrx,pie,observations)
    return (seq_states,deltas)
def main():
    numstates = 2
    numobscases = 3
    numsamples = 1
    exmodel = hmmforward(numstates,numobscases,1,20)
    observations = exmodel.observations
    (seqofstates,deltas) = getseqofstates(numstates,numobscases,numsamples,observations)
    realseqofstates = exmodel.seqofstates
    print seqofstates == realseqofstates
    print deltas
    print np.shape(seqofstates)
    print np.shape(realseqofstates)
    print np.shape(deltas)
    print np.concatenate((seqofstates,realseqofstates,deltas),axis = 1)
main()