import numpy as np,numpy.random
from init_gaussian import hmmgaussian
from scipy import stats
from forwardcont import forwardcont
from backwardcont import backwardcont
from forward_backward_cont import forward_backwardcont
from Baumwelchcont import Baumwelchcont
from viterbicont import viterbicont
 

def getseqofstatescont(numstates,observations,exmodel):
    ''' returns most probable seq of states and probability of 
    being in each state for each timestep, as a second argument of size numtimesteps * numstates
    Input : numstates,numobscases,numsamples,observations,exmodel
    Output: Most likely sequence of state, Optimal Z's and probabilites of being in each state for 
    each time step. 
    Exmodel may only be used for the close to reality initialization and is optional

    '''
    # print gammas
    # print most_likely_seq
    hard = False
    sensitivity = 5
    threshold_exponential = 10 ** (-sensitivity)
    (pie,transmtrx,obsmtrx,likelihood) = Baumwelchcont(observations,numstates,exmodel,hard,threshold_exponential)
    (seq_states ,deltas)= viterbicont(transmtrx,obsmtrx,pie,observations)
    return (seq_states,deltas,likelihood,pie,transmtrx,obsmtrx)
# def main():
#     numstates = 2
#     numsamples = 100
#     numbofobsrv = 10
#     exmodel = hmmgaussian(numstates,1,numbofobsrv,numsamples,2,True)
#     observations = exmodel.observations
#     (seqofstates,deltas) = getseqofstatescont(numstates,numsamples,observations,exmodel)
#     realseqofstates = exmodel.seqofstates
#     # print np.sum(seqofstates[0,:] == realseqofstates[0,:])
#     print "I got this much of the sequence right!"
#     print np.shape(realseqofstates)
#     print np.shape(seqofstates)
#     NoCorstates = 0
#     if numsamples > 1:
#         for sample in range(numsamples):
#             NoCorstates += float(np.sum(seqofstates[sample,:] == realseqofstates[sample,:])) 
#     else:
#         NoCorstates += float(np.sum(seqofstates == realseqofstates))         
#     print float(NoCorstates) /float(numsamples * numbofobsrv)
#     # print deltas
#     # print np.shape(seqofstates)
#     # print np.shape(realseqofstates)
#     # print np.shape(deltas)
#     # print np.concatenate((seqofstates,realseqofstates,deltas),axis = 1)
# main()