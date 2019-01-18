import numpy as np,numpy.random
from init_forward import hmmforward
from scipy import stats
from forward import forward
from backward import backward
from forward_backward import forward_backward
from Baumwelch import Baumwelch
np.set_printoptions(precision=4,suppress=True)


def main():
    # exmodel = hmmforward(2,3,1,20,1)
    # exmodel.pie = np.array([0.5,0.5])
    # exmodel.transitionmtrx = np.array([[.5,.5],[.5,.5]])
    # exmodel.obsmtrx = np.array([[.4,.1,.5],[.1,.5,.4]])
    numstates = 2
    numobscases = 3
    # numstates = exmodel.numofstates
    # numobscases = exmodel.numofobsercases
    observations = np.array([2,0,0,2,1,2,1,1,1,2,1,1,1,1,1,2,2,0,0,1])
    # transmtrx = exmodel.transitionmtrx
    # obsmtrx = exmodel.obsmtrx
    # pie = exmodel.pie
    # (gammas,betas,alphas,log_prob_most_likely_seq,most_likely_seq,forward_most_likely_seq,forward_log_prob_most_likely_seq,Ziis) = forward_backward(transmtrx,obsmtrx,pie,observations)
    # print gammas
    # print most_likely_seq
    (pie,transmtrx,obsmtrx) = Baumwelch(observations,numstates,numobscases)
    # print transmtrx
    # print exmodel.transitionmtrx
    # print obsmtrx
    # print exmodel.obsmtrx
    # (pie,transmtrx,obsmtrx ) =  Baumwelch(observations,numstates,numobscases,numsamples,exmodel)
    # (pie,transmtrx,obsmtrx) = clipvalues_prevunderflow_small(pie,transmtrx,obsmtrx)
    # piedist = np.linalg.norm(pie - exmodel.pie ) / float(numstates)
    # transdist = np.linalg.norm(transmtrx - exmodel.transitionmtrx) / float(numstates **2)
    # obsdist = np.linalg.norm(obsmtrx - exmodel.obsmtrx) / float(numobscases * numstates)
    # print "realpie"
    # # print exmodel.pie
    # print pie
    # print "realtrans"
    # # print exmodel.transitionmtrx
    print transmtrx
    # print "real obsmtrx"
    # # print exmodel.obsmtrx
    print obsmtrx
    # print piedist,transdist,obsdist
    # print "dooshag"
main()