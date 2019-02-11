import pytest
from Baumwelchcont import Baumwelchcont
import numpy as np,numpy.random
from init_gaussian import hmmgaussian
from scipy import stats
from forwardcont import forwardcont
from backwardcont import backwardcont
from forward_backward_cont import forward_backwardcont
from viterbicont import viterbicont
import itertools
from sklearn.preprocessing import normalize
import matplotlib 
from sklearn.cluster import KMeans
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
# np.set_printoptions(precision=4,suppress=True)


# def computeloglikelihoodd(pie,transmtrx,obsmtrx,observations):
#     numobscases = np.shape(obsmtrx)[1]
#     timelength = np.shape(observations)[1]
#     numsamples = np.shape(observations)[0]
#     nstates = np.shape(obsmtrx)[0] 
#     Z = list(itertools.product(range(nstates),repeat = timelength))
#     firstelemsum = 0.0
#     secondelemsum = 0.0
#     thirdelemsum = 0.0
#     for z in Z:
#         thirdelem = 1.0
#         secpmdsum = 0.0
#         thirdsum = 0.0
#         for time in range(1,timelength):
#             secpmdsum += np.log(transmtrx[z[time-1],z[time]])
#             for sample in range(numsamples):
#                 thirdelem *= (float(transmtrx[z[time-1],z[time]]) * obsmtrx[z[time],observations[sample,time]])
#                 thirdsum += np.log(obsmtrx[z[time],observations[sample,time]])
#         # np.prod(transmtrx[z[1]]) extra ?
#         p = float(pie[z[0]] * obsmtrx[z[0],observations[0]] * thirdelem)
#         firstelemsum += float(np.log(pie[z[0]]) * p)
#         for time2 in range(1,timelength):
#             secpmdsum += np.log(transmtrx[z[time2-1],z[time2]]) 
#             thirdsum += np.log(obsmtrx[z[time2],observations[time2]]) 
#         thirdsum += np.log(obsmtrx[z[0],observations[0]])        
#         secondelemsum += float(secpmdsum * p)         
#         thirdelemsum += float(thirdsum * p)
#     print "doosh doosgh"
#     return firstelemsum + secondelemsum + thirdelemsum
@pytest.mark.parametrize("nums,piequality,numobservs,numsamples,Distinctness",[(3,2,10,10, False),(2,2,20,10, True),(4,2,10,100, False),(7,2,5,100, False)])
# ,(4,2,10,100, False),(7,2,5,100, False)

 
def test_Baumwelch(nums,piequality,numobservs,numsamples,Distinctness):
    exmodel = hmmgaussian(nums,piequality,numobservs,numsamples,Distinctness)
    numstates = exmodel.numofstates
    observations = exmodel.observations
    # print "sequence of states is"
    # print exmodel.seqofstates
    hard = False
    sensitivity = 15
    threshold_exponential = 10 ** (-sensitivity)
    (pie,transmtrx,obsmtrx) = Baumwelchcont(observations,numstates,exmodel,hard,threshold_exponential)
    # (pie,transmtrx,obsmtrx) = clipvalues_prevunderflow_small(pie,transmtrx,obsmtrx)
    piedist = np.linalg.norm(pie - exmodel.pie ) / float(numstates)
    transdist = np.linalg.norm(transmtrx - exmodel.transitionmtrx) / float(numstates **2)
    obsdist = np.linalg.norm(obsmtrx - exmodel.obsmtrx) / float( numstates)
    assert abs(np.max(pie) - np.max(exmodel.pie)) < 0.4
    assert abs(np.min(pie) - np.min(exmodel.pie)) < 0.4
    assert abs(np.max(transmtrx) - np.max(exmodel.transitionmtrx))< 0.4
    assert abs(np.min(transmtrx) - np.min(exmodel.transitionmtrx))< 0.4
    assert abs(np.max(obsmtrx[:,1]) - np.max(exmodel.obsmtrx[:,1])) < 2.5
    assert abs(np.min(obsmtrx[:,1]) - np.min(exmodel.obsmtrx[:,1]))< 2.5
    assert abs(np.max(obsmtrx[:,0]) - np.max(exmodel.obsmtrx[:,0])) < 0.8
    assert abs(np.min(obsmtrx[:,0]) - np.min(exmodel.obsmtrx[:,0]))< 0.8
    assert abs(np.sum(pie) - 1) < 0.01
    assert np.sum(np.isnan(pie)) == 0
    assert np.sum(np.isnan(transmtrx)) == 0
    assert np.sum(np.isnan(obsmtrx)) == 0

    for i in range(numstates):
        assert abs(np.sum(transmtrx[i,:]) - 1) < 0.01