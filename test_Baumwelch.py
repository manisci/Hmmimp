import pytest 
import numpy as np,numpy.random
from init_forward import hmmforward
from scipy import stats
from forward import forward
from backward import backward
from forward_backward import forward_backward
from Baumwelch import Baumwelch

@pytest.fixture
def hmmexample():
    exmodel = hmmforward(2,3,1,20,1)
    exmodel.pie = np.array([0.5,0.5])
    exmodel.transitionmtrx = np.array([[.5,.5],[.5,.5]])
    exmodel.obsmtrx = np.array([[.4,.1,.5],[.1,.5,.4]])
    return exmodel
 

def test_Baumwelch(hmmexample):

    numstates = hmmexample.numofstates
    numobscases = hmmexample.numofobsercases
    observations = np.array([2,0,0,2,1,2,1,1,1,2,1,1,1,1,1,2,2,0,0,1])
    # expected values after running the algorithm
    realtransmtrx = np.array([[.69,.31],[.09,.91]])
    realobsmtrx = np.array([[.58 ,.001,.41],[0 , .76,.23]])
    realpie = np.array([1,0])
    # print gammas
    # print most_likely_seq
    (pie,transmtrx,obsmtrx) = Baumwelch(observations,numstates,numobscases,1,hmmexample)
    # assert (np.sum(pie - realpie) / float(numstates))  < 0.01
    print transmtrx
    print realtransmtrx
    print obsmtrx
    print realobsmtrx
    assert np.sum(transmtrx - realtransmtrx) /float(numstates**2) < 0.01 
    assert np.sum(obsmtrx - realobsmtrx) /float(numstates*numobscases) < 0.01 
    # assert (np.max(pie - realpie) )< 0.1
    # assert np.max(transmtrx - realtransmtrx) < 0.1 
    # assert np.max(obsmtrx - realobsmtrx)  < 0.1 

    # (pie,transmtrx,obsmtrx ) =  Baumwelch(observations,numstates,numobscases,numsamples,exmodel)
    # (pie,transmtrx,obsmtrx) = clipvalues_prevunderflow_small(pie,transmtrx,obsmtrx)
    # piedist = np.linalg.norm(pie - exmodel.pie ) / float(numstates)
    # transdist = np.linalg.norm(transmtrx - exmodel.transitionmtrx) / float(numstates **2)
    # obsdist = np.linalg.norm(obsmtrx - exmodel.obsmtrx) / float(numobscases * numstates)
    # print "realpie"
    # print exmodel.pie
    # print pie
    # print "realtrans"
    # print exmodel.transitionmtrx
    # print transmtrx
    # print "real obsmtrx"
    # print exmodel.obsmtrx
    # print obsmtrx
    # print piedist,transdist,obsdist
    # print "dooshag"
def test_Baumwelch2(hmmexample):
    
    numstates = hmmexample.numofstates
    numobscases = hmmexample.numofobsercases
    observations = hmmexample.observations
    # expected values after running the algorithm
    realtransmtrx = hmmexample.transitionmtrx
    realobsmtrx = hmmexample.obsmtrx
    realpie = hmmexample.pie
    # print gammas
    # print most_likely_seq
    (pie,transmtrx,obsmtrx) = Baumwelch(observations,numstates,numobscases,1,hmmexample)
    # assert (np.sum(pie - realpie) / float(numstates))  < 0.01
    assert np.sum(transmtrx - realtransmtrx) /float(numstates**2) < 0.01 
    assert np.sum(obsmtrx - realobsmtrx) /float(numstates*numobscases) < 0.01 
    print transmtrx
    print realtransmtrx
    print obsmtrx
    print realobsmtrx
    # assert (np.max(pie - realpie) )< 0.1
    # assert np.max(transmtrx - realtransmtrx) < 0.2
    # assert np.max(obsmtrx - realobsmtrx)  < 0.2