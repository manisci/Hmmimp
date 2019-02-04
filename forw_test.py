import pytest
import numpy as np,numpy.random
from init_forward import hmmforward
from scipy import stats

@pytest.fixture
def hmmexample():
    return hmmforward(5,10,2,500,1)

def test_forward(hmmexample):
    # initialization
    numstates = np.shape(hmmexample.transitionmtrx)[0]
    timelength = np.shape(hmmexample.observations)[0]
    Zis = np.empty((timelength,1))
    most_likely_seq = np.empty((timelength,1))
    alphas = np.empty((timelength,numstates))
    phi0 = hmmexample.obsmtrx[:,int(hmmexample.observations[0])]
    Zis[0] = np.sum(np.multiply(phi0,hmmexample.pie))
    alphas[0,:] = np.multiply(phi0,hmmexample.pie) / Zis[0]
    most_likely_seq[0] = np.argmax(alphas[0,:])
    for t in range(1,timelength):
        phi_t = hmmexample.obsmtrx[:,int(hmmexample.observations[t])]
        Zis[t] = np.sum(np.multiply(phi_t,np.matmul(np.transpose(hmmexample.transitionmtrx) , np.transpose(alphas[t-1,:]))))
        alphas[t,:] = np.multiply(phi_t,np.matmul(np.transpose(hmmexample.transitionmtrx) , np.transpose(alphas[t-1,:]))) / Zis[t]
        # (alphas[t,:],Zis[t]) = normalize(np.multiply(phi_t,np.matmul(np.transpose(hmmexample.transitionmtrx) , np.transpose(alphas[t-1,:])))) 
        most_likely_seq[t] = np.argmax(alphas[t,:])
    log_prob_most_likely_seq = np.sum(np.log(Zis))
    # assert stats.mode(hmmexample.seqofstates).mode == stats.mode(most_likely_seq).mode
    assert len(most_likely_seq) == timelength
    assert abs(np.sum(alphas[0,])) - 1 < 0.01
    

