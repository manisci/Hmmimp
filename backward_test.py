import pytest
import numpy as np,numpy.random
from init_forward import hmmforward
from scipy import stats

# @pytest.fixture
# def hmmexample():
#     return hmmforward(5,10,2,500)

@pytest.mark.parametrize("nums,numobscase,piequality,numobservs",[(5,10,2,500),(2,4,1,500),[8,20,1,1000]])

def test_backward(nums,numobscase,piequality,numobservs):
    # initialization
    hmmexample = hmmforward(nums,numobscase,piequality,numobservs)
    numstates = np.shape(hmmexample.transitionmtrx)[0]
    timelength = np.shape(hmmexample.observations)[0]
    betas = np.empty((timelength,numstates))
    betas[timelength-1,:] = np.ones((1,numstates))
    for t in range(timelength-1,0,-1):
        phi_t = hmmexample.obsmtrx[:,int(hmmexample.observations[t])]
        betas[t-1,:] = np.matmul(hmmexample.transitionmtrx,np.multiply(phi_t , (betas[t,:]))) 
    assert np.sum(betas[timelength-2,]) > np.sum(betas[0,])
    assert abs(np.mean(betas[timelength-1,])) - 1 < 0.01
    