import numpy as np,numpy.random
import pytest
from init_forward import hmmforward
import numpy as np

@pytest.fixture
def hmmexample():
    return hmmforward(2,4,1,10)

def test_setting(hmmexample):
    assert hmmexample.numofstates == 2
    assert hmmexample.numofobsercases == 4
    assert hmmexample.piequality == 1
    assert hmmexample.obserlength==10
def test_pie(hmmexample):
    assert len(hmmexample.pie) == 2
    assert np.sum(hmmexample.pie) - 1 < 0.01
def test_obsmtrx(hmmexample):
    assert np.sum(hmmexample.obsmtrx[0,]) -  1 < 0.01
    assert np.sum(hmmexample.obsmtrx[1,]) -  1 < 0.01
    assert len(hmmexample.obsmtrx[0,:]) == 4
def test_transitionmtrx(hmmexample):
    assert np.sum(hmmexample.transitionmtrx[0,])- 1 < 0.01
    assert len(hmmexample.transitionmtrx[0,])==hmmexample.numofstates
def test_observations(hmmexample):
    assert len(hmmexample.observations) == hmmexample.obserlength
    assert np.max(hmmexample.observations) == hmmexample.numofobsercases-1

def test_seqofstates(hmmexample):
    assert len(hmmexample.seqofstates) == hmmexample.obserlength


# class hmmforward(object):
#     def __inint__(self,initnumofstate=5,initnumofobsercases = 10,initpiequality = 1 ,initobserlength = 100):
#         self.numofstates = initnumofstate
#         self.numofobsercases = initnumofobsercases
#         self.piequality = initpiequality
#         self.obserlength = initpiequality
#         self.generatepie()
#         self.generateobsmtrx()
#         self.generatetransitionmtrx()
#         self.generateobservations()
#         # self.transitionmtrxpriors = transitionmtrxpriors  # what to do if i don't have the num of states yet? 
#     def generatepie(self):
#         self.pie = np.random.dirichlet(np.ones(self.numofstates) * self.initpiequality,size=1)[0]
#     def generateobsmtrx(self):
#         self.obsmtrx = np.empty((self.numofstates,self.numofobsercases))
#         self.obsmtrxpriors = np.random.randint(1,self.numofstates)
#         for i in range(self.numofstates):
#             self.obsmtrx[i,:] = np.random.dirichlet(np.ones(self.numofobsercases) / self.obsmtrxpriors[i],size=1)[0]
#     def generatetransitionmtrx(self):
#         self.transitionmtrx = np.empty((self.numofstates,self.numofstates))
#         self.transitionmtrxpriors = np.random.randint(1,self.numofstates)
#         for i in range(self.numofstates):
#             self.transitionmtrx[i,:] = np.random.dirichlet(np.ones(self.numofobsercases) / self.transitionmtrxpriors[i],size=1)[0]
#     def generateobservations(self):
#         self.observations = np.empty((self.obserlength,1))
#         elements = range(self.numofstates)
#         initialstate = np.random.choice(elements, 1, p=self.pie)
#         elements = range(self.numofobsercases)
#         self.observations[0] = np.random.choice(elements, 1, p=self.obsmtrx[initialstate,:])
#         prevstate = initialstate
#         for i in range(1,self.obserlength):
#             elements = range(self.numofstates)
#             nextstate = np.random.choice(elements, 1, p=self.transitionmtrx[prevstate,:])
#             elements = range(self.numofobsercases)
#             self.observations[i] = np.random.choice(elements, 1, p=self.obsmtrx[nextstate,:])
#             prevstate = nextstate

        
            
        
