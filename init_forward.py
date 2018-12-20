import numpy as np,numpy.random


class hmmforward(object):
    def __inint__(self,initnumofstate=5,initnumofobsercases = 10,initpiequality = 1 ,initobserlength = 100):
        self.numofstates = initnumofstate
        self.numofobsercases = initnumofobsercases
        self.piequality = initpiequality
        self.obserlength = initpiequality
        # self.transitionmtrxpriors = transitionmtrxpriors  # what to do if i don't have the num of states yet? 
    def generatepie(self):
        self.pie = np.random.dirichlet(np.ones(self.numofstates) * self.initpiequality,size=1)[0]
    def generateobsmtrx(self):
        self.obsmtrx = np.empty((self.numofstates,self.numofobsercases))
        self.obsmtrxpriors = np.random.randint(1,self.numofstates)
        for i in range(self.numofstates):
            self.obsmtrx[i,:] = np.random.dirichlet(np.ones(self.numofobsercases) / self.obsmtrxpriors[i],size=1)[0]
    def generatetransitionmtrx(self):
        self.transitionmtrx = np.empty((self.numofstates,self.numofstates))
        self.transitionmtrxpriors = np.random.randint(1,self.numofstates)
        for i in range(self.numofstates):
            self.transitionmtrx[i,:] = np.random.dirichlet(np.ones(self.numofobsercases) / self.transitionmtrxpriors[i],size=1)[0]
    def generateobservations(self):
        elements = range(self.numofstates)
        initialstate = np.random.choice(elements, 1, p=self.pie)
        
            
        
