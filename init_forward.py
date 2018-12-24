import numpy as np,numpy.random


class hmmforward(object):
    ''' Initializes a completely known Discrete observation Discreste states HMM model, prior probablities vector over states,
     transition matrix between states, observation matrix showing probability of observing each of 
     possible observations in each state, generates a sequence of states based on pi and transition
     matrix, and then using that and  observation matrix generates the sequence of observationsself.
     The HMM models is fully specified with its number of states, number of possible observations,
     init pi equality which says how much equality we would like in initial distribution of the states
     ( the higher means less diversity, while a fraction means more diversity and default value is one), 
     and finally number of observations we want to generate from this model.
     '''
    def __init__(self,initnumofstate=5,initnumofobsercases = 10,initpiequality = 1 ,initobserlength = 100):
        self.numofstates = initnumofstate
        self.numofobsercases = initnumofobsercases
        self.piequality = initpiequality
        self.obserlength = initobserlength
        self.generatepie()
        self.generateobsmtrx()
        self.generatetransitionmtrx()
        self.generateobservations()
        # self.transitionmtrxpriors = transitionmtrxpriors  # what to do if i don't have the num of states yet? 
    def generatepie(self):
        self.pie = np.random.dirichlet(np.ones(self.numofstates) * self.piequality,size=1)[0]
    def generateobsmtrx(self):
        self.obsmtrx = np.empty((self.numofstates,self.numofobsercases))
        self.obsmtrxpriors = np.random.randint(1,self.numofstates+1,size = self.numofobsercases)
        for i in range(self.numofstates):
            (self.obsmtrx)[i,:] = np.random.dirichlet(np.ones(self.numofobsercases) / self.obsmtrxpriors[i],size=1)[0]
    def generatetransitionmtrx(self):
        self.transitionmtrx = np.empty((self.numofstates,self.numofstates))
        self.transitionmtrxpriors = np.random.randint(1,self.numofstates+1 ,size = self.numofstates)
        for i in range(self.numofstates):
            (self.transitionmtrx)[i,:] = np.random.dirichlet(np.ones(self.numofstates) / self.transitionmtrxpriors[i],size=1)[0]
    def generateobservations(self):
        self.observations = np.empty((self.obserlength,1),dtype = numpy.int8)
        self.seqofstates = np.empty((self.obserlength,1))
        elements = range(self.numofstates)
        initialstate = np.random.choice(elements, 1, p=self.pie)[0]
        elements = range(self.numofobsercases)
        (self.observations)[0] = np.random.choice(elements, 1, p=list(self.obsmtrx[initialstate,:]))[0]
        prevstate = initialstate
        (self.seqofstates)[0] = initialstate
        for i in range(1,self.obserlength):
            elements = range(self.numofstates)
            nextstate = np.random.choice(elements, 1, p=self.transitionmtrx[prevstate,:])[0]
            elements = range(self.numofobsercases)
            (self.observations)[i] = (np.random.choice(elements, 1, p=list(self.obsmtrx[nextstate,:])))[0]
            (self.seqofstates)[i] = (nextstate)
            prevstate = nextstate
        

# # just testing
# def main():
#     exmodel = hmmforward(5,10,1,80)
#     print "observations"
#     print exmodel.observations
#     print "pi"
#     print exmodel.pie
#     print "transition matrix"
#     print exmodel.transitionmtrx
#     print "observation matrix"
#     print exmodel.obsmtrx
#     print "trans mtrx priors"
#     print exmodel.transitionmtrxpriors
#     print "obsrvationmtrx priors"
#     print exmodel.obsmtrxpriors
#     print "sequence of states"
#     print exmodel.seqofstates
    
# main()

            
        
