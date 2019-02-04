import numpy as np,numpy.random


class hmmforward(object):
    ''' Initializes a completely known Discrete observation Discreste states HMM model, prior probablities vector over states,
     transition matrix between states, observation matrix showing probability of observing each of 
     possible observations in each state, generates a sequence of states based on pi and transition
     matrix, and then using that and  bservation matrix generates the sequence of observations itself.
     The HMM models is fully specified with its number of states, number of possible observations,
     init pi equality which says how much equality we would like in initial distribution of the states
     ( the higher means less diversity, while a fraction means more diversity and default value is one) and desired number of samples, 
     and finally length of indivdual samples we want to generate from this model.

    You can convert the observations and obsmtrx all together into one matrix
    called soft evidence which is a K * T matrix by using the corresponding
    distribution across all the states for each time point and use that instead
    '''

    
    def __init__(self,initnumofstate=5,initnumofobsercases = 10,initpiequality = 1 ,initobserlength = 100, initnumsamples = 1):
        self.numofstates = initnumofstate
        self.numofobsercases = initnumofobsercases
        self.piequality = initpiequality
        self.obserlength = initobserlength
        self.numsamples = initnumsamples
        self.generatepie()
        self.generateobsmtrx()
        self.generatetransitionmtrx()
        self.generateobservations()
        # 2.22044604925e-16 = 2.22044604925e-16
        # self.transitionmtrxpriors = transitionmtrxpriors  # what to do if i don't have the num of states yet? 
    def generatepie(self):
        self.pie = np.random.dirichlet(np.ones(self.numofstates) * self.piequality,size=1)[0]
    def generateobsmtrx(self):
        # used dirchlet distribution adding up to one for probabiliteis of  obervations in a single state
        self.obsmtrx = 2.22044604925e-16 * np.ones((self.numofstates,self.numofobsercases))
        self.obsmtrxpriors = np.random.randint(1,self.numofstates+1,size = self.numofobsercases)
        for i in range(self.numofstates):
            (self.obsmtrx)[i,:] = np.random.dirichlet(np.ones(self.numofobsercases) / self.obsmtrxpriors[i],size=1)[0]
    def generatetransitionmtrx(self):
        # used dirchlet distribution adding up to one, for probabiliteis of transition in a state
        self.transitionmtrx = 2.22044604925e-16 * np.ones((self.numofstates,self.numofstates))
        self.transitionmtrxpriors = np.random.randint(1,self.numofstates+1 ,size = self.numofstates)
        for i in range(self.numofstates):
            (self.transitionmtrx)[i,:] = np.random.dirichlet(np.ones(self.numofstates) / self.transitionmtrxpriors[i],size=1)[0]
    def generateobservations(self):
        # uses numsamples, numobscases, time length, and initial probability and transition matrix and obsmtrx to generate sequence of states and observations
        if self.numsamples == 1:
            self.observations = 2.22044604925e-16 *np.ones((self.obserlength),dtype = numpy.int8)
            self.seqofstates = 2.22044604925e-16 *np.ones((self.obserlength))
            # available choices for states and setting the initial state based on pie
            elements = range(self.numofstates)
            initialstate = np.random.choice(elements, 1, p=self.pie)[0]
            # available choices for observations
            elements = range(self.numofobsercases)
            # setting first observation 
            (self.observations)[0] = np.random.choice(elements, 1, p=list(self.obsmtrx[initialstate,:]))[0]
            prevstate = initialstate
            (self.seqofstates)[0] = initialstate
            for i in range(1,self.obserlength):
                # choosing next state based on the transition matrix
                elements = range(self.numofstates)
                nextstate = np.random.choice(elements, 1, p=self.transitionmtrx[prevstate,:])[0]
                # choosing next observation based on the new state
                elements = range(self.numofobsercases)
                (self.observations)[i] = (np.random.choice(elements, 1, p=list(self.obsmtrx[nextstate,:])))[0]
                (self.seqofstates)[i] = (nextstate)
                prevstate = nextstate
        else:
            self.observations = 2.22044604925e-16 * np.ones((self.numsamples,self.obserlength),dtype = numpy.int8)
            self.seqofstates = 2.22044604925e-16 * np.ones((self.numsamples,self.obserlength))
            for samnum in range(self.numsamples):
                elements = range(self.numofstates)
                initialstate = np.random.choice(elements, 1, p=self.pie)[0]
                elements = range(self.numofobsercases)
                (self.observations)[samnum,0] = np.random.choice(elements, 1, p=list(self.obsmtrx[initialstate,:]))[0]
                prevstate = initialstate
                (self.seqofstates)[samnum,0] = initialstate
                for i in range(1,self.obserlength):
                    elements = range(self.numofstates)
                    nextstate = np.random.choice(elements, 1, p=self.transitionmtrx[prevstate,:])[0]
                    elements = range(self.numofobsercases)
                    (self.observations)[samnum,i] = (np.random.choice(elements, 1, p=list(self.obsmtrx[nextstate,:])))[0]
                    (self.seqofstates)[samnum,i] = (nextstate)
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

            
        
