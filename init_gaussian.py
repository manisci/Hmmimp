import numpy as np,numpy.random


class hmmgaussian(object):
    ''' Initializes a completely known continuous observation Discreste states HMM model, prior probablities vector over states,
     transition matrix between states, observation matrix of size numstates by 2 showing mean and variance of the gaussian in each state, generates a sequence of states based on pi and transition
     matrix, and then using that and  observation mus and sigmas generates the sequence of observations itself.
     The HMM models is fully specified with its number of states, number of possible observations,
     init pi equality which says how much equality we would like in initial distribution of the states
     ( the higher means less diversity, while a fraction means more diversity and default value is one) and desired number of samples, 
     and finally length of indivdual samples we want to generate from this model.
    '''

    
    def __init__(self,initnumofstate=5,initpiequality = 1 ,initobserlength = 10, initnumsamples = 1):
        self.numofstates = initnumofstate
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
        self.obsmtrx = 2.22044604925e-16 * np.ones((self.numofstates,2))
        self.obsmtrxmeanpriors = np.random.permutation(range(1,self.numofstates+1))
        self.obsmtrxvarpriors = np.random.permutation(range(1,self.numofstates+1))
        for i in range(self.numofstates):
            (self.obsmtrx)[i,0] = np.random.normal(1.0 / self.obsmtrxmeanpriors[i],1.0 / self.obsmtrxvarpriors[i],size=1)
            (self.obsmtrx)[i,1] = abs(np.random.normal(self.obsmtrxmeanpriors[i] + (1.0 / self.obsmtrxmeanpriors[i]), self.obsmtrxvarpriors[i] + (1.0 / self.obsmtrxvarpriors[i]),size=1))
    def generatetransitionmtrx(self):
        # used dirchlet distribution adding up to one, for probabiliteis of transition in a state
        self.transitionmtrx = 2.22044604925e-16 * np.ones((self.numofstates,self.numofstates))
        self.transitionmtrxpriors = np.random.permutation(range(1,self.numofstates+1))
        for i in range(self.numofstates):
            (self.transitionmtrx)[i,:] = np.random.dirichlet(np.ones(self.numofstates) / self.transitionmtrxpriors[i],size=1)
    def generateobservations(self):
        # uses numsamples, numobscases, time length, and initial probability and transition matrix and obsmtrx to generate sequence of states and observations
        if self.numsamples == 1:
            self.observations = 2.22044604925e-16 *np.ones((self.obserlength),dtype = numpy.int8)
            self.seqofstates = 2.22044604925e-16 *np.ones((self.obserlength))
            # available choices for states and setting the initial state based on pie
            elements = range(self.numofstates)
            initialstate = np.random.choice(elements, 1, p=self.pie)[0]
            # available choices for observations
            # setting first observation 
            (self.observations)[0] = np.random.normal((self.obsmtrx)[initialstate,0],(self.obsmtrx)[initialstate,1])
            prevstate = initialstate
            (self.seqofstates)[0] = initialstate
            for i in range(1,self.obserlength):
                # choosing next state based on the transition matrix
                elements = range(self.numofstates)
                nextstate = np.random.choice(elements, 1, p=self.transitionmtrx[prevstate,:])[0]
                # choosing next observation based on the new state
                (self.observations)[i] = np.random.normal((self.obsmtrx)[nextstate,0],(self.obsmtrx)[nextstate,1])
                (self.seqofstates)[i] = (nextstate)
                prevstate = nextstate
        else:
            self.observations = 2.22044604925e-16 * np.ones((self.numsamples,self.obserlength),dtype = numpy.int8)
            self.seqofstates = 2.22044604925e-16 * np.ones((self.numsamples,self.obserlength))
            for samnum in range(self.numsamples):
                elements = range(self.numofstates)
                initialstate = np.random.choice(elements, 1, p=self.pie)[0]
                elements = range(self.numofobsercases)
                (self.observations)[samnum,0] = np.random.normal((self.obsmtrx)[initialstate,0],(self.obsmtrx)[initialstate,1])
                prevstate = initialstate
                (self.seqofstates)[samnum,0] = initialstate
                for i in range(1,self.obserlength):
                    elements = range(self.numofstates)
                    nextstate = np.random.choice(elements, 1, p=self.transitionmtrx[prevstate,:])[0]
                    elements = range(self.numofobsercases)
                    (self.observations)[samnum,i] = np.random.normal((self.obsmtrx)[nextstate,0],(self.obsmtrx)[nextstate,1])
                    (self.seqofstates)[samnum,i] = (nextstate)
                    prevstate = nextstate
        
# just testing
# def main():
#     exmodel = hmmgaussian(3,1,20,1)
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
#     print exmodel.obsmtrxmeanpriors
#     print exmodel.obsmtrxvarpriors
#     print "sequence of states"
#     print exmodel.seqofstates
    
# main()

            
        
