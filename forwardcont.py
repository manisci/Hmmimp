import numpy as np,numpy.random
from init_gaussian import hmmgaussian
from scipy import stats
from sklearn.preprocessing import normalize



def forwardcont(transmtrx,obsmtrx,pie,observations):
    ''' Input: Transition matrix, pie, state_observation probs, observations
    Output: alphas Probabilites of being in different states at each time point for each sample given the observations till that point, i.e filteing 
    also most likely sequence of staets and its associated probabilies
    Used the equations in Machine learning a probabilistic appraoch, Kevin Murphy
    '''
    eps = 2.22044605e-16
    # initialization
    if len(np.shape(observations)) == 1 :
        numstates = np.shape(transmtrx)[0]
        timelength = np.shape(observations)[0]
        Zis =  eps * np.ones((timelength))
        most_likely_seq = eps * np.ones((timelength))
        alphas = eps * np.ones((timelength,numstates))
        phi0 = eps * np.empty(numstates)
        for state in range(numstates):
            distr = stats.norm(obsmtrx[state,0], obsmtrx[state,1])
            phi0[state] = distr.pdf(observations[0])
        (alphas[0,:]) = (np.multiply(phi0,pie)) 
        most_likely_seq[0] = np.argmax(alphas[0,:])
        for t in range(1,timelength):
            phi_t = eps * np.empty(numstates)
            for state in range(numstates):
                distr = stats.norm(obsmtrx[state,0], obsmtrx[state,1])
                phi_t[state] = distr.pdf(observations[t])
            alphas[t,:] = np.multiply(phi_t,np.matmul(np.transpose(transmtrx) , np.transpose(alphas[t-1,:])))
            most_likely_seq[t] = np.argmax(alphas[t,:])
        # print "likelihood at this stage is "
        logobservations = np.sum(alphas[timelength-1,:])
        # print logobservations
        for time in range(timelength):
            Zis[time] = np.sum(alphas[time,:])
            alphas[time,:]= normalize(alphas[time,:].reshape(1, -1),norm = 'l1')
            # print alphas[time,:]
        log_prob_most_likely_seq = np.sum(np.log(Zis) + 2.22044604925e-16 )
    else:
        numsamples = np.shape(observations)[0]
        numstates = np.shape(transmtrx)[0]
        timelength = np.shape(observations)[1]
        Zis =  eps * np.ones((numsamples,timelength))
        most_likely_seq = eps * np.ones((numsamples,timelength))
        alphas = eps * np.ones((numsamples,timelength,numstates))
        log_prob_most_likely_seq = eps * np.ones((numsamples))
        logobservations = np.empty(numsamples)
        for sample in range(numsamples):
            phi0 = eps * np.empty(numstates)
            for state in range(numstates):
                distr = stats.norm(obsmtrx[state,0], obsmtrx[state,1])
                phi0[state] = distr.pdf(observations[sample,0])
            alphas[sample,0,:] = np.multiply(phi0,pie)
            most_likely_seq[sample,0] = np.argmax(alphas[sample,0,:])
            for t in range(1,timelength):
                phi_t = eps * np.empty(numstates)
                for state in range(numstates):
                    distr = stats.norm(obsmtrx[state,0], obsmtrx[state,1])
                    phi_t[state] = distr.pdf(observations[sample,t])
                alphas[sample,t,:] = np.multiply(phi_t,np.matmul(np.transpose(transmtrx) , np.transpose(alphas[sample,t-1,:])))
                most_likely_seq[sample,t] = np.argmax(alphas[sample,t,:])
            log_prob_most_likely_seq[sample] = np.sum(np.log(Zis[sample,:]) + 2.22044604925e-16 )
        for sample in range(numsamples):
            # print "likelihood at this stage for sample " + str(sample) + "is"
            logobservations[sample] = np.sum(alphas[sample,timelength-1,:])
            # print logobservations[sample]
            for time in range(timelength):
                Zis[sample,time] = np.sum(alphas[sample,t,:])
                alphas[sample,t,:] = normalize(alphas[sample,t,:].reshape(1, -1) ,norm = 'l1')
    return (alphas,log_prob_most_likely_seq,most_likely_seq,Zis,logobservations)



# def main():
#     exmodel = hmmgaussian(3,1,50,1)
#     observations = exmodel.observations
#     pie = exmodel.pie
#     transmtrx = exmodel.transitionmtrx
#     obsmtrx = exmodel.obsmtrx
#     seqofstates = exmodel.seqofstates
#     (alphas,log_prob_most_likely_seq,most_likely_seq,Zis,logobservations) = forward(transmtrx,obsmtrx,pie,observations)
#     print np.sum(seqofstates==most_likely_seq) / float(exmodel.obserlength)
#     # print stats.mode(seqofstates)
#     # print stats.mode(most_likely_seq)
# main()