import numpy as np,numpy.random
from init_gaussian import hmmgaussian
from scipy import stats
from sklearn.preprocessing import normalize
def clipvalues_prevunderflowfw(vector):
    eps = 2.22044604925e-16
    minpie = np.min(vector)
    vector[np.argmin(vector)] = eps
    for i in range(np.shape(vector)[0]):
        if vector[i] < eps:
            vector[i] = eps +( vector[i] - minpie)
        if vector[i] == 0:
            vector[i] =eps    
    return vector
# def clipvalues_prevoverflowfw(vector):
#     eps = 2.22044604925e-16
#     minpie = np.min(vector)
#     maxpie = np.max(vector)
#     vector[np.argmin(vector)] = eps
#     vector[np.argmax(vector)] = 1.0
#     for i in range(np.shape(vector)[0]):
#         if vector[i] < eps:
#             vector[i] = eps +( vector[i] - minpie)
#         if vector[i] > 1:
#             vector[i] = 1.0 - (maxpie - vector[i])
#         if vector[i] == 0:
#             vector[i] =eps 
       
#     return vector


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
        phi0 = eps * np.ones(numstates)
        sortedprobs = (sorted(observations))
        probeps = 0.1 * (sortedprobs[1] - sortedprobs[0])
        for state in range(numstates):
            distr = stats.norm(obsmtrx[state,0], obsmtrx[state,1])
            phi0[state] = distr.cdf(observations[0]+probeps) - distr.cdf(observations[0]- probeps)
        (alphas[0,:]) = (np.multiply(phi0,pie)) 
        alphas[0,:] = clipvalues_prevunderflowfw(alphas[0,:])
        most_likely_seq[0] = np.argmax(alphas[0,:])
        for t in range(1,timelength):
            phi_t = eps * np.ones(numstates)
            for state in range(numstates):
                distr = stats.norm(obsmtrx[state,0], obsmtrx[state,1])
                phi_t[state] = distr.cdf(observations[t]+ probeps) - distr.cdf(observations[t] - probeps)
                # print observations[t]
                # print phi_t[state]
            # print 'trouble making probs'
            # print phi_t
            alphas[t,:] = np.multiply(phi_t,np.matmul(np.transpose(transmtrx) , np.transpose(alphas[t-1,:])))
            most_likely_seq[t] = np.argmax(alphas[t,:])
            alphas[t,:] = clipvalues_prevunderflowfw(alphas[t,:])
        # print "likelihood at this stage is "
        # print alphas[timelength-1,:]
        logobservations = np.sum(alphas[timelength-1,:])
        # print logobservations
        for time in range(timelength):
            suspect = np.sum(alphas[time,:])
            Zis[time] = suspect
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
        logobservations = eps *  np.ones(numsamples)
        for sample in range(numsamples):
            sortedprobs = (sorted(observations[sample,:]))
            probeps = 0.1 * (sortedprobs[1] - sortedprobs[0])
            phi0 = eps * np.ones(numstates)
            for state in range(numstates):
                distr = stats.norm(obsmtrx[state,0], obsmtrx[state,1])
                phi0[state] = distr.cdf(observations[sample,0]+probeps) - distr.cdf(observations[sample,0]- probeps)
            alphas[sample,0,:] = np.multiply(phi0,pie)
            alphas[sample,0,:] = clipvalues_prevunderflowfw(alphas[sample,0,:])

            most_likely_seq[sample,0] = np.argmax(alphas[sample,0,:])
            for t in range(1,timelength):
                phi_t = eps * np.ones(numstates)
                for state in range(numstates):
                    distr = stats.norm(obsmtrx[state,0], obsmtrx[state,1])
                    phi_t[state] = distr.cdf(observations[sample,t]+probeps) - distr.cdf(observations[sample,t]- probeps)
                alphas[sample,t,:] = np.multiply(phi_t,np.matmul(np.transpose(transmtrx) , np.transpose(alphas[sample,t-1,:])))
                most_likely_seq[sample,t] = np.argmax(alphas[sample,t,:])
                alphas[sample,t,:] = clipvalues_prevunderflowfw(alphas[sample,t,:])
            log_prob_most_likely_seq[sample] = np.sum(np.log(Zis[sample,:]) + 2.22044604925e-16 )
        for sample in range(numsamples):
            # print "likelihood at this stage for sample " + str(sample) + "is"
            logobservations[sample] = np.sum(alphas[sample,timelength-1,:])
            # print logobservations[sample]
            for time in range(timelength):
                Zis[sample,time] = np.sum(alphas[sample,t,:])
                alphas[sample,t,:] = normalize(alphas[sample,t,:].reshape(1, -1) ,norm = 'l1')
    # print "dear Ziees"
    # print Zis
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