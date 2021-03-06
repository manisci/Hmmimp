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
        phi0 = eps * np.ones(numstates)
        for state in range(numstates):
            distr = stats.norm(obsmtrx[state,0], obsmtrx[state,1])
            phi0[state] = distr.pdf(observations[0])
        denom = float(np.inner(phi0,pie))
        Zis[0] = np.sum(phi0)
        for state in range(numstates):
            phi0[state] = ( pie[state] * phi0[state]) / denom
        # (alphas[0,:]) = (np.multiply(phi0,pie)) 
        (alphas[0,:]) = phi0
        most_likely_seq[0] = np.argmax(alphas[0,:])
        alphas[0,:]= normalize(alphas[0,:].reshape(1, -1),norm = 'l1')
        for t in range(1,timelength):
            phi_t = eps * np.ones(numstates)
            for state in range(numstates):
                distr = stats.norm(obsmtrx[state,0], obsmtrx[state,1])
                phi_t[state] = distr.pdf(observations[t])
            alphas[t,:] = np.multiply(phi_t,np.matmul(np.transpose(transmtrx) , np.transpose(alphas[t-1,:])))
            Zis[t] = np.sum(alphas[t,:])
            if t == timelength-1:
                logobservations = np.sum(alphas[timelength-1,:])
            alphas[t,:] /= Zis[t] 
            most_likely_seq[t] = np.argmax(alphas[t,:])
            alphas[t,:]= normalize(alphas[t,:].reshape(1, -1),norm = 'l1')
        print "likelihood at this stage is "
        print logobservations
            # print alphas[time,:]
        log_prob_most_likely_seq = np.sum(np.log(Zis) + 2.22044604925e-16 )
    elif len(np.shape(observations)) == 2:
        numsamples = np.shape(observations)[0]
        numstates = np.shape(transmtrx)[0]
        timelength = np.shape(observations)[1]
        Zis =  eps * np.ones((numsamples,timelength))
        most_likely_seq = eps * np.ones((numsamples,timelength))
        alphas = eps * np.ones((numsamples,timelength,numstates))
        log_prob_most_likely_seq = eps * np.ones((numsamples))
        logobservations = eps *  np.ones(numsamples)
        for sample in range(numsamples):
            phi0 = eps * np.ones(numstates)
            for state in range(numstates):
                distr = stats.norm(obsmtrx[state,0], obsmtrx[state,1])
                phi0[state] = distr.pdf(observations[sample,0])
            alphas[sample,0,:] = np.multiply(phi0,pie)
            most_likely_seq[sample,0] = np.argmax(alphas[sample,0,:])
            for t in range(1,timelength):
                phi_t = eps * np.ones(numstates)
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
                Zis[sample,time] = np.sum(alphas[sample,time,:])
                alphas[sample,time,:] = normalize(alphas[sample,time,:].reshape(1, -1) ,norm = 'l1')
    else:
        numsamples = np.shape(observations)[0]
        numstates = np.shape(transmtrx)[0]
        numfeats = np.shape(observations)[1]
        timelength = np.shape(observations)[2]
        Zis =  eps * np.ones((numsamples,timelength))
        most_likely_seq = eps * np.ones((numsamples,timelength))
        alphas = eps * np.ones((numsamples,timelength,numstates))
        log_prob_most_likely_seq = eps * np.ones((numsamples))
        logobservations = eps *  np.ones(numsamples)
        # print obsmtrx
        for sample in range(numsamples):
            phi0 =  np.ones(numstates)
            for state in range(numstates):
                for feat in range(numfeats):
                    distr = stats.norm(obsmtrx[feat,state,0], obsmtrx[feat,state,1])
                    phi0[state] *= distr.pdf(observations[sample,feat,0])
            alphas[sample,0,:] = np.multiply(phi0,pie)
            most_likely_seq[sample,0] = np.argmax(alphas[sample,0,:])
            for t in range(1,timelength):
                phi_t = np.ones(numstates)
                # print obsmtrx[feat,state,0]
                for state in range(numstates):
                    # print obsmtrx[feat,state,1]
                    for feat in range(numfeats):
                        distr = stats.norm(obsmtrx[feat,state,0], obsmtrx[feat,state,1])
                        phi_t[state] *= distr.pdf(observations[sample,feat,t])
                        # print "mean is"
                        # print obsmtrx[feat,state,0]
                        # print "variance is"
                        # print obsmtrx[feat,state,1]
                        # print "probability is"
                        # print distr.pdf(observations[sample,feat,t])
                        # print observations[sample,feat,t]
                        # print phi_t[state]
                alphas[sample,t,:] = np.multiply(phi_t,np.matmul(np.transpose(transmtrx) , np.transpose(alphas[sample,t-1,:])))
                most_likely_seq[sample,t] = np.argmax(alphas[sample,t,:])
            log_prob_most_likely_seq[sample] = np.sum(np.log(Zis[sample,:]) + 2.22044604925e-16 )
        for sample in range(numsamples):
            # print "likelihood at this stage for sample " + str(sample) + "is"
            logobservations[sample] = np.sum(alphas[sample,timelength-1,:])
            # print logobservations[sample]
            for time in range(timelength):
                Zis[sample,time] = np.sum(alphas[sample,time,:])
                # print alphas[sample,t,:]
                alphas[sample,time,:] = normalize(alphas[sample,time,:].reshape(1, -1) ,norm = 'l1')
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