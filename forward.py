import numpy as np,numpy.random
from init_forward import hmmforward
from scipy import stats
from sklearn.preprocessing import normalize

# def normalize(u):
#     Z = np.sum(u)
#     if Z == 0:
#         return (u,1.0)
#     v = u / Z
#     return (v,Z)


def forward(transmtrx,obsmtrx,pie,observations):
    ''' Input: Transition matrix, pie, state_observation probs, observations
    Output: alphas Probabilites of being in different states at each time point for each sample given the observations till that point, i.e filteing 
    also most likely sequence of staets and its associated probabilies
    Used the equations in Machine learning a probabilistic appraoch, Kevin Murphy
    '''
    # initialization
    if len(np.shape(observations)) == 1 :
        numstates = np.shape(transmtrx)[0]
        timelength = np.shape(observations)[0]
        Zis =  np.empty((timelength,1))
        most_likely_seq = np.empty((timelength,1))
        alphas = np.empty((timelength,numstates))
        phi0 = obsmtrx[:,int(observations[0])]
        (alphas[0,:]) = (np.multiply(phi0,pie)) 
        most_likely_seq[0] = np.argmax(alphas[0,:])
        for t in range(1,timelength):
            phi_t = obsmtrx[:,int(observations[t])]
            alphas[t,:] = np.multiply(phi_t,np.matmul(np.transpose(transmtrx) , np.transpose(alphas[t-1,:])))
            most_likely_seq[t] = np.argmax(alphas[t,:])
        # print "likelihood at this stage is "
        print np.sum(alphas[timelength-1,:])
        for time in range(timelength):
            Zis[time] = np.sum(alphas[time,:])
            alphas[time,:]= normalize(alphas[time,:].reshape(1, -1),norm = 'l1')
            # print alphas[time,:]
        log_prob_most_likely_seq = np.sum(np.log(Zis) + 2.22044604925e-16 )
    else:
        numsamples = np.shape(observations)[0]
        numstates = np.shape(transmtrx)[0]
        timelength = np.shape(observations)[1]
        Zis =  np.empty((numsamples,timelength))
        most_likely_seq = np.empty((numsamples,timelength))
        alphas = np.empty((numsamples,timelength,numstates))
        log_prob_most_likely_seq = np.empty((numsamples,1))
        for sample in range(numsamples):
            phi0 = obsmtrx[:,int(observations[sample,0])]
            alphas[sample,0,:] = np.multiply(phi0,pie)
            most_likely_seq[sample,0] = np.argmax(alphas[sample,0,:])
            for t in range(1,timelength):
                phi_t = obsmtrx[:,int(observations[sample,t])]
                alphas[sample,t,:] = np.multiply(phi_t,np.matmul(np.transpose(transmtrx) , np.transpose(alphas[sample,t-1,:])))
                most_likely_seq[sample,t] = np.argmax(alphas[sample,t,:])
            log_prob_most_likely_seq[sample] = np.sum(np.log(Zis[sample,:]) + 2.22044604925e-16 )
        for sample in range(numsamples):
            print "likelihood at this stage for sample " + str(sample) + "is"
            print np.sum(alphas[sample,timelength-1,:])
            for time in range(timelength):
                Zis[sample,time] = np.sum(alphas[sample,t,:])
                alphas[sample,t,:] = normalize(alphas[sample,t,:].reshape(1, -1) ,norm = 'l1')
    return (alphas,log_prob_most_likely_seq,most_likely_seq,Zis)



# def main():
# small test case
#     exmodel = hmmforward(5,10,1,500)
#     observations = exmodel.observations
#     pie = exmodel.pie
#     transmtrx = exmodel.transitionmtrx
#     obsmtrx = exmodel.obsmtrx
#     seqofstates = exmodel.seqofstates
#     (alphas,log_prob_most_likely_seq,most_likely_seq,,Zis) = forward(transmtrx,obsmtrx,pie,observations)
#     # print np.sum(seqofstates==most_likely_seq) / float(exmodel.obserlength)
#     # print stats.mode(seqofstates)
#     # print stats.mode(most_likely_seq)
# main()