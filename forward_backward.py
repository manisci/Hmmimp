import numpy as np,numpy.random
from init_forward import hmmforward
from scipy import stats
from forward import forward
from backward import backward
from sklearn.preprocessing import normalize
# from sklearn.preprocessing import normalize

def clipmatrix(mtrx):
    # preventing underflow ?
    eps = 2.22044604925e-16
    minpie = np.min(mtrx)
    minarg = np.unravel_index(np.argmin(mtrx, axis=None), mtrx.shape)
    mtrx[minarg] = eps
    if len(np.shape(mtrx)) == 2:
        for i in range(np.shape(mtrx)[0]):
            for j in range(np.shape(mtrx)[1]):
                if mtrx[i,j] < eps:
                    mtrx[i,j] = eps +( mtrx[i,j] - minpie)
                # if mtrx[i,j] > 1:
                #     mtrx[i,j] = 1.0
                if mtrx[i,j] == 0:
                    mtrx[i,j] = eps
    elif len(np.shape(mtrx)) == 3 :
        for i in range(np.shape(mtrx)[0]):
            for j in range(np.shape(mtrx)[1]):
                for k in range(np.shape(mtrx)[2]):
                    if mtrx[i,j,k] < eps:
                        mtrx[i,j,k] = eps +( mtrx[i,j,k] - minpie)
                    # if mtrx[i,j,k] > 1:
                    #     mtrx[i,j,k] = 1.0
                    if mtrx[i,j,k] == 0:
                        mtrx[i,j,k] = eps
    return mtrx
# def normalize(u):
#     Z = np.sum(u)
#     if Z==0:
#         return (u,1.0)
#     else:
#         v = u / Z
#     return (v,Z)


def clipvalues_prevunderflowfw(alphas,betas,gammas,Ziis):
    eps = 2.22044604925e-16
    minpie = np.min(Ziis)
    Ziis[np.argmin(Ziis)] = eps
    for i in range(np.shape(Ziis)[0]):
        if Ziis[i] < eps:
            Ziis[i] = eps +( Ziis[i] - minpie)
        if Ziis[i] > 1:
            Ziis[i] = 1.0
        if Ziis[i] == 0:
            Ziis[i] =eps    
    alphas = clipmatrix(alphas)
    betas = clipmatrix(betas)
    gammas = clipmatrix(gammas)
    return (alphas,betas,gammas,Ziis)

def forward_backward(transmtrx,obsmtrx,pie,observations):
    ''' Input : Transition matrix, pie, state_observation probs, observations
    Output : alphas, betas, gammas -->  Probablities of being at different states, at each timepoint for each sample, given the observation at all time, smoothing
    also most likely sequence of staets and its associated probabilies
    '''
    # initialization
    eps = 2.22044604925e-16
    if len(np.shape(observations)) == 2:
        print "I'm in the wrong  place"
        numstates = np.shape(transmtrx)[0]
        timelength = np.shape(observations)[1]
        numsamples = np.shape(observations)[0]
        gammas = eps * np.ones((numsamples,timelength,numstates))
        (alphas,forward_log_prob_most_likely_seq,forward_most_likely_seq,Ziis,logobservations) = forward(transmtrx,obsmtrx,pie,observations)
        betas = backward(transmtrx,obsmtrx,pie,observations)
        Zis = eps * np.ones((numsamples,timelength))
        most_likely_seq = eps * np.ones((numsamples,timelength))
        log_prob_most_likely_seq = eps * np.ones((numsamples))
        for sample in range(numsamples):
            for i in range(timelength):
                betas[sample,i,:] /= float(Ziis[sample,i])
        for sample in range(numsamples):
            for t in range(timelength):
                gammas[sample,t,:] = normalize(np.multiply(alphas[sample,t,:],betas[sample,t,:]).reshape(1, -1),norm = 'l1')
                most_likely_seq[sample,t] = np.argmax(gammas[sample,t,:])
            # (alphas[sample,:,:],betas[sample,:,:],gammas[sample,:,:],Ziis) = clipvalues_prevunderflowfw(alphas[sample,:,:],betas[sample,:,:],gammas[sample,:,:],Ziis)
            log_prob_most_likely_seq[sample] = np.sum(np.log(Zis[sample,:]))
    else:
        # print "I'm in the right  place"
        numstates = np.shape(transmtrx)[0]
        timelength = np.shape(observations)[0]
        # print "time length is "
        # print timelength
        gammas = eps *  np.ones((timelength,numstates))
        (alphas,forward_log_prob_most_likely_seq,forward_most_likely_seq,Ziis,logobservations) = forward(transmtrx,obsmtrx,pie,observations)
        betas = backward(transmtrx,obsmtrx,pie,observations)
        Zis = eps * np.ones(timelength)
        most_likely_seq = eps * np.ones(timelength)
        for i in range(timelength):
            betas[i,:] /= float(Ziis[i])
        for t in range(timelength):
            gammas[t,:] = normalize(np.multiply(alphas[t,:],betas[t,:]).reshape(1, -1),norm = 'l1')
            # print gammas
            most_likely_seq[t] = np.argmax(gammas[t,:])
        # (alphas,betas,gammas,Ziis) = clipvalues_prevunderflowfw(alphas,betas,gammas,Ziis)
        log_prob_most_likely_seq = np.sum(np.log(Zis[:]))


    return (gammas,betas,alphas,log_prob_most_likely_seq,most_likely_seq,forward_most_likely_seq,forward_log_prob_most_likely_seq,Ziis,logobservations)

# def main():
#     exmodel = hmmforward(5,10,2,50)
#     observations = exmodel.observations
#     pie = exmodel.pie
#     transmtrx = exmodel.transitionmtrx
#     obsmtrx = exmodel.obsmtrx
#     seqofstates = exmodel.seqofstates
#     (gammas,betas,alphas,log_prob_most_likely_seq,most_likely_seq,forward_most_likely_seq,forward_log_prob_most_likely_seq,,Zis) = forward_backward(transmtrx,obsmtrx,pie,observations)
#     print "forward_backward acc"
#     print np.sum(seqofstates==most_likely_seq) / float(exmodel.obserlength)
#     print "forward acc"
#     print np.sum(seqofstates==forward_most_likely_seq) / float(exmodel.obserlength)
#     print "forward_backward prob"
#     print log_prob_most_likely_seq
#     print "forward prob"
#     print forward_log_prob_most_likely_seq
#     print "forward_backward is more certain at each time point"
#     numwins = 0
#     for i in range(exmodel.obserlength):
#         if max(alphas[i,:]) <= max(gammas[i,:]):
#             numwins +=1
#     print numwins / float(exmodel.obserlength)

    

#     # print stats.mode(seqofstates)
#     # print stats.mode(most_likely_seq)
# main()