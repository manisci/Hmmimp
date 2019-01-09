import numpy as np,numpy.random
from init_forward import hmmforward
from scipy import stats
from forward import forward
from backward import backward
from forward_backward import forward_backward
import itertools

def permutations(iterable, r):
    # permutations('ABCD', 2) --> AB AC AD BA BC BD CA CB CD DA DB DC
    # permutations(range(3)) --> 012 021 102 120 201 210
    pool = tuple(iterable)
    n = len(pool)
    r = n if r is None else r
    if r > n:
        return
    indices = range(n)
    cycles = range(n, n-r, -1)
    yield tuple(pool[i] for i in indices[:r])
    while n:
        for i in reversed(range(r)):
            cycles[i] -= 1
            if cycles[i] == 0:
                indices[i:] = indices[i+1:] + indices[i:i+1]
                cycles[i] = n - i
            else:
                j = cycles[i]
                indices[i], indices[-j] = indices[-j], indices[i]
                yield tuple(pool[i] for i in indices[:r])
                break
        else:
            return

# def computeloglikelihood(pie,transmtrx,obsmtrx,observations):
#     numobscases = np.shape(obsmtrx)[1]
#     timelength = np.shape(observations)[1]
#     numsamples = np.shape(observations)[0]
#     nstates = np.shape(obsmtrx)[0] 
#     Z = list(itertools.product(range(nstates),repeat = timelength))
#     firstelemsum = 0.0
#     secondelemsum = 0.0
#     thirdelemsum = 0.0
#     for z in Z:
#         thirdelem = 1.0
#         sumak = 0.0
#         summ = 0.0
#         for time in range(1,timelength):
#             sumak += np.log(transmtrx[z[time-1],z[time]])
#             for sample in range(numsamples):
#                 thirdelem *= (float(transmtrx[z[time-1],z[time]]) * obsmtrx[z[time],observations[sample,time]])
#                 summ += np.log(obsmtrx[z[time],observations[sample,time]])
#         # np.prod(transmtrx[z[1]]) extra ?
#         p = float(pie[z[0]] * obsmtrx[z[0],observations[0]] * thirdelem)
#         firstelemsum += float(np.log(pie[z[0]]) * p)
#         for time2 in range(1,timelength):
#             sumak += np.log(transmtrx[z[time2-1],z[time2]]) 
#             summ += np.log(obsmtrx[z[time2],observations[time2]]) 
#         summ += np.log(obsmtrx[z[0],observations[0]])        
#         secondelemsum += float(sumak * p)         
#         thirdelemsum += float(summ * p)
#     print "doosh doosgh"
#     return firstelemsum + secondelemsum + thirdelemsum



def clipmatrix(mtrx):
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
                    
    elif len(np.shape(mtrx)) == 4:
        for i in range(np.shape(mtrx)[0]):
            for j in range(np.shape(mtrx)[1]):
                for k in range(np.shape(mtrx)[2]):
                    for l in range(np.shape(mtrx)[3]):
                        if mtrx[i,j,k,l] < eps:
                            mtrx[i,j,k,l] = eps +( mtrx[i,j,k,l] - minpie)
                        # if mtrx[i,j,k,l] > 1:
                        #     mtrx[i,j,k,l] = 1.0 
                        if mtrx[i,j,k,l] == 0:
                            mtrx[i,j,k,l] = eps      
        


        
    return mtrx
def clipvalues_prevunderflow(pie,transmtrx,obsmtrx,gammas,kissies):
    eps = 2.22044604925e-16
    minpie = np.min(pie)
    pie[np.argmin(pie)] = eps
    for i in range(np.shape(pie)[0]):
        if pie[i] < eps:
            pie[i] = eps +( pie[i] - minpie)
        if pie[i] > 1:
            pie[i] = 1.0
        if pie[i] == 0:
            pie[i] = eps
    transmtrx = clipmatrix(transmtrx)
    obsmtrx = clipmatrix(obsmtrx)
    gammas = clipmatrix(gammas)
    kissies = clipmatrix(kissies)
    return (pie,transmtrx,obsmtrx,gammas,kissies)

def clipvalues_prevunderflow_small(pie,transmtrx,obsmtrx):
    eps = 2.22044604925e-16
    minpie = np.min(pie)
    pie[np.argmin(pie)] = eps
    for i in range(np.shape(pie)[0]):
        if pie[i] < eps:
            pie[i] = eps +( pie[i] - minpie)
        if pie[i] > 1:
            pie[i] = 1.0
        if pie[i] == 0:
            pie[i] = eps    
    transmtrx = clipmatrix(transmtrx)
    obsmtrx = clipmatrix(obsmtrx)
    return (pie,transmtrx,obsmtrx)

def normalize(u):
    Z = np.sum(u)
    if Z==0:
        return (u,1.0)
    else:
        v = u / Z
    return (v,Z)

def initializeparameters(observations,numstates,numobscases,numsamples):
    obscounts = [0] * numobscases
    if numsamples == 1:
        timelength = np.shape(observations)[0]
        for obs in observations:
            obscounts[int(obs)] +=1
    else:
        timelength = np.shape(observations)[1]
        for samp in range(numsamples):
            for obs in observations[samp,:]:
                obscounts[int(obs)] +=1

    obsprobs = np.array([float(item)/ float(numsamples * timelength) for item in obscounts])
    pie = np.random.dirichlet(np.ones(numstates),size=1)[0]
    transmtrx = np.empty((numstates,numstates))
    for i in range(numstates):
        transmtrx[i,:] = np.random.dirichlet(np.ones(numstates),size=1)[0]
    obsmtrx = np.empty((numstates,numobscases))
    for i in range(numstates):
        (obsmtrx[i,:],dummy) = normalize(obsprobs + abs(np.random.normal(0,1,numobscases)))
    return (pie,transmtrx,obsmtrx)

def initializeparameters_closetoreality(observations,numstates,numobscases,numsamples,exmodel):
    scale = 1
    (pie,dummy) = normalize(exmodel.pie + abs(np.random.normal(0,scale,numstates)))
    transmtrx = np.empty((numstates,numstates))
    for i in range(numstates):
        (transmtrx[i,:],dummy) = normalize(exmodel.transitionmtrx[i,:] + abs(np.random.normal(0,scale,numstates)))
    obsmtrx = np.empty((numstates,numobscases))
    for j in range(numstates):
        (obsmtrx[j,:],dummy) = normalize(exmodel.obsmtrx[j,:] + abs(np.random.normal(0,scale,numobscases)))
    return (pie,transmtrx,obsmtrx)
    
def E_step(pie,transmtrx,obsmtrx,observations):
    (gammas,betas,alphas,log_prob_most_likely_seq,most_likely_seq,forward_most_likely_seq,forward_log_prob_most_likely_seq,Zis) = \
    forward_backward(transmtrx,obsmtrx,pie,observations)
    # normalizing betas by the same scale as alphas
    if len(np.shape(observations)) == 2:
        timelength = np.shape(observations)[1]
        numsamples = np.shape(observations)[0]

        # for time in range(timelength):
        #     (betas[time,:],dumak) = normalize(betas[time,:])
            # alphas[time,:] /= float(np.sum(alphas[time,:]))
        numstate = np.shape(transmtrx)[0]
        kissies = np.empty((numsamples,timelength,numstate,numstate))
        for sample in range(numsamples):
            for t in range(timelength-1):
                for q in range(numstate):
                    for s in range(numstate):
                        kissies[sample,t,q,s] = float(alphas[sample,t,q]) * float(transmtrx[q,s]) * float(obsmtrx[s,int(observations[sample,t+1])] * betas[sample,t+1,s])
                (kissies[sample,t,:,:],dummy) = normalize(kissies[sample,t,:,:])
        # (pie,transmtrx,obsmtrx,gammas,kissies) = clipvalues_prevunderflow(pie,transmtrx,obsmtrx,gammas,kissies)
    else:
        timelength = len(observations)

        # for time in range(timelength):
        #     (betas[time,:],dumak) = normalize(betas[time,:])
            # alphas[time,:] /= float(np.sum(alphas[time,:]))
        numstate = np.shape(transmtrx)[0]
        kissies = np.empty((timelength,numstate,numstate))
        for t in range(timelength-1):
            for q in range(numstate):
                for s in range(numstate):
                    kissies[t,q,s] = float(alphas[t,q]) * float(transmtrx[q,s]) * float(obsmtrx[s,int(observations[t+1])] * betas[t+1,s])
            (kissies[t,:,:],dummy) = normalize(kissies[t,:,:])
        # (pie,transmtrx,obsmtrx,gammas,kissies) = clipvalues_prevunderflow(pie,transmtrx,obsmtrx,gammas,kissies)

    return (gammas,kissies)

def M_step(gammas,kissies,numobscases,observations):
    if len(np.shape(observations)) == 2:
        numstate = np.shape(gammas)[2]
        timelength = np.shape(gammas)[1]
        newpie = np.empty((numstate))
        newtransmtrx =np.empty((numstate,numstate))
        newobsmtrx = np.empty((numstate,numobscases))
        numsamples = np.shape(gammas)[0]
        for i in range(numstate):
            newpie[i] = np.mean((gammas[:,0,i]))
        for q in range(numstate):
            denominator = np.sum(gammas[:,:timelength-1,q])
            for s in range(numstate):
                newtransmtrx[q,s] = float(np.sum(kissies[:,:timelength-1,q,s]) )/ float(denominator)
        for state in range(numstate):
            denom = np.sum(gammas[:,:,state])
            for obs in range(numobscases):
                numerator = 0.0
                for time in range(timelength):
                    for sample in numsamples:
                        if int(observations[sample,time]) == obs:
                            # print "here"
                            numerator += gammas[sample,time,state]
                newobsmtrx[state,obs] = float(numerator) / float(denom)
    else:
        # single observation
        numstate = np.shape(gammas)[1]
        timelength = np.shape(gammas)[0]
        newpie = np.empty((numstate))
        newtransmtrx =np.empty((numstate,numstate))
        newobsmtrx = np.empty((numstate,numobscases))
        for i in range(numstate):
            newpie[i] = float((gammas[0,i]))
        for q in range(numstate):
            denominator = np.sum(gammas[:timelength-1,q])
            for s in range(numstate):
                newtransmtrx[q,s] = float(np.sum(kissies[:timelength-1,q,s]) )/ float(denominator)
        for state in range(numstate):
            denom = np.sum(gammas[:,state])
            for obs in range(numobscases):
                numerator = 0.0
                for time in range(timelength):
                    if int(observations[time]) == obs:
                        # print "here"
                        numerator += gammas[time,state]
                newobsmtrx[state,obs] = float(numerator) / float(denom)
    # (newpie,newtransmtrx,newobsmtrx,gammas,kissies) = clipvalues_prevunderflow(newpie,newtransmtrx,newobsmtrx,gammas,kissies)
    return (newpie,newtransmtrx,newobsmtrx)

def Baumwelch(observations,numstates,numobscases,exmodel = None):
    ''' exmodel is only used for initializations close to reality'''
    if len(np.shape(observations)) != 1:
        numsamples = np.shape(observations)[0]
    else:
        numsamples = 1

    # initialization
    # (pie,transmtrx,obsmtrx )= initializeparameters(observations,numstates,numobscases,numsamples)
    (pie,transmtrx,obsmtrx )= initializeparameters_closetoreality(observations,numstates,numobscases,numsamples,exmodel)
    # (pie,transmtrx,obsmtrx ) = clipvalues_prevunderflow_small(pie,transmtrx,obsmtrx)
    # print "realpie"
    # print exmodel.pie
    # print pie
    # print "realtrans"
    # print exmodel.transitionmtrx
    # print transmtrx
    # print "real obsmtrx"
    # print exmodel.obsmtrx
    # print obsmtrx
    # print pie
    noiterations = 20
    conv_threshold = 0.001
    diff_consec_params = 100
    counter = 0
    # print "should be getting smaller"
    print "log likelihood value"
    while(counter < noiterations):
        (gammas,kissies) = E_step(pie,transmtrx,obsmtrx,observations)
        # (pie,transmtrx,obsmtrx,gammas,kissies) = clipvalues_prevunderflow(pie,transmtrx,obsmtrx,gammas,kissies)
        prevpie = np.copy(pie)
        prevobsmtrx = np.copy(obsmtrx)
        prevtransmtrx = np.copy(transmtrx)
        (pie,transmtrx,obsmtrx) = M_step(gammas,kissies,numobscases,observations)
        # curloglikelihood = computeloglikelihood(pie,transmtrx,obsmtrx,observations)
        # print curloglikelihood
        # (pie,transmtrx,obsmtrx,gammas,kissies) = clipvalues_prevunderflow(pie,transmtrx,obsmtrx,gammas,kissies)        
        piedist = np.linalg.norm(pie - prevpie ) / float(numstates)
        transdist = np.linalg.norm(transmtrx - prevtransmtrx) / float(numstates **2)
        obsdist = np.linalg.norm(obsmtrx - prevobsmtrx) / float(numobscases * numstates)
        diff_consec_params = obsdist
        # diff_consec_params = piedist + transdist + obsdist
        counter +=1
        piedistak = float(np.linalg.norm(pie - exmodel.pie ) )/ float(numstates)
        transdistak = float(np.sqrt(np.sum((transmtrx - exmodel.transitionmtrx)**2)) )/ float(numstates * numstates)
        obsdistak = np.linalg.norm(obsmtrx - exmodel.obsmtrx) / float(numobscases * numstates)

        # print piedistak
        # print transdistak
        # print obsdistak

        # print pie
    # print "went this much in loop"
    return (pie,transmtrx,obsmtrx) 

# def main():
#     exmodel = hmmforward(2,3,1,10,1)
#     numstates = exmodel.numofstates
#     numobscases = exmodel.numofobsercases
#     observations = exmodel.observations
#     (pie,transmtrx,obsmtrx ) =  Baumwelch(observations,numstates,numobscases,exmodel)
#     (pie,transmtrx,obsmtrx) = clipvalues_prevunderflow_small(pie,transmtrx,obsmtrx)
#     piedist = np.linalg.norm(pie - exmodel.pie ) / float(numstates)
#     transdist = np.linalg.norm(transmtrx - exmodel.transitionmtrx) / float(numstates **2)
#     obsdist = np.linalg.norm(obsmtrx - exmodel.obsmtrx) / float(numobscases * numstates)
#     print "realpie"
#     print exmodel.pie
#     print pie
#     # print "realtrans"
#     # print exmodel.transitionmtrx
#     print transmtrx
#     # print "real obsmtrx"
#     # print exmodel.obsmtrx
#     # print obsmtrx
#     print piedist,transdist,obsdist
#     print "dooshag"
# main()