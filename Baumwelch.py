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

def computeloglikelihood(pie,transmtrx,obsmtrx,observations):
    numobscases = np.shape(obsmtrx)[1]
    timelength = len(observations)
    nstates = np.shape(obsmtrx)[0] 
    Z = list(itertools.product(range(nstates),repeat = timelength))
    firstelemsum = 0
    secondelemsum = 0
    thirdelemsum = 0
    for z in Z:
        thirdelem = 1.0
        sumak = 0.0
        summ = 0.0
        for time in range(1,timelength):
            thirdelem *= (float(transmtrx[z[time-1],z[time]]) * obsmtrx[z[time],observations[time]])
            sumak += np.log(transmtrx[z[time-1],z[time]])
            summ += np.log(obsmtrx[z[time],observations[time]])
        p = pie[z[0]] * obsmtrx[z[0],observations[0]] * np.prod(transmtrx[z[1]]) * thirdelem
        firstelemsum += float(np.log(pie[z[0]]) * p)
        for time2 in range(1,timelength):
            sumak += np.log(transmtrx[z[time2-1],z[time2]]) * p
            summ += np.log(obsmtrx[z[time2],observations[time2]]) * p
        summ += np.log(obsmtrx[z[0],observations[0]])        
        secondelemsum += float(sumak * p)         
        thirdelemsum += float(summ * p)
    return firstelemsum + secondelemsum + thirdelemsum
        

    return 0




def clipvalues_prevunderflow(pie,transmtrx,obsmtrx,gammas,kissies):
    pie = np.clip(pie,2.22044604925e-16,1.0)
    transmtrx = np.clip(transmtrx,2.22044604925e-16,1.0)
    obsmtrx = np.clip(obsmtrx,2.22044604925e-16,1.0)
    gammas = np.clip(gammas,2.22044604925e-16,1.0)
    kissies = np.clip(kissies,2.22044604925e-16,1.0)
    return (pie,transmtrx,obsmtrx,gammas,kissies)

def clipvalues_prevunderflow_small(pie,transmtrx,obsmtrx):
    pie = np.clip(pie,2.22044604925e-16,1.0)
    transmtrx = np.clip(transmtrx,2.22044604925e-16,1.0)
    obsmtrx = np.clip(obsmtrx,2.22044604925e-16,1.0)
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
    for obs in observations:
        obscounts[int(obs)] +=1
    obsprobs = np.array([float(item)/ float(len(observations)) for item in obscounts])
    pie = np.random.dirichlet(np.ones(numstates),size=1)[0]
    transmtrx = np.empty((numstates,numstates))
    for i in range(numstates):
        transmtrx[i,:] = np.random.dirichlet(np.ones(numstates),size=1)[0]
    obsmtrx = np.empty((numstates,numobscases))
    for i in range(numstates):
        (obsmtrx[i,:],dummy) = normalize(obsprobs + abs(np.random.normal(0,1,numobscases)))
    return (pie,transmtrx,obsmtrx)

def initializeparameters_closetoreality(observations,numstates,numobscases,numsamples,exmodel):
    scale = 2
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
    (pie,transmtrx,obsmtrx,gammas,kissies) = clipvalues_prevunderflow(pie,transmtrx,obsmtrx,gammas,kissies)
    return (gammas,kissies)

def M_step(gammas,kissies,numobscases,observations):
    numstate = np.shape(gammas)[1]
    timelength = np.shape(gammas)[0]
    newpie = np.empty((numstate))
    newtransmtrx =np.empty((numstate,numstate))
    newobsmtrx = np.empty((numstate,numobscases))
    for i in range(numstate):
        newpie[i] = float(gammas[0,i])
    for q in range(numstate):
        denominator = np.sum(gammas[:timelength-1,q])
        for s in range(numstate):
            num = np.sum(kissies[:timelength-1,q,s])
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
    (newpie,newtransmtrx,newobsmtrx,gammas,kissies) = clipvalues_prevunderflow(newpie,newtransmtrx,newobsmtrx,gammas,kissies)
    return (newpie,newtransmtrx,newobsmtrx)



def Baumwelch(observations,numstates,numobscases,numsamples,exmodel):
    ''' you can convert the observations and obsmtrx all together into one matrix
    called soft evidence which is a K * T matrix by using the corresponding
    distribution across all the states for each time point and use that instead'''
    # initialization
    # (pie,transmtrx,obsmtrx )= initializeparameters(observations,numstates,numobscases,numsamples)
    (pie,transmtrx,obsmtrx )= initializeparameters_closetoreality(observations,numstates,numobscases,numsamples,exmodel)
    (pie,transmtrx,obsmtrx ) = clipvalues_prevunderflow_small(pie,transmtrx,obsmtrx)
    print "realpie"
    print exmodel.pie
    print pie
    # print "realtrans"
    # print exmodel.transitionmtrx
    # print transmtrx
    # print "real obsmtrx"
    # print exmodel.obsmtrx
    # print obsmtrx
    # print pie
    noiterations = 20
    conv_threshold = 0.01
    diff_consec_params = 100
    counter = 0
    # print "should be getting smaller"
    print "log likelihood value"
    while(counter < noiterations):
        (gammas,kissies) = E_step(pie,transmtrx,obsmtrx,observations)
        (pie,transmtrx,obsmtrx,gammas,kissies) = clipvalues_prevunderflow(pie,transmtrx,obsmtrx,gammas,kissies)
        prevpie = np.copy(pie)
        prevobsmtrx = np.copy(obsmtrx)
        prevtransmtrx = np.copy(transmtrx)
        (pie,transmtrx,obsmtrx) = M_step(gammas,kissies,numobscases,observations)
        # curloglikelihood = computeloglikelihood(pie,transmtrx,obsmtrx,observations)
        # print curloglikelihood
        (pie,transmtrx,obsmtrx,gammas,kissies) = clipvalues_prevunderflow(pie,transmtrx,obsmtrx,gammas,kissies)        
        piedist = np.linalg.norm(pie - prevpie ) / float(numstates)
        transdist = np.linalg.norm(transmtrx - prevtransmtrx) / float(numstates **2)
        obsdist = np.linalg.norm(obsmtrx - prevobsmtrx) / float(numobscases * numstates)
        diff_consec_params = piedist
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
    print counter
    return (pie,transmtrx,obsmtrx) 

# def main():
#     exmodel = hmmforward(2,2,1,10)
#     numstates = exmodel.numofstates
#     numobscases = exmodel.numofobsercases
#     numsamples = 1
#     observations = exmodel.observations
#     (pie,transmtrx,obsmtrx ) =  Baumwelch(observations,numstates,numobscases,numsamples,exmodel)
#     (pie,transmtrx,obsmtrx) = clipvalues_prevunderflow_small(pie,transmtrx,obsmtrx)
#     piedist = np.linalg.norm(pie - exmodel.pie ) / float(numstates)
#     transdist = np.linalg.norm(transmtrx - exmodel.transitionmtrx) / float(numstates **2)
#     obsdist = np.linalg.norm(obsmtrx - exmodel.obsmtrx) / float(numobscases * numstates)
#     print "realpie"
#     print exmodel.pie
#     print pie
#     # print "realtrans"
#     # print exmodel.transitionmtrx
#     # print transmtrx
#     # print "real obsmtrx"
#     # print exmodel.obsmtrx
#     # print obsmtrx
#     print piedist,transdist,obsdist
#     print "dooshag"
# main()