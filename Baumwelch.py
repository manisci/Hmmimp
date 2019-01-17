import numpy as np,numpy.random
from init_forward import hmmforward
from scipy import stats
from forward import forward
from backward import backward
from forward_backward import forward_backward
from viterbi import viterbi
import itertools
from sklearn.preprocessing import normalize
np.set_printoptions(precision=4,suppress=True)


# def computeloglikelihoodd(pie,transmtrx,obsmtrx,observations):
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
#         secpmdsum = 0.0
#         thirdsum = 0.0
#         for time in range(1,timelength):
#             secpmdsum += np.log(transmtrx[z[time-1],z[time]])
#             for sample in range(numsamples):
#                 thirdelem *= (float(transmtrx[z[time-1],z[time]]) * obsmtrx[z[time],observations[sample,time]])
#                 thirdsum += np.log(obsmtrx[z[time],observations[sample,time]])
#         # np.prod(transmtrx[z[1]]) extra ?
#         p = float(pie[z[0]] * obsmtrx[z[0],observations[0]] * thirdelem)
#         firstelemsum += float(np.log(pie[z[0]]) * p)
#         for time2 in range(1,timelength):
#             secpmdsum += np.log(transmtrx[z[time2-1],z[time2]]) 
#             thirdsum += np.log(obsmtrx[z[time2],observations[time2]]) 
#         thirdsum += np.log(obsmtrx[z[0],observations[0]])        
#         secondelemsum += float(secpmdsum * p)         
#         thirdelemsum += float(thirdsum * p)
#     print "doosh doosgh"
#     return firstelemsum + secondelemsum + thirdelemsum

def computeloglikelihood(pie,transmtrx,obsmtrx,observations):
    numobscases = np.shape(obsmtrx)[1]
    if len(np.shape(observations)) == 1:
        timelength = np.shape(observations)[0]
    else:
        timelength = np.shape(observations)[1]
    nstates = np.shape(obsmtrx)[0] 
    Z = list(itertools.product(range(nstates),repeat = timelength))
    firstelemsum = 0.0
    secondelemsum = 0.0
    thirdelemsum = 0.0
    for z in Z:
        thirdelem = 1.0
        secpmdsum = 0.0
        thirdsum = 0.0
        for time in range(1,timelength):
            thirdelem *= (float(transmtrx[z[time-1],z[time]]) * obsmtrx[z[time],int(observations[time])])
            secpmdsum += np.log(transmtrx[z[time-1] , z[time]])
            thirdsum += np.log(obsmtrx[z[time],int(observations[time])])
        # np.prod(transmtrx[z[1]]) extra ?
        p = float(pie[z[0]] * obsmtrx[z[0],observations[0]] * thirdelem)
        # for time2 in range(1,timelength):
        #     secpmdsum += np.log(transmtrx[z[time2-1] , z[time2]]) 
        #     thirdsum += np.log(obsmtrx[z[time2] , observations[time2]]) 
        thirdsum += np.log(obsmtrx [z[0] , int(observations[0])])
        firstelemsum += float(np.log(pie[z[0]]) * p)        
        secondelemsum += float(secpmdsum * p)         
        thirdelemsum += float(thirdsum * p)
    return firstelemsum + secondelemsum  + thirdelemsum

def computeloglikelihoodnew(pie,transmtrx,obsmtrx,observations):
    eps = 2.22044604925e-16
    numobscases = np.shape(obsmtrx)[1]
    if len(np.shape(observations)) == 1:
        timelength = np.shape(observations)[0]
    else:
        timelength = np.shape(observations)[1]
    nstates = np.shape(obsmtrx)[0] 
    firstelemsum = 0.0
    secondelemsum = 0.0
    thirdelemsum = 0.0
    for state1 in range(nstates):
        firstelemsum += np.log(pie[state1])
        for time in range(timelength):
            for state2 in range(nstates):
                    secondelemsum += np.log(transmtrx[state1,state2] )
            for obs in range(numobscases):
                if observations[time] == obs:
                    thirdelemsum += np.log(obsmtrx[state1,int(obs)])
    return -(firstelemsum + secondelemsum + thirdelemsum)


# advanced version :D
def clipmatrix(mtrx):
    # print "in clipmatrix"
    # print np.shape(mtrx)
    eps = 2.22044604925e-16
    minpie = np.min(mtrx)
    minarg = np.unravel_index(np.argmin(mtrx, axis=None), mtrx.shape)
    mtrx[minarg] = eps
    if len(np.shape(mtrx)) == 2:
        for i in range(np.shape(mtrx)[0]):
            for j in range(np.shape(mtrx)[1]):
                if mtrx[i,j] < eps:
                    mtrx[i,j] = eps +( mtrx[i,j] - minpie)
                if mtrx[i,j] > 1:
                    mtrx[i,j] = 1.0
                if mtrx[i,j] == 0:
                    mtrx[i,j] = eps
    elif len(np.shape(mtrx)) == 3 :
        for i in range(np.shape(mtrx)[0]):
            for j in range(np.shape(mtrx)[1]):
                for k in range(np.shape(mtrx)[2]):
                    if mtrx[i,j,k] < eps:
                        mtrx[i,j,k] = eps +( mtrx[i,j,k] - minpie)
                    if mtrx[i,j,k] > 1:
                        mtrx[i,j,k] = 1.0
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
        

# Simple version
def testprobabilities(pie,transmtrx,obsmtrx):
    numstate = np.shape(transmtrx)[0]
    for state in range(numstate):
        if abs(np.sum(transmtrx[state,:]) -  1 ) > 0.0001:
            print "sth wrong in trans mtrx"
            print np.sum(transmtrx[state,:])
            print state
            print transmtrx[state,:]
        if abs(np.sum(obsmtrx[state,:]) -1 )> 0.0001:
            print "sth wrong in obs mtrx"
            print np.sum(obsmtrx[state,:])
            print state
            print obsmtrx[state,:]
    if abs(np.sum(pie) -  1) > 0.0001:
        print "sth wrong in pie"
        print pie

# def clipvalues_prevunderflow(pie,transmtrx,obsmtrx,gammas,kissies):
#     pie = np.clip(pie,2.22044604925e-16,1.0)
#     transmtrx = np.clip(transmtrx,2.22044604925e-16,1.0)
#     obsmtrx = np.clip(obsmtrx,2.22044604925e-16,1.0)
#     gammas = np.clip(gammas,2.22044604925e-16,1.0)
#     kissies = np.clip(kissies,2.22044604925e-16,1.0)
#     return (pie,transmtrx,obsmtrx,gammas,kissies)

# def clipvalues_prevunderflow_small(pie,transmtrx,obsmtrx):
#     pie = np.clip(pie,2.22044604925e-16,1.0)
#     transmtrx = np.clip(transmtrx,2.22044604925e-16,1.0)
#     obsmtrx = np.clip(obsmtrx,2.22044604925e-16,1.0)
#     return (pie,transmtrx,obsmtrx)
 
def clipvalues_prevunderflow(pie,transmtrx,obsmtrx,gammas,kissies):
    eps = 2.22044604925e-16
    minpie = np.min(pie)
    pie[np.argmin(pie)] = eps
    # print "in the beginning of the prevunderflow"
    # print np.shape(gammas)
    for i in range(np.shape(pie)[0]):
        if pie[i] < eps:
            pie[i] = eps +( pie[i] - minpie)
        if pie[i] > 1:
            pie[i] = 1.0
        if pie[i] == 0:
            pie[i] = eps
    transmtrx = clipmatrix(transmtrx)
    obsmtrx = clipmatrix(obsmtrx)
    # print " in prevunderlow before clipping gamma debug"
    # print np.shape(gammas)
    # print gammas
    gammas = clipmatrix(gammas)
    # print  "after clipping gammas in prevunderflow"
    # print np.shape(gammas)    
    # print gammas

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

# def normalize(u):
#     Z = np.sum(u)
#     if Z==0:
#         return (u,1.0)
#     else:
#         v = u / Z
#     return (v,Z)

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
        obsmtrx[i,:] = normalize((obsprobs + abs(np.random.normal(0,1,numobscases))).reshape(1, -1),norm = 'l1')
    return (pie,transmtrx,obsmtrx)

def initializeparameters_closetoreality(observations,numstates,numobscases,numsamples,exmodel):
    scale = 0.5
    (pie) = normalize((exmodel.pie + abs(np.random.normal(0,scale,numstates))).reshape(1, -1),norm = 'l1')
    transmtrx = np.empty((numstates,numstates))
    for i in range(numstates):
        transmtrx[i,:] = normalize((exmodel.transitionmtrx[i,:] + abs(np.random.normal(0,scale,numstates))).reshape(1, -1),norm = 'l1')
    obsmtrx = np.empty((numstates,numobscases))
    for j in range(numstates):
        obsmtrx[j,:] = normalize((exmodel.obsmtrx[j,:] + abs(np.random.normal(0,scale,numobscases))).reshape(1, -1),norm = 'l1')
    return (pie,transmtrx,obsmtrx)
    
def E_step(pie,transmtrx,obsmtrx,observations):
    (gammas,betas,alphas,log_prob_most_likely_seq,most_likely_seq,forward_most_likely_seq,forward_log_prob_most_likely_seq,Zis) = \
    forward_backward(transmtrx,obsmtrx,pie,observations)
    # print " in E step after forward backward"
    # print np.shape(gammas)    
    # print log_prob_most_likely_seq
    # print log_prob_most_likely_seq[0]
    # normalizing betas by the same scale as alphas
    if len(np.shape(observations)) == 2:
        timelength = np.shape(observations)[1]
        numsamples = np.shape(observations)[0]
        numstate = np.shape(transmtrx)[0]
        kissies = np.empty((numsamples,timelength,numstate,numstate))
        for sample in range(numsamples):
            for t in range(timelength-1):
                for q in range(numstate):
                    for s in range(numstate):
                        kissies[sample,t,q,s] = float(alphas[sample,t,q]) * float(transmtrx[q,s]) * float(obsmtrx[s,int(observations[sample,t+1])] * betas[sample,t+1,s])
                kissies[sample,t,:,:] = normalize(kissies[sample,t,:,:],norm = 'l1')
        # (pie,transmtrx,obsmtrx,gammas,kissies) = clipvalues_prevunderflow(pie,transmtrx,obsmtrx,gammas,kissies)
    else:
        timelength = np.shape(observations)[0]

        # for time in range(timelength):
        #     (betas[time,:],dumak) = normalize(betas[time,:])
            # alphas[time,:] /= float(np.sum(alphas[time,:]))
        numstate = np.shape(transmtrx)[0]
        kissies = np.empty((timelength,numstate,numstate))
        for t in range(timelength-1):
            for q in range(numstate):
                for s in range(numstate):
                    kissies[t,q,s] = float(alphas[t,q]) * float(transmtrx[q,s]) * float(obsmtrx[s,int(observations[t+1])] * betas[t+1,s])
            kissies[t,:,:] = normalize(kissies[t,:,:],norm = 'l1')
        # (pie,transmtrx,obsmtrx,gammas,kissies) = clipvalues_prevunderflow(pie,transmtrx,obsmtrx,gammas,kissies)
    # print " In E step after clipping "
    # print np.shape(gammas)
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
                    for sample in range(numsamples):
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
    (newpie,newtransmtrx,newobsmtrx,gammas,kissies) = clipvalues_prevunderflow(newpie,newtransmtrx,newobsmtrx,gammas,kissies)
    return (newpie,newtransmtrx,newobsmtrx)

def Baumwelch(observations,numstates,numobscases,exmodel = None):
    ''' exmodel is only used for initializations close to reality'''
    # print "this should be increasing"
    # print "timelength"
    # print timelength

    if len(np.shape(observations)) != 1:
        numsamples = np.shape(observations)[0]
    else:
        numsamples = 1

    # initialization
    # (pie,transmtrx,obsmtrx )= initializeparameters(observations,numstates,numobscases,numsamples)
    # (pie,transmtrx,obsmtrx )= initializeparameters_closetoreality(observations,numstates,numobscases,numsamples,exmodel)
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
    pie = exmodel.pie
    transmtrx = exmodel.transitionmtrx
    obsmtrx = exmodel.obsmtrx
    noiterations = 20
    conv_threshold = 0.001
    diff_consec_params = 100
    # (gammas,betas,alphas,log_prob_most_likely_seq,most_likely_seq,forward_most_likely_seq,forward_log_prob_most_likely_seq,Zis) = \
    # forward_backward(transmtrx,obsmtrx,pie,observations)
    # print log_prob_most_likely_seq
    # print "initial log likelihood is "
    # print log_prob_most_likely_seq
    # print betas[timelength-1,np.argmax(pie)]
    # print np.sum(alphas[timelength-1,:])
    # (Optstate, deltas) = viterbi(transmtrx,obsmtrx,pie,observations)
    # print "seq of states"
    # print Optstate
    # print "deltas, ending up in state s"
    # print deltas
    # print "initial dooshag value "
    # prob = 1;
    # for i in range(len(Optstate)):
    #     prob *= deltas[i,Optstate[i][0]]
    # print prob
    counter = 0
    # print "should be getting smaller"
    # print "log likelihood value"
    while(counter < noiterations):
        # print "be differnt old:"
        # print transmtrx
        (gammas,kissies) = E_step(pie,transmtrx,obsmtrx,observations)
        # print "befpre clipping"
        # print np.shape(gammas)
        # (pie,transmtrx,obsmtrx,gammas,kissies) = clipvalues_prevunderflow(pie,transmtrx,obsmtrx,gammas,kissies)
        # print "after clipping"
        # print np.shape(gammas)
        prevpie = np.copy(pie)
        prevobsmtrx = np.copy(obsmtrx)
        prevtransmtrx = np.copy(transmtrx)
        (pie,transmtrx,obsmtrx) = M_step(gammas,kissies,numobscases,observations)
        # print "after m steps"
        # print np.shape(gammas)I'm in the right  place

        # print "new"
        # print transmtrx
        # testprobabilities(pie,transmtrx,obsmtrx)
        # (pie,transmtrx,obsmtrx,gammas,kissies) = clipvalues_prevunderflow(pie,transmtrx,obsmtrx,gammas,kissies) 
        # print "after clipping of the m step"
        # print "one iteration is done now for the fun part calculation of likelihood"
        # print "gammas are"
        # print gammas
        # curloglikelihood = computeloglikelihood(pie,transmtrx,obsmtrx,observations)
        # print curloglikelihood
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
    # print "final log likelihood is "
    # (gammas,betas,alphas,log_prob_most_likely_seq,most_likely_seq,forward_most_likely_seq,forward_log_prob_most_likely_seq,Zis) = \
    # forward_backward(transmtrx,obsmtrx,pie,observations)
    # print log_prob_most_likely_seq
    # print np.sum(alphas[timelength-1,:])
    # (Optstate, deltas) = viterbi(transmtrx,obsmtrx,pie,observations)
    # print "seq of states"
    # print Optstate
    # print "deltas, ending up in state s"
    # print deltas
    # print "final dooshag value "
    # prob = 1;
    # for i in range(len(Optstate)):
    #     prob *= deltas[i,Optstate[i][0]]
    # print prob
    return (pie,transmtrx,obsmtrx) 

# def main():
#     exmodel = hmmforward(2,4,1,150,2)
#     numstates = exmodel.numofstates
#     numobscases = exmodel.numofobsercases
#     observations = exmodel.observations
#     (pie,transmtrx,obsmtrx) =  Baumwelch(observations,numstates,numobscases,exmodel)
#     # (pie,transmtrx,obsmtrx) = clipvalues_prevunderflow_small(pie,transmtrx,obsmtrx)
#     piedist = np.linalg.norm(pie - exmodel.pie ) / float(numstates)
#     transdist = np.linalg.norm(transmtrx - exmodel.transitionmtrx) / float(numstates **2)
#     obsdist = np.linalg.norm(obsmtrx - exmodel.obsmtrx) / float(numobscases * numstates)
#     print "realpie"
#     print exmodel.pie
#     print pie
#     print "realtrans"
#     print exmodel.transitionmtrx
#     print transmtrx
#     print "real obsmtrx"
#     print exmodel.obsmtrx
#     print obsmtrx
#     print piedist,transdist,obsdist
#     print "dooshag"
# main()