import numpy as np,numpy.random
from init_gaussian import hmmgaussian
from scipy import stats
from forwardcont import forwardcont
from backwardcont import backwardcont
from forward_backward_cont import forward_backwardcont
from viterbicont import viterbicont
import itertools
from sklearn.preprocessing import normalize
import matplotlib 
from sklearn.cluster import KMeans
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
# np.set_printoptions(precision=4,suppress=True)


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
def initialize_with_kmeans(observations,numstates,numsamples,exmodel):
    eps = 2.22044604925e-16

    if numsamples == 1:
            timelength = np.shape(observations)[0]
    else:
        timelength = np.shape(observations)[1]

    pie = np.random.dirichlet(np.ones(numstates),size=1)[0]
    transmtrx = eps * np.ones((numstates,numstates))
    for i in range(numstates):
        transmtrx[i,:] = np.random.dirichlet(np.ones(numstates),size=1)[0]
    vals = []
    for state in range(numstates):
        vals.append([])
    obsmtrx = eps * np.ones((numstates,2))
    observations = np.ndarray.flatten(observations)
    kmeans = KMeans(n_clusters=numstates, random_state=0).fit(observations.reshape(-1, 1))
    clusterpreds = kmeans.labels_
    # print len(observations)
    for obsidx in range(len(observations)):
        vals[clusterpreds[obsidx]].append(observations[obsidx])
    for state in range(numstates):
        obsmtrx[state,0] = np.mean(vals[state])
        obsmtrx[state,1] = np.std(vals[state])
    return (pie,transmtrx,obsmtrx)

def computeloglikelihood(pie,transmtrx,obsmtrx,observations):
    eps = 2.22044604925e-16
    numobscases = np.shape(obsmtrx)[1]
    if len(np.shape(observations)) == 1:
        timelength = np.shape(observations)[0]
    else:
        timelength = np.shape(observations)[1]
    nstates = np.shape(obsmtrx)[0] 
    Z = list(itertools.product(range(nstates),repeat = timelength))
    firstelemsum = eps
    secondelemsum = eps
    thirdelemsum = eps
    for z in Z:
        thirdelem = 1.0
        secpmdsum = eps
        thirdsum = eps
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
        for state2 in range(nstates):
            secondelemsum += np.log(transmtrx[state1,state2] )
        for time in range(timelength):
            for obs in range(numobscases):
                if observations[time] == obs:
                    thirdelemsum += np.log(obsmtrx[state1,int(obs)])
    return (firstelemsum + secondelemsum + thirdelemsum)


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


def initializeparameters(observations,numstates,numsamples):
    eps = 2.22044605e-16
    if numsamples == 1:
        timelength = np.shape(observations)[0]
    else:
        timelength = np.shape(observations)[1]

    obsmean = np.mean(observations)
    obsvar = np.var(observations)+ eps
    pie = np.random.dirichlet(np.ones(numstates),size=1)[0]
    transmtrx = eps * np.ones((numstates,numstates))
    for i in range(numstates):
        transmtrx[i,:] = np.random.dirichlet(np.ones(numstates),size=1)[0]
    obsmtrx = eps * np.ones((numstates,2))
    for i in range(numstates):
        obsmtrx[i,0] = obsmean + abs(np.random.normal(0,1,1))
        obsmtrx[i,1] = np.sqrt(obsvar + abs(np.random.normal(0,1,1)))
    return (pie,transmtrx,obsmtrx)

def initializeparameters_closetoreality(observations,numstates,numsamples,exmodel):
    eps = 2.22044605e-16
    scale = 0.8
    pie = np.array(normalize((exmodel.pie + abs(np.random.normal(0,scale,numstates))).reshape(1, -1),norm = 'l1'))[0]
    transmtrx = eps * np.ones ((numstates,numstates))
    for i in range(numstates):
        transmtrx[i,:] = normalize((exmodel.transitionmtrx[i,:] + abs(np.random.normal(0,scale,numstates))).reshape(1, -1),norm = 'l1')
    obsmtrx = eps * np.ones((numstates,2))
    for j in range(numstates):
        obsmtrx[j,0] = exmodel.obsmtrx[j,0] + abs(np.random.normal(0,scale,1))
        obsmtrx[j,1] = exmodel.obsmtrx[j,1] + abs(np.random.normal(0,scale,1))
    return (pie,transmtrx,obsmtrx)
    
def E_step(pie,transmtrx,obsmtrx,observations):
    eps = 2.22044605e-16
    (gammas,betas,alphas,log_prob_most_likely_seq,most_likely_seq,forward_most_likely_seq,forward_log_prob_most_likely_seq,Zis,logobservations) = \
    forward_backwardcont(transmtrx,obsmtrx,pie,observations)
    # print 'dear gammas'
    # print gammas
    # print " in E step after forward backward"
    # print np.shape(gammas)    
    # print log_prob_most_likely_seq
    # print log_prob_most_likely_seq[0]
    # normalizing betas by the same scale as alphas
    if len(np.shape(observations)) == 2:
        timelength = np.shape(observations)[1]
        numsamples = np.shape(observations)[0]
        numstate = np.shape(transmtrx)[0]
        kissies = eps * np.ones((numsamples,timelength,numstate,numstate))
        for sample in range(numsamples):
            for t in range(timelength-1):
                for q in range(numstate):
                    for s in range(numstate):
                        distr = stats.norm(obsmtrx[s,0], obsmtrx[s,1])
                        obsprob = distr.pdf(observations[sample,t+1])
                        kissies[sample,t,q,s] = float(alphas[sample,t,q]) * float(transmtrx[q,s]) * float(obsprob * betas[sample,t+1,s])
                kissies[sample,t,:,:] /= (np.sum(kissies[sample,t,:,:]))
        for sample in range(numsamples):
            kissies[sample,timelength-1,:,:] /= np.sum(kissies[sample,timelength-1,:,:])
        
        # (pie,transmtrx,obsmtrx,gammas,kissies) = clipvalues_prevunderflow(pie,transmtrx,obsmtrx,gammas,kissies)
    else:
        timelength = np.shape(observations)[0]

        # for time in range(timelength):
        #     (betas[time,:],dumak) = normalize(betas[time,:])
            # alphas[time,:] /= float(np.sum(alphas[time,:]))
        numstate = np.shape(transmtrx)[0]
        kissies = eps * np.ones((timelength,numstate,numstate))
        for t in range(timelength-1):
            for q in range(numstate):
                for s in range(numstate):
                    distr = stats.norm(obsmtrx[s,0], obsmtrx[s,1])
                    obsprob = distr.pdf(observations[t+1])
                    # print obsprob
                    kissies[t,q,s] = float(alphas[t,q]) * float(transmtrx[q,s]) * float(obsprob * betas[t+1,s])
            kissies[t,:,:] /= np.sum(kissies[t,:,:])
        kissies[timelength-1,:,:] /= np.sum(kissies[timelength-1,:,:])
        # print "kissies"
        # print kissies[timelength-1,:,:] 
        # (pie,transmtrx,obsmtrx,gammas,kissies) = clipvalues_prevunderflow(pie,transmtrx,obsmtrx,gammas,kissies)
    # print " In E step after clipping "
    # print np.shape(gammas)
    return (gammas,kissies,logobservations)

def M_step(gammas,kissies,observations,hard = False):
    eps = 2.22044605e-16
    if len(np.shape(observations)) == 2:
        numstate = np.shape(gammas)[2]
        timelength = np.shape(gammas)[1]
        newpie = eps *np.ones((numstate))
        newtransmtrx = eps * np.ones((numstate,numstate))
        newobsmtrx = eps * np.ones((numstate,2))
        numsamples = np.shape(gammas)[0]
        for i in range(numstate):
            newpie[i] = np.mean((gammas[:,0,i]))
        for q in range(numstate):
            denominator = np.sum(gammas[:,:timelength-1,q]) 
            for s in range(numstate):
                newtransmtrx[q,s] = float(np.sum(kissies[:,:timelength-1,q,s]) )/ float(denominator)
        if hard == False:
            for state in range(numstate):
                meanreprval = 0
                varreprval = 0
                for time in range(timelength):
                    for sample in range(numsamples):
                        meanreprval += gammas[sample,time,state] * observations[sample,time]
                # print np.sum(gammas[:,:,state])
                meanak = meanreprval / np.sum(gammas[:,:,state])
                for time2 in range(timelength):
                    for sample in range(numsamples):
                        varreprval += gammas[sample,time2,state] * (((observations[sample,time2]) - meanak) ** 2)
                newobsmtrx[state,0] = meanreprval / np.sum(gammas[:,:,state])
                newobsmtrx[state,1] = np.sqrt(varreprval / np.sum(gammas[:,:,state]))
                if newobsmtrx[state,1] == 0:
                    raise ValueError("Too many states, either initialize again, or reduce the number of states by one")
        else:
            assignments = []
            for i in range(numstate):
                assignments.append([])
            for time in range(timelength):
                for sample in range(numsamples):
                    most_likely_state = np.argmax(gammas[sample,time,:])
                    assignments[most_likely_state].append(observations[sample,time])
            for state in range(numstate):
                if len(assignments[state]) >= 1:
                    newobsmtrx[state,0] = np.mean(assignments[state])
                    newobsmtrx[state,1] = np.sqrt(np.var(assignments[state]) )
                if newobsmtrx[state,1] == 0:
                    raise ValueError("Too many states, either initialize again, or reduce the number of states by one")

    else:
        # single observation
        numstate = np.shape(gammas)[1]
        timelength = np.shape(gammas)[0]
        newpie = eps * np.ones((numstate))
        newtransmtrx = eps * np.ones((numstate,numstate))
        newobsmtrx = eps * np.ones((numstate,2))
        for i in range(numstate):
            newpie[i] = float((gammas[0,i]))
        for q in range(numstate):
            denominator = np.sum(gammas[:timelength-1,q])
            for s in range(numstate):
                newtransmtrx[q,s] = float(np.sum(kissies[:timelength-1,q,s]) )/ float(denominator)
        if hard == False:
            for state in range(numstate):
                meanreprval = 0
                varreprval = 0
                for time in range(timelength):
                    meanreprval += gammas[time,state] * observations[time]
                meanak = meanreprval / np.sum(gammas[:,state])
                for time2 in range(timelength):
                    varreprval += gammas[time2,state] * (((observations[time2]) - meanak) ** 2)
                # print "numerator is"
                # print numerator
                newobsmtrx[state,0] = meanreprval / np.sum(gammas[:,state])
                newobsmtrx[state,1] = np.sqrt(varreprval / np.sum(gammas[:,state]))
                if newobsmtrx[state,1] == 0:
                    raise ValueError("Too many states, either initialize again, or reduce the number of states by one")
        else:
            assignments = []
            for i in range(numstate):
                assignments.append([])
            for time in range(timelength):
                most_likely_state = np.argmax(gammas[time,:])
                assignments[most_likely_state].append(observations[time])
            for state in range(numstate):
                if len(assignments[state]) >= 1:
                    newobsmtrx[state,0] = np.mean(assignments[state])
                    newobsmtrx[state,1] = np.sqrt(np.var(assignments[state]))
                if newobsmtrx[state,1] == 0:
                    raise ValueError("Too many states, either initialize again, or reduce the number of states by one")
    # (newpie,newtransmtrx,newobsmtrx,gammas,kissies) = clipvalues_prevunderflow(newpie,newtransmtrx,newobsmtrx,gammas,kissies)
    return (newpie,newtransmtrx,newobsmtrx)

def Baumwelchcont(observations,numstates,exmodel,hard = False,conv_threshold = 1e-16 ):
    eps = 2.22044605e-16
    ''' Uses an EM moedel and maximul likelihood estimation to learn the parameteres of an HMM model given the observations 
    In order to compute log likelihood, probabilitey of seeing the observations given the model at that iteration is used. 
    For convergence purposes, the updating continues till the maximum value of difference between 
    previous iteration likelihood and current iteartion likelihood among all samples is smaller than machine epsilon.
    Inputs : observations,numstates,numobscases
    Output: Learned parameteters, pie,transmtrx,obsmtrx
    ** Note: exmodel is only used for initializations close to reality. 
    '''
    # print "this should be increasing"
    # print "timelength"
    # print timelength

    if len(np.shape(observations)) != 1:
        numsamples = np.shape(observations)[0]
    else:
        numsamples = 1
    # initialization
    if numsamples > 1 :
        (pie,transmtrx,obsmtrx )= initialize_with_kmeans(observations,numstates,numsamples,exmodel)
    else:
        (pie,transmtrx,obsmtrx )= initializeparameters_closetoreality(observations,numstates,numsamples,exmodel)
    # (pie,transmtrx,obsmtrx )= initializeparameters(observations,numstates,numsamples)
    # (pie,transmtrx,obsmtrx )= initializeparameters_closetoreality(observations,numstates,numsamples,exmodel)
    # (pie,transmtrx,obsmtrx ) = clipvalues_prevunderflow_small(pie,transmtrx,obsmtrx)
    # print "init observation matrix is "
    # print obsmtrx
    likelihoods = []
    counter = 0
    diffprobproduct = 1.0
    # pie = exmodel.pie
    # transmtrx = exmodel.transitionmtrx
    # obsmtrx = exmodel.obsmtrx
    if numsamples == 1:
        prevlogobservation = 2.0
    else:
        prevlogobservation = [2.0] * numsamples
    while(diffprobproduct > conv_threshold):
        (gammas,kissies,logobservations) = E_step(pie,transmtrx,obsmtrx,observations)
        (pie,transmtrx,obsmtrx) = M_step(gammas,kissies,observations,hard)
        # print "pie is"
        # print pie
        # print "obsmtrx is"
        print obsmtrx
        # print "transmtrx is"
        # print transmtrx

        likelihoods.append(np.log(logobservations))
        counter +=1
        if len(np.shape(observations)) == 1:
            diffprobproduct = abs((prevlogobservation - logobservations))
        else:
            diffprobproduct = abs(np.max(prevlogobservation - logobservations))
        # print diffprobproduct
        prevlogobservation = logobservations
    # print counter
    title = "likelihoodtrend.png"
    if numsamples ==1:
        plt.plot(range(counter),likelihoods)
    else:
        plt.plot(range(counter),np.mean(likelihoods,axis=1,dtype=np.float64))
    plt.savefig(title)
    
    return (pie,transmtrx,obsmtrx) 

# def main():
#     exmodel = hmmgaussian(3,2,10,10, True)
#     numstates = exmodel.numofstates
#     observations = exmodel.observations
#     # print "sequence of states is"
#     # print exmodel.seqofstates
#     print "real mean is"
#     print np.mean(observations)
#     print "real std is"
#     print np.std(observations)
#     # hard = True
#     hard = False
#     print "realpie is "
#     print exmodel.pie
#     print "realtrans"
#     print exmodel.transitionmtrx
#     print "real obsmtrx"
#     print exmodel.obsmtrx
#     sensitivity = 16
#     threshold_exponential = 10 ** (-sensitivity)
#     (pie,transmtrx,obsmtrx) = Baumwelchcont(observations,numstates,exmodel,hard,threshold_exponential)
#     # (pie,transmtrx,obsmtrx) = clipvalues_prevunderflow_small(pie,transmtrx,obsmtrx)
#     piedist = np.linalg.norm(pie - exmodel.pie ) / float(numstates)
#     transdist = np.linalg.norm(transmtrx - exmodel.transitionmtrx) / float(numstates **2)
#     # obsdist = np.linalg.norm(obsmtrx - exmodel.obsmtrx) / float( numstates)
#     print "realpie is "
#     print exmodel.pie
#     print "estimated pie is"
#     print pie
#     print "realtrans"
#     print exmodel.transitionmtrx
#     print "estimated transition matrix is"
#     print transmtrx
#     print "real obsmtrx"
#     print exmodel.obsmtrx
#     print "estimated observation matrix is"
#     print obsmtrx

# main()