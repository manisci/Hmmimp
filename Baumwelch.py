import numpy as np,numpy.random
from init_forward import hmmforward
from scipy import stats
from forward import forward
from backward import backward
from forward_backward import forward_backward

def normalize(u):
    Z = np.sum(u)
    if Z==0:
        v = u / (Z+ 2.22044604925e-16)
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
    

def E_step(pie,transmtrx,obsmtrx,observations):
    (gammas,betas,alphas,log_prob_most_likely_seq,most_likely_seq,forward_most_likely_seq,forward_log_prob_most_likely_seq) = \
    forward_backward(transmtrx,obsmtrx,pie,observations)
    numstate = np.shape(transmtrx)[0]
    timelength = len(observations)
    kissies = np.empty((timelength,numstate,numstate))
    for t in range(timelength-1):
        for q in range(numstate):
            for s in range(numstate):
                kissies[t,q,s] = alphas[t,q] * transmtrx[q,s] * obsmtrx[s,observations[t+1]] * betas[t+1,s]
        (kissies[t,:,:],dummy) = normalize( kissies[t,:,:])
    return (gammas,kissies)

def M_step(gammas,kissies,numobscases,observations):
    numstate = np.shape(gammas)[1]
    timelength = np.shape(gammas)[0]
    newpie = np.empty((numstate))
    newtransmtrx =np.empty((numstate,numstate))
    newobsmtrx = np.empty((numstate,numobscases))
    for i in range(numstate):
        newpie[i] = gammas[0,i]
    for q in range(numstate):
        denominator = np.sum(gammas[:timelength-1,q])
        for s in range(numstate):
            newtransmtrx[q,s] = np.sum(kissies[:timelength-1,q,s]) / float(denominator)
    for state in range(numstate):
        denom = np.sum(gammas[:,state])
        for obs in range(numobscases):
            numerator = 0.0
            for time in range(timelength):
                if observations[time] == obs:
                    numerator += gammas[time,state]
            newobsmtrx[state,obs] = numerator / float(denom)
    return (newpie,newtransmtrx,newobsmtrx)



def Baumwelch(observations,numstates,numobscases,numsamples):
    ''' you can convert the observations and obsmtrx all together into one matrix
    called soft evidence which is a K * T matrix by using the corresponding
    distribution across all the states for each time point and use that instead'''
    # initialization
    (pie,transmtrx,obsmtrx )= initializeparameters(observations,numstates,numobscases,numsamples)
    conv_threshold = 2.22044604925e-16
    diff_consec_params = 100
    counter = 0
    while(diff_consec_params > conv_threshold):
        (gammas,kissies) = E_step(pie,transmtrx,obsmtrx,observations)
        prevpie = np.copy(pie)
        prevobsmtrx = np.copy(obsmtrx)
        prevtransmtrx = np.copy(transmtrx)
        (pie,transmtrx,obsmtrx) = M_step(gammas,kissies,numobscases,observations)
        piedist = np.linalg.norm(pie - prevpie ) / float(numstates)
        transdist = np.linalg.norm(transmtrx - prevtransmtrx) / float(numstates **2)
        obsdist = np.linalg.norm(obsmtrx - prevobsmtrx) / float(numobscases * numstates)
        diff_consec_params = piedist + transdist + obsdist
        counter +=1
    print "went this much in loop"
    print counter
    return (pie,transmtrx,obsmtrx) 

    # numstates = np.shape(transmtrx)[0]
    # timelength = np.shape(observations)[0]
    # deltas = np.empty((timelength,numstates))
    # optzis = np.empty((timelength,1))
    # As = np.empty((timelength,numstates))
    # (deltas[0,:] ,Z) = normalize((np.multiply(obsmtrx[:,int(observations[0])],pie)))
    # otherzero = np.argmax(deltas[0,:])
    # for t in range(1,timelength):
    #     # set A here
    #     for j in range(numstates):
    #         # print deltas[t-1,:] * transmtrx[:,j] * obsmtrx[j,int(observations[t])]
    #         (normed,Z) = normalize(deltas[t-1,:] * transmtrx[:,j] * obsmtrx[j,int(observations[t])])
    #         # print normed
    #         As[t,j] = int(np.argmax(normed))
    #         cands = np.empty((numstates,1))
    #         for i in range(numstates):
    #             cands[i] = deltas[t-1,i] *(transmtrx[i,j]) *(obsmtrx[j,int(observations[t])])
    #         deltas[t,j] = max(cands)
    #     (deltas[t,:],Z) = normalize(deltas[t,:])
    # optzis[timelength-1] = int(np.argmax(deltas[timelength-1,:]))
    # for k in range(timelength-2,-1,-1):
    #     optzis[k] = As[k+1,int(optzis[k+1])]
    return (pie,transmtrx,obsmtrx )

def main():
    exmodel = hmmforward(5,10,1,50)
    numstates = exmodel.numofstates
    numobscases = exmodel.numofobsercases
    numsamples = 1
    observations = exmodel.observations
    (pie,transmtrx,obsmtrx ) =  Baumwelch(observations,numstates,numobscases,numsamples)
    piedist = np.linalg.norm(pie - exmodel.pie ) / float(numstates)
    transdist = np.linalg.norm(transmtrx - exmodel.transitionmtrx) / float(numstates **2)
    obsdist = np.linalg.norm(obsmtrx - exmodel.obsmtrx) / float(numobscases * numstates)
    print piedist,transdist,obsdist


main()