import numpy as np,numpy.random
from init_forward import hmmforward
from scipy import stats
from forward import forward
from backward import backward
from forward_backward import forward_backward
from sklearn.preprocessing import normalize
from viterbi import viterbi
from Baumwelch import Baumwelch

def generatedata():
    numsamples = 50
    numstate = 8
    numobsercase = 10
    seqlenght = 50
    exmodel = hmmforward(numstate,numobsercase,1,seqlenght,numsamples)
    observations = exmodel.observations
    pie = exmodel.pie
    halfstates = int(exmodel.numofstates) / 2
    truestate2label = np.random.permutation(([0] * halfstates ) + [1] * (exmodel.numofstates - halfstates) )
    transmtrx = exmodel.transitionmtrx
    obsmtrx = exmodel.obsmtrx
    seqofstates = exmodel.seqofstates
    numtraining = int(0.8 * numsamples)    
    trainclasses = np.array([truestate2label[int(i)] for i in list((exmodel.seqofstates)[:numtraining,-1])])
    testclasses = np.array([truestate2label[int(j)] for j in list((exmodel.seqofstates)[numtraining:,-1])])
    supposednumstates = exmodel.numofstates
    trainobservations = exmodel.observations[:numtraining,:]
    testobservations = exmodel.observations[numtraining:,:]
    (pie,transmtrx,obsmtrx)  = Baumwelch(trainobservations,supposednumstates,exmodel.numofobsercases,numtraining,exmodel)
    (trainoptzis,deltas) = viterbi(transmtrx,obsmtrx,pie,trainobservations)
    (testoptzis,deltas) = viterbi(transmtrx,obsmtrx,pie,testobservations)
    TrainFinalState = list(trainoptzis[:,-1])
    TestFinalState = list(testoptzis[:,-1])
    estimatedstate2label = [0] * supposednumstates
    statecounts = [0] * supposednumstates
    for state in range(supposednumstates):
        counts = [0,0]
        for i in range(numtraining):
            if int(TrainFinalState[i]) == state:
                counts[trainclasses[i]] +=1
        classak = np.argmax(counts)
        estimatedstate2label[state] = classak


    # print "true mapping from states to classes is"
    # print truestate2label
    # print "estimated mapping from states to classes is"
    # print estimatedstate2label
        
    trainpredclasses = np.array([estimatedstate2label[int(i)] for i in TrainFinalState])
    testpredclasses = np.array([estimatedstate2label[int(i)] for i in TestFinalState])
    # print testclasses
    # print testpredclasses
    # print (trainclasses == trainpredclasses)
    trainacc = float(np.sum(trainclasses == trainpredclasses ))/ float(len(trainclasses))
    testacc = float(np.sum(testclasses == testpredclasses ))/ float(len(testclasses))
    print "training accuracy is"
    print trainacc
    print "test accuracy is"
    print testacc
    # (gammas,betas,alphas,log_prob_most_likely_seq,most_likely_seq,forward_most_likely_seq,forward_log_prob_most_likely_seq,Ziis,logobservations) = forward_backward(transmtrx,obsmtrx,pie,observations)
    # print "forward_backward acc"
    # print np.sum(seqofstates==most_likely_seq) / (float(exmodel.obserlength) * float(exmodel.numsamples))
    # print "forward acc"
    # print np.sum(seqofstates==forward_most_likely_seq) / (float(exmodel.obserlength) * float(exmodel.numsamples))
    # print "forward_backward prob"
    # print log_prob_most_likely_seq
    # print "forward prob"
    # print forward_log_prob_most_likely_seq
    # print "forward_backward is more certain at each time point"
    # numwins = 0
    # for i in range(exmodel.obserlength):
    #     if max(alphas[i,:]) <= max(gammas[i,:]):
    #         numwins +=1
    # print numwins / float(exmodel.obserlength) 

    # (mlpath,deltas) = viterbi(transmtrx,obsmtrx,pie,observations)
    # # print mlpath
    # print "viterbi similar to reality"
    # print np.sum(seqofstates==mlpath) / (float(exmodel.obserlength) * float(exmodel.numsamples))
    # print "viterbi similarity to forward_backward"
    # print np.sum(mlpath==most_likely_seq) / (float(exmodel.obserlength) * float(exmodel.numsamples))
    # print "viterbi similarity to forward seq"
    # print np.sum(mlpath==forward_most_likely_seq) / (float(exmodel.obserlength) * float(exmodel.numsamples))
generatedata()