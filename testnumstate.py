import numpy as np,numpy.random
from init_forward import hmmforward
from scipy import stats
from forward import forward
from backward import backward
from forward_backward import forward_backward
from sklearn.preprocessing import normalize
from viterbi import viterbi
from Baumwelch import Baumwelch
import matplotlib.pyplot as plt
def generateobs(numsamples,pie,obsmtrx,obserlength,numofstates,numofobsercases,transitionmtrx):
    observations = 2.22044604925e-16 * np.ones((numsamples,obserlength),dtype = numpy.int8)
    seqofstates = 2.22044604925e-16 * np.ones((numsamples,obserlength))
    for samnum in range(numsamples):
        elements = range(numofstates)
        initialstate = np.random.choice(elements, 1, p=pie)[0]
        elements = range(numofobsercases)
        (observations)[samnum,0] = np.random.choice(elements, 1, p=list(obsmtrx[initialstate,:]))[0]
        prevstate = initialstate
        (seqofstates)[samnum,0] = initialstate
        for i in range(1,obserlength):
            elements = range(numofstates)
            nextstate = np.random.choice(elements, 1, p=transitionmtrx[prevstate,:])[0]
            elements = range(numofobsercases)
            (observations)[samnum,i] = (np.random.choice(elements, 1, p=list(obsmtrx[nextstate,:])))[0]
            (seqofstates)[samnum,i] = (nextstate)
            prevstate = nextstate
    return (observations,seqofstates)

def generatedata():
    # Genearating data and assigning random labels to samples in the dataset
    numsamplescases = [200,500,800,1000,1200,1400,1600,1800,2000,3000,4000]
    trainaccs = []
    testaccs = []
    naivecases = []
    naivetestaccs = []
    for case in numsamplescases:
        numsamples = case
        numstate = 8
        numobsercase = 10
        seqlenght = 20
        exmodel = hmmforward(numstate,numobsercase,0.4,seqlenght,numsamples)
        
        # Fixing stuff to make sure models are comparable

        pie = np.array([0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.125])
        obsmtrx = np.array([[1.66314928e-01, 7.42267394e-02, 8.02335414e-02, 1.01325028e-01,8.48965206e-02, 1.46916053e-01, 2.77734983e-02, 1.72648415e-01,3.14561463e-02, 1.14209130e-01],
       [9.07996537e-06, 3.32933668e-01, 3.36480754e-02, 7.80739676e-04,
        4.81344112e-01, 1.75604999e-04, 1.50901388e-01, 2.05111895e-06,
        3.92088962e-06, 2.01359589e-04],
       [1.08934113e-02, 5.94799852e-03, 1.53897369e-02, 1.50836557e-03,
        8.90684020e-03, 7.02942441e-02, 6.40486924e-02, 4.78843623e-01,
        1.58312943e-02, 3.28335794e-01],
       [1.77209227e-02, 2.50183131e-03, 2.70488652e-02, 2.43412287e-01,
        6.57008914e-05, 8.53424081e-05, 6.06230655e-02, 1.80261626e-01,
        4.68202484e-01, 7.78744459e-05],
       [6.39279127e-02, 3.10493889e-03, 3.55676657e-03, 3.53068612e-01,
        3.63236835e-03, 8.99271540e-04, 2.24932293e-01, 1.22618869e-01,
        2.22774527e-01, 1.48444106e-03],
       [2.14212726e-03, 1.95844888e-06, 1.21031314e-01, 8.81229342e-03,
        2.27504067e-02, 3.78747490e-01, 3.96316413e-05, 2.07278627e-02,
        4.45680794e-01, 6.61224301e-05],
       [7.26858152e-06, 9.92654290e-03, 1.87307407e-01, 1.11461493e-01,
        5.43787242e-04, 9.30317675e-04, 2.76630589e-01, 2.09372000e-01,
        2.03711199e-01, 1.09395658e-04],
       [6.93884556e-02, 1.02831618e-01, 3.92342845e-04, 2.47995123e-02,
        2.34924469e-01, 5.16842486e-01, 1.43358037e-02, 7.62407765e-06,
        5.36069680e-05, 3.64240821e-02]])
        # observations = exmodel.observations
        transmtrx = np.array([[0.1,0.2,0.3,0.05,0.05,0.1,0.15,0.05],
                                   [0.05,0.05,0.1,0.15,0.05,0.1,0.2,0.3],[0.1,0.05,0.05,0.1,0.15,0.2,0.3,0.05],
                                   [0.05,0.05,0.1,0.1,0.2,0.3,0.15,0.05],[0.1,0.2,0.3,0.15,0.05,0.05,0.05,0.1,],
                                   [0.1,0.2,0.05,0.05,0.1,0.3,0.15,0.05],[0.1,0.05,0.05,0.1,0.1,0.3,0.25,0.05],
                                   [0.1,0.2,0.05,0.1,0.15,0.05,0.1,0.25]])
        (observations,seqofstates) = generateobs(numsamples,pie,obsmtrx,seqlenght,numstate,numobsercase,transmtrx)
        # pie = exmodel.pie
        # pie = np.array([0.5,0.5])
        halfstates = int(numstate) / 2
        truestate2label = np.random.permutation(([0] * halfstates ) + [1] * (numstate - halfstates) )
        # transmtrx = exmodel.transitionmtrx
        # obsmtrx = exmodel.obsmtrx
        # seqofstates = exmodel.seqofstates
        numtraining = int(0.8 * numsamples)    
        trainclasses = np.array([truestate2label[int(i)] for i in list((seqofstates)[:numtraining,-1])])
        testclasses = np.array([truestate2label[int(j)] for j in list((seqofstates)[numtraining:,-1])])

        # Training an HMM model to learn the sequence of states, and later using the majority class of samples who end up in each state as a mapping from that 
        # state to the labels.
        supposednumstates = numstate
        trainobservations = observations[:numtraining,:]
        testobservations = observations[numtraining:,:]
        (learnedpie,learnedtransmtrx,learnedobsmtrx)  = Baumwelch(trainobservations,supposednumstates,numobsercase,numtraining,exmodel)
        (trainoptzis,traindeltas) = viterbi(learnedtransmtrx,learnedobsmtrx,learnedpie,trainobservations)
        (testoptzis,testdeltas) = viterbi(learnedtransmtrx,learnedobsmtrx,learnedpie,testobservations)
        TrainFinalState = list(trainoptzis[:,-1])
        TestFinalState = list(testoptzis[:,-1])
        # Finding the mappings from supposed states to the labels
        estimatedstate2label = [0] * supposednumstates
        statecounts = [0] * supposednumstates
        for state in range(supposednumstates):
            counts = [0,0]
            for i in range(numtraining):
                if int(TrainFinalState[i]) == state:
                    counts[trainclasses[i]] +=1
            print counts
            classak = np.argmax(counts)
            estimatedstate2label[state] = classak

        # print "true mapping from states to classes is"
        # print truestate2label
        # print "estimated mapping from states to classes is"
        # print estimatedstate2label
        # Reporting test and training accuracy
        trainpredclasses = np.array([estimatedstate2label[int(i)] for i in TrainFinalState])
        testpredclasses = np.array([estimatedstate2label[int(i)] for i in TestFinalState])
        trainacc = float(np.sum(trainclasses == trainpredclasses ))/ float(len(trainclasses))
        testacc = float(np.sum(testclasses == testpredclasses ))/ float(len(testclasses))
        # print "training accuracy is"
        # print trainacc
        # print "test accuracy is"
        # print testacc
        # print "are you beating guessing the majority in test or not?"
        majority = float(np.sum(trainclasses)) / float(len(trainclasses))
        trainnaiveacc = max(majority, 1- majority)
        majorityclass = int(np.argmax([1- majority,majority]))
        testmajorityclassifieracc = float(np.sum(testclasses == np.array([majorityclass] *len(testclasses) ) ))/ float(len(testclasses))
        testaccs.append(testacc)
        trainaccs.append(trainacc)
        naivecases.append(trainnaiveacc)
        naivetestaccs.append(testmajorityclassifieracc)
    print "train"
    print trainaccs
    print "test"
    print testaccs
    print "naiveaccs"
    print naivecases
    plt.close()
    plt.plot(numsamplescases,trainaccs,'r',label = 'Training Accuracy')
    plt.plot(numsamplescases,testaccs,'b', label = 'Test Accuracy')
    plt.plot(numsamplescases,naivecases,'k',label = 'Majority Classifier Accuracy')
    plt.xlabel("Number of Samples")
    plt.ylabel("Accuracy")
    plt.legend(loc='upper left')
    plt.show()
    title1 = "AccvsNumsamples" + ".png"
    plt.savefig(title1)
    plt.close()
    plt.plot(numsamplescases,np.array(testaccs)-np.array(naivetestaccs),'g',label = 'Improvement over Majority classifier')
    plt.xlabel("Number of Samples")
    plt.ylabel("Improvement over Majority classifier")
    plt.legend(loc='upper left')
    title1 = "DiffNumsamples" + ".png"
    plt.savefig(title1)
    plt.close()

generatedata()