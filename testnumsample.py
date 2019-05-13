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
from scipy.optimize import linear_sum_assignment


def stabilityrep(ordpiis,ordtransmats,emisssmtrxs):
    avgordtransmat = np.mean(ordtransmats, axis = 0)
    avgordpii = np.mean(ordpiis, axis = 0)
    avgemisssmtrx = np.mean(emisssmtrxs, axis = 0)
    nruns = np.shape(ordpiis)[0]
    numofstates = np.shape(ordpiis)[1]
    numobsercase = np.shape(emisssmtrxs)[2]
    # calculating the average matrices 
    costmatrices = np.empty((nruns,nruns,numofstates,numofstates))
    bestassignment = np.empty((nruns,nruns,numofstates))
    flatbestassignment = np.empty((nruns*nruns,numofstates),dtype= int)
    for i in range(nruns):
        for j in range(nruns):
            for k in range(numofstates):
                for l in range(numofstates):
                    costmatrices[i,j,k,l] = np.dot(ordtransmats[i,k,:],ordtransmats[j,l,:])
            row,bestassignment[i,j,:] = linear_sum_assignment(costmatrices[i,j,:,:])
            print bestassignment[i,j,:]
    # finding the best assignment across all the pair
    k= 0
    for i in range(nruns):
        for j in range(nruns):
            flatbestassignment[k,:] = bestassignment[i,j,:]
            k +=1
        
    print "you should be looking here"
    print flatbestassignment
            
    diffordtransmat = np.empty((numofstates,numofstates))
    diffordpii = np.empty((1,numofstates))
    diffemissmtrx = np.empty((numofstates,numobsercase))
    k = 0
    for i in range(nruns):
        for j in range(nruns):
            diffordtransmat += np.absolute(ordtransmats[i,:,:] - ordtransmats[j,flatbestassignment[k],flatbestassignment[k]])
            diffordpii += np.absolute(ordpiis[i,:] - ordpiis[j,flatbestassignment[k]])
            diffemissmtrx += np.absolute(emisssmtrxs[i,:,:] - emisssmtrxs[j,flatbestassignment[k],:])
            k += 1
    diffordtransmat /= float(nruns*nruns)
    diffordpii /= float(nruns*nruns)
    diffemissmtrx /= float(nruns*nruns)
    
    print "difference in pie"
    print diffordpii
    print "difference in transition matrix"
    print diffordtransmat
    print "difference in emission matrix"
    print diffemissmtrx
    
    print "mean of differences in pie "
    print np.mean(diffordpii)
    print "mean of differences in transitions "
    print np.sum(diffordtransmat) / float(numofstates * numofstates)
    print "mean of difference in emission matrix"
    print np.sum(diffemissmtrx) / float(numofstates * numobsercase)
    
    # varavgordpii = float(np.sum(diffordpii)) / float(numofstates * nruns)
    # varavgovlappii = float(np.sum(diffovlappii)) /  float(numofstates * nruns)
    # varavgordtransmat = float(np.sum(diffordtransmat)) / float(nelemsmtrx * nruns)
    # varavgovlaptransmat = float(np.sum(diffovlaptransmat)) / float(nelemsmtrx * nruns)

    # print varavgordpii,varavgovlappii,varavgordtransmat,varavgovlaptransmat

    # return (varavgordpii,varavgordpii,varavgordtransmat,varavgordtransmat)
def generateobs(numsamples,pie,obsmtrx,obserlength,numofstates,numofobsercases,transitionmtrx):
    # 800,pie,obsmtrx,seqlenght,numstate,numobsercase,transmtrx
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
    # numsamplescases = [200,500,800,1000,1200,1400,1600,1800,2000,3000,4000]
    # fixed stuff 
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
    numsamplecases = [200,400,600,800,1000,1200,1400,1600,1800,2000,2400,2600,2800,3000,3400,3600,4000]
    
    numstate = 8
    numobsercase = 10
    nruns = len(numsamplecases)
    transmats = np.empty((nruns,numstate,numstate))
    piis = np.empty((nruns,numstate))
    emisssmtrxs = np.empty((nruns,numstate,numobsercase))
    trainaccs = []
    testaccs = []
    naivecases = []
    naivetestaccs = []
    seqlenght = 20
    (testobservations,testseqofstates) = generateobs(800,pie,obsmtrx,seqlenght,numstate,numobsercase,transmtrx)
    halfstates = int(numstate) / 2
    truestate2label = np.random.permutation(([0] * halfstates ) + [1] * (numstate - halfstates) )
    testclasses = np.array([truestate2label[int(j)] for j in list(testseqofstates[:,-1])])    
    for i in range(len(numsamplecases)):
        numsamples =numsamplecases[i]
        # this is dummy not really used 
        exmodel = hmmforward(numstate,numobsercase,0.4,seqlenght,numsamples)
        
        # Fixing stuff to make sure models are comparable
        if i == 0:
            (trainobservations,trainseqofstates) = generateobs(numsamples,pie,obsmtrx,seqlenght,numstate,numobsercase,transmtrx)
        else:
            (extraobs,extraseq) = generateobs(numsamplecases[i]-numsamplecases[i-1],pie,obsmtrx,seqlenght,numstate,numobsercase,transmtrx)
            trainobservations = np.concatenate((trainobservations,extraobs))
            trainseqofstates = np.concatenate((trainseqofstates,extraseq))
        # pie = exmodel.pie
        # pie = np.array([0.5,0.5])
        # transmtrx = exmodel.transitionmtrx
        # obsmtrx = exmodel.obsmtrx
        # seqofstates = exmodel.seqofstates
        trainclasses = np.array([truestate2label[int(j)] for j in list(trainseqofstates[:,-1])])

        # Training an HMM model to learn the sequence of states, and later using the majority class of samples who end up in each state as a mapping from that 
        # state to the labels.
        supposednumstates = numstate
        # trainobservations = observations[:numtraining,:]
        # testobservations = observations[numtraining:,:]
        (learnedpie,learnedtransmtrx,learnedobsmtrx)  = Baumwelch(trainobservations,supposednumstates,numobsercase,numsamples,exmodel)
        piis[i,:] = learnedpie 
        transmats[i,:,:] = learnedtransmtrx 
        emisssmtrxs[i,:,:] = learnedobsmtrx 
        (trainoptzis,traindeltas) = viterbi(learnedtransmtrx,learnedobsmtrx,learnedpie,trainobservations)
        (testoptzis,testdeltas) = viterbi(learnedtransmtrx,learnedobsmtrx,learnedpie,testobservations)
        TrainFinalState = list(trainoptzis[:,-1])
        TestFinalState = list(testoptzis[:,-1])
        # Finding the mappings from supposed states to the labels
        estimatedstate2label = [0] * supposednumstates
        for state in range(supposednumstates):
            counts = [0,0]
            for i in range(numsamples):
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
        testpredclasses = np.array([estimatedstate2label[int(j)] for j in TestFinalState])
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
    plt.plot(numsamplecases,trainaccs,'r',label = 'Training Accuracy')
    plt.plot(numsamplecases,testaccs,'b', label = 'Test Accuracy')
    plt.plot(numsamplecases,naivecases,'k',label = 'Majority Classifier Accuracy')
    plt.xlabel("Number of Samples")
    plt.ylabel("Accuracy")
    plt.legend(loc='upper left')
    plt.show()
    title1 = "AccvsNumsamples" + ".png"
    plt.savefig(title1)
    plt.close()
    plt.plot(numsamplecases,np.array(testaccs)-np.array(naivetestaccs),'g',label = 'Improvement over Majority classifier')
    plt.xlabel("Number of Samples")
    plt.ylabel("Improvement over Majority classifier")
    plt.legend(loc='upper left')
    title1 = "DiffNumsamples" + ".png"
    plt.savefig(title1)
    plt.close()
    stabilityrep(piis,transmats,emisssmtrxs)

generatedata()