import numpy as np
import scipy
import operator
import matplotlib.pyplot; 
matplotlib.pyplot.switch_backend('agg')
import seaborn as sns ; sns.set(style="ticks", color_codes=True)
import os
import csv
import pandas as pd
from scipy import stats
from sklearn.neighbors import KNeighborsRegressor
# from hmmlearn import hmm     
import warnings
from sklearn.model_selection import cross_val_score
from babel.util import missing
from IPython.core.magics import pylab
from boto.ec2.cloudwatch import dimension
warnings.filterwarnings("ignore", category=DeprecationWarning)
from sklearn.externals import joblib
import sklearn
from scipy.optimize import linear_sum_assignment
import matplotlib.collections
import statsmodels.api as sm
from sklearn.decomposition import PCA
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.cov_struct import (Exchangeable,
    Independence,Autoregressive)
from statsmodels.genmod.families import Poisson
import statsmodels.formula.api as smf
from sklearn import linear_model
from sklearn import svm
import matplotlib.pyplot as plt
import math
from numpy import dual
import sys
# /Users/manisci/Library/Mobile Documents/com~apple~CloudDocs/Documents/research/Winbraek18/Hmmimp/
sys.path.insert(0, '/Users/manisci/Library/Mobile Documents/com~apple~CloudDocs/Documents/research/Winbraek18/Hmmimp/')
# sys.path.insert(0, '/Users/manisci/Documents/research/Winbraek18/Hmmimp/')
import get_seq_statescont
import init_gaussian
np.set_printoptions(precision=3,suppress=True)

def swapaxes(matrix):
    # matrix = np.swapaxes(matrix, 0, -1)
    matrix = np.swapaxes(matrix, -1, -2)
    return matrix
def returnoutputvar(filename):
    content = open(filename, 'r').read()
    return content.split()
def removeinvalidptsfromdict(diction,allmiss):
    '''
    removing the invalid patients and coordinating the rest of patients to have the correct index
    '''
    for ind in range(len(allmiss)-1):
            diction.pop(allmiss[ind])
            for val in sorted(diction.keys()):
                if val in range(allmiss[ind] +1,allmiss[ind+1]):
                    diction[val - (ind+1)]= diction[val]
                    diction.pop(val)
    diction.pop(allmiss[-1])
    for val in sorted(diction.keys()):
        if val > allmiss[-1]:
                diction[val - len(allmiss)]= diction[val]
                diction.pop(val)
    return diction
def main():
    '''
    Runs the HMM model on a single combined or original ICU type.
    '''
    # Uses already available files to extract values of these normal medical meterics
    saps =  returnoutputvar("saps.txt")
    apache =  returnoutputvar("apache.txt")
    mpm =returnoutputvar("mpm.txt")
    sofa  = returnoutputvar("sofa.txt")
    recid = returnoutputvar("recid.txt")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore",category=DeprecationWarning)
    # Change current directory to train folder to read the files
    os.chdir("..")
    os.chdir(os.path.abspath(os.curdir) + "/train")
    # Extracting top 7 features mentioned in the paper 
    (wholefeat,featmtrx1,featmtrx2,featmtrx3,featmtrx4,\
    featfirst1,featfirst2,featfirst3,featfirst4,\
    featsecond1,featsecond2,featsecond3,featsecond4,\
    paticutypedictindex, heartratedict,ages,genders,WBCdict,tempdict,GCSdict,glucosedict,NIDSdict,urinedict,AVGmeanchangefeats,AVGmedianchangefeats,recidees) = xtracttopfeat() 

    # Extracting length of stay of patients
    (los,los1,los2,los3,los4,invalids,losrecids) = xtrlenofstay(paticutypedictindex)
    # dealing with patients who have no LOS, who have missign saps score
    recid = [int(j) for j in recid]
    misak = (set(list(recidees)) - set(list(recid)))
    misakidxs = [recidees.index(i) for i in misak]
    invalididxs = [recidees.index(i) for i in invalids]
    allmiss = misakidxs + invalididxs
    allmiss = sorted(allmiss)
    newsaps = []
    newapache =  []
    newmpm =[]
    newsofa  = []
    # coordinating the dictionaries obtained before removing invalid patients 
    for ind in range(len(allmiss)-1):
        for icuind in range(1,5):
            if allmiss[ind] in paticutypedictindex[icuind]:
                paticutypedictindex[icuind].remove(allmiss[ind])
            if allmiss[ind+1] in paticutypedictindex[icuind]:
                paticutypedictindex[icuind].remove(allmiss[ind+1])
            for val in paticutypedictindex[icuind]:
                if val in range(allmiss[ind] +1,allmiss[ind+1]):
                    paticutypedictindex[icuind][(paticutypedictindex[icuind]).index(val)] = val - (ind+1)
    for icuind in range(1,5):
        for val in paticutypedictindex[icuind]:
            if val > allmiss[-1]:
                paticutypedictindex[icuind][(paticutypedictindex[icuind]).index(val)] = val - (len(allmiss))
    
    realwholefeat = np.empty((3937,35))
    reallos = [los[i] for i in range(len(los)) if losrecids[i] not in misak and losrecids[i] not in invalids ]
    realages = []
    realgenders = []
    j = 0
    l = 0
    # fixing age and genders given the omitted patients
    for k in recidees:
        if k not in misak and k not in invalids :
            realwholefeat[j,:] = wholefeat[l,:]
            j +=1
            realages.append(ages[l])
            realgenders.append(genders[l])
        l +=1
    j = 0
    l = 0
    for k in recid:
        if k not in invalids :
            newsaps.append(float(saps[l]))
            newapache.append(float(apache[l]))
            newmpm.append(float(mpm[l]))
            newsofa.append(float(sofa[l]))
        l +=1
    wholefeat = realwholefeat
    los = reallos
    ages = realages
    genders = realgenders
    allpats = 0
    maxpat = 0
    for i in range(1,5):
        allpats += len(paticutypedictindex[i])
        maxpat = max(max(paticutypedictindex[i]),maxpat)
    # fixing dictionary values for the removed patients
    heartratedict = removeinvalidptsfromdict(heartratedict,allmiss)
    WBCdict = removeinvalidptsfromdict(WBCdict,allmiss)
    tempdict = removeinvalidptsfromdict(tempdict,allmiss)
    GCSdict = removeinvalidptsfromdict(GCSdict,allmiss)
    glucosedict = removeinvalidptsfromdict(glucosedict,allmiss)
    NIDSdict = removeinvalidptsfromdict(NIDSdict,allmiss)
    urinedict = removeinvalidptsfromdict(urinedict,allmiss)
    # settting all values to the new ones
    saps = newsaps
    apache = newapache 
    mpm = newmpm 
    sofa = newsofa 
    outputfeats = np.column_stack((saps,apache,mpm,sofa))
    # list of the dicts that you want to include in the model
    listofalldicts = [heartratedict,WBCdict,tempdict,GCSdict,glucosedict,NIDSdict,urinedict] 
    nstates = range(2,9)
    overlapflag = False
    ordscores = [0] * len(nstates)
    ovlapscores = [0] *  len(nstates)
    ordAvgVarPatches = [0] * len(nstates)
    ordVarRadiiPatchesmean = [0] * len(nstates)
    ordVarRadiiPatchesmedian = [0] * len(nstates)
    ovlapAvgVarPatches = [0] * len(nstates)
    ovlapVarRadiiPatchesmean = [0] * len(nstates)
    ovlapVarRadiiPatchesmedian = [0] * len(nstates)
    overcases = [True]
    logcases = [False]
    icutypes = [int(sys.argv[1])]
    numofstatescases = [8]
    resolutions = [8.0]
    dictcases = {}
    models = ["firstlastprobs"]
    # name of the file to save the performance metrics in.
    realtitle = "sapspaperreal5" + ".csv"
    w = csv.writer(open(realtitle, "a"))
    w.writerow(["model " , "abbrevName","Overlapping " , "Log scale for LOS","Number of States","Time Resolution ", "ICUType " , "HMM RMSE ","Baseline RMSE","sapsRMSE ","alloutRMSE ""Win1","win2","win3"])
    for numofstates in numofstatescases:
        for over in overcases:
            for log in logcases:
                for model in models:
                    for resolution in resolutions:
                        for icutype in icutypes:
                            name = ""
                            if over:
                                name += "over"
                            else :
                                name += "nonoverlapping"
                            if log :
                                name += "logLOS"
                            else :
                                name += "normalLOS"
                            name += "numofstates="
                            name += str(numofstates)
                            name += "icutype="
                            listofalldicts = [heartratedict,WBCdict,tempdict,GCSdict,glucosedict,NIDSdict,urinedict]
                            resolutions = [resolution] * 7
                            (validpatientsindices,realfeatmtrxtrain1,realfeatmtrxtest1,KNNfeats,reallos1,inputHmmallVars,ovlapinputHmmallVars,trainindices,testindices) = generatetraintestsplit(listofalldicts,wholefeat,los,icutype,ages,genders,urinedict,paticutypedictindex,resolutions)
                            dictindices= range(7)
                            (ordscores[0],ordselalg,ordselcovartype,ovlapscores[0],ovlapselalg,ovlapselcovartype ,traininghmmfeats1,testhmmfeats1,ytrain1,ytest1,ordAvgVarPatches[0],ordVarRadiiPatchesmean[0],ordVarRadiiPatchesmedian[0],ovlapAvgVarPatches[0],ovlapVarRadiiPatchesmean[0],ovlapVarRadiiPatchesmedian[0],ordtransmat, ovlaptransmat , ordpii , ovlappii   ) = learnhmm (validpatientsindices,KNNfeats,reallos1,inputHmmallVars,ovlapinputHmmallVars,trainindices,testindices,dictindices,resolution,numofstates,icutype,over,model)
                            sapstrain = [saps[lam] for lam in trainindices]
                            sapstest = [saps[lamb] for lamb in testindices]
                            sapstrain = np.array(sapstrain)
                            sapstest = np.array(sapstest)
                            outputfeatstrain = outputfeats[trainindices,:]
                            outputfeatstest = outputfeats[testindices,:]
                            # setting hmm flag to true and running the linear regression                          
                            hmm = True
                            hmmmsescore = Linearregr(traininghmmfeats1, testhmmfeats1, ytrain1, ytest1,hmm,log,numofstates,resolution,icutype,model)
                            hmm = False    
                            balgmse = performbaseline(realfeatmtrxtrain1,ytrain1,realfeatmtrxtest1,ytest1)
                            baselinescore = Linearregr(realfeatmtrxtrain1, realfeatmtrxtest1, ytrain1, ytest1,hmm,log,numofstates,resolution,icutype,model)
                            hmm = True
                            sapsscore = Linearregr(sapstrain.reshape(-1, 1), sapstest.reshape(-1, 1), ytrain1, ytest1,hmm,log,numofstates,resolution,icutype,model)
                            success1 = int(hmmmsescore < baselinescore) 
                            success2 = int(balgmse< baselinescore)
                            success3 = int(hmmmsescore< sapsscore)
                            print "hmm"
                            print hmmmsescore
                            print "baseline"
                            print baselinescore
                            print "saps"
                            print sapsscore
                            # writes the statistics for this run to the file for later inspection
                            w.writerow([model,name ,over,log, numofstates,resolution,icutype,hmmmsescore ,baselinescore,balgmse,sapsscore,success1,success2,success3])
def heatmapofvitalschange(ovlaplosidx,nstates,ovlapinputtest):
    '''
    plots heatmap of changes across time for different features and different start end pairs
    '''
    clusters = ovlaplosidx.keys()
    for cluster in clusters:
        if len(ovlaplosidx[cluster]) > 0:
            mtrx = []
            for i in range(7):
                row = np.mean(ovlapinputtest[ovlaplosidx[cluster],:,i],axis = 0)
                mtrx.append(row) 
            plt.close()
            mtrx = scipy.stats.zscore(mtrx,axis = 1)
            ax = sns.heatmap(mtrx,yticklabels = [ "HR " ,"WBC  " , "Temp  ",  "GCS  ","Glucose ", "NIDBP ","Urine "],cmap="Greens")
            plt.xlabel("Time point Index")
            realtitle = "vitalsheatmap" + "cluster " + str(cluster) +  str(nstates)  +  ".png"
            (plt.savefig(realtitle))
def sortdiagonal(mtrx):
    diagonal = np.diag(mtrx)
    idx = np.argsort(diagonal)
    sorted = mtrx[idx,:][:,idx]
    return sorted
def stabilityrep(resolution,icutype,wholefeat,los,ages,genders,urinedict,paticutypedictindex,listofalldicts,numofstates,over,model,nruns):
    '''
    Shows how much stable the model is given different initialization 
    '''
    ordtransmats = np.empty((nruns,numofstates,numofstates))
    ordpiis = np.empty((nruns,numofstates))
    ovlaptransmats = np.empty((nruns,numofstates,numofstates))
    ovlappiis = np.empty((nruns,numofstates))
    avgordtransmat = np.empty((numofstates,numofstates))
    avgovlaptransmat = np.empty((numofstates,numofstates))
    avgordpii = np.empty((1,numofstates))
    avgovlappii = np.empty((1,numofstates))
    nelemsmtrx = numofstates * numofstates
    for i in range(nruns):
        resolutions = [resolution] * 7
        (validpatientsindices,realfeatmtrxtrain1,realfeatmtrxtest1,KNNfeats,reallos1,inputHmmallVars,ovlapinputHmmallVars,trainindices,testindices) = generatetraintestsplit(listofalldicts,wholefeat,los,icutype,ages,genders,urinedict,paticutypedictindex,resolutions)
        dictindices= range(7)
        (ordscores,ordselalg,ordselcovartype,ovlapscores,ovlapselalg,ovlapselcovartype ,traininghmmfeats1,testhmmfeats1,ytrain1,ytest1,ordAvgVarPatches,ordVarRadiiPatchesmean,ordVarRadiiPatchesmedian,ovlapAvgVarPatches, \
        ovlapVarRadiiPatchesmean,ovlapVarRadiiPatchesmedian ,ordtransmat, ovlaptransmat , ordpii , ovlappii  ) = \
        learnhmm (validpatientsindices,KNNfeats,reallos1,inputHmmallVars,ovlapinputHmmallVars,trainindices,testindices,dictindices,resolution,numofstates,icutype,over,model)
        ordtransmats[i,:,:] = sortdiagonal(ordtransmat)
        ovlaptransmats[i,:,:] = sortdiagonal(ovlaptransmat)
        ordpiis[i,:] = sortdiagonal(ordpii)
        ovlappiis[i,:] = sortdiagonal(ovlappii)
    for i in range(numofstates):
        avgordpii = np.mean(ordpiis[:,i])
        avgovlappii = np.mean(ovlappiis[:,i])            
        for j in range(numofstates):
            avgordtransmat[i,j] = np.mean(ordtransmats[:,i,j])
            avgovlaptransmat[i,j] = np.mean(ovlaptransmats[:,i,j])
    diffordtransmat = np.empty((numofstates,numofstates))
    diffovlaptransmat = np.empty((numofstates,numofstates))
    diffordpii = np.empty((1,numofstates))
    diffovlappii = np.empty((1,numofstates))
    for i in range(nruns):
        diffordtransmat += np.absolute(ordtransmats[i,:,:] - avgordtransmat )
        diffovlaptransmat += np.absolute(ovlaptransmats[i,:,:] - avgovlaptransmat )
        diffordpii += np.absolute(ordpiis[i,:] - avgordpii )
        diffovlappii += np.absolute(ovlappiis[i,:] - avgovlappii )
    varavgordpii = float(np.sum(diffordpii)) / float(numofstates * nruns)
    varavgovlappii = float(np.sum(diffovlappii)) /  float(numofstates * nruns)
    varavgordtransmat = float(np.sum(diffordtransmat)) / float(nelemsmtrx * nruns)
    varavgovlaptransmat = float(np.sum(diffovlaptransmat)) / float(nelemsmtrx * nruns)
    return (varavgordpii,varavgordpii,varavgordtransmat,varavgordtransmat)
def generatetraintestsplit(listofalldicts,featmtrx,los,icutype,ages,genders,urinedict,paticutypedictindex,resolutions):
    '''
    Given the ICU type generates the los and feature matrices of train and test plus indices associated with test and train
    Input: aggregate feature matrix, ages, genders, dictionary with key as ICU type and value showing indexes showing the patients
    who belong to that ICU type

    Output: test and train indices, length of stays, feature for KNN clustering, input for non overlapping and overlapping cases of HMM modeling

    '''
    numvars = 7 
    inputHmmallVars = [0] * numvars
    ovlapinputHmmallVars = [0] * numvars
    validpatientsindicesallVars = [[] for i in range(numvars)]
    urineflag = False
    if icutype < 5:
        npatients = len(paticutypedictindex[icutype])
        numfeats = np.shape(featmtrx)[1]
        ages = [int(age) for age in ages]
        ages = [ages[paticutypedictindex[icutype][i]] for i in range(len(paticutypedictindex[icutype]))]
        genders = [genders[paticutypedictindex[icutype][i]] for i in range(len(paticutypedictindex[icutype]))]
        # generating input of hmm for each dictionary separately
        for i in range(numvars):
            if listofalldicts[i] == urinedict:
                urineflag= True
            (ovlapinputHmmallVars[i],inputHmmallVars[i],validpatientsindicesallVars[i])  = generateinputoutputbasedondict(npatients, resolutions[i], listofalldicts[i], paticutypedictindex, icutype,urineflag)
        validpatientsindices = validpatientsindicesallVars[0]
        # Intersecting to get the patients who were valid across all dictionaries
        for i in range(1,numvars):
            validpatientsindices = list(set(validpatientsindices).intersection(set(validpatientsindicesallVars[i])))
        realfeatmtrx = np.empty((len(validpatientsindices),numfeats))
        reallos = [0] * len(validpatientsindices)
        KNNfeats = np.column_stack((ages,genders))
        for i in range(len(validpatientsindices)):
            realfeatmtrx[i,:] = featmtrx[paticutypedictindex[icutype][validpatientsindices[i]],:]
            reallos[i] = los[paticutypedictindex[icutype][validpatientsindices[i]]]
    # aggregating age, gender and feature matrices for combination of icu types
    elif icutype > 7:
        numfeats = np.shape(featmtrx)[1]
        if icutype == 8:
            npatients = len(paticutypedictindex[1]) + len(paticutypedictindex[2])
            ages1 = [ages[paticutypedictindex[1][i]] for i in range(len(paticutypedictindex[1]))]
            genders1 = [genders[paticutypedictindex[1][i]] for i in range(len(paticutypedictindex[1]))]
            ages2 = [ages[paticutypedictindex[2][i]] for i in range(len(paticutypedictindex[2]))]
            genders2 = [genders[paticutypedictindex[2][i]] for i in range(len(paticutypedictindex[2]))]
            ages = ages1 + ages2
            genders = genders1 + genders2
            realpaticutypedictindex = paticutypedictindex[1] + paticutypedictindex[2]
        if icutype == 12:
            npatients = len(paticutypedictindex[3]) + len(paticutypedictindex[4])
            ages1 = [ages[paticutypedictindex[3][i]] for i in range(len(paticutypedictindex[3]))]
            genders1 = [genders[paticutypedictindex[3][i]] for i in range(len(paticutypedictindex[3]))]
            ages2 = [ages[paticutypedictindex[4][i]] for i in range(len(paticutypedictindex[4]))]
            genders2 = [genders[paticutypedictindex[4][i]] for i in range(len(paticutypedictindex[4]))]
            ages = ages1 + ages2
            genders = genders1 + genders2
            realpaticutypedictindex = paticutypedictindex[3] + paticutypedictindex[4]
        if icutype == 10:
            npatients = len(paticutypedictindex[2]) + len(paticutypedictindex[3])
            ages1 = [ages[paticutypedictindex[2][i]] for i in range(len(paticutypedictindex[2]))]
            genders1 = [genders[paticutypedictindex[2][i]] for i in range(len(paticutypedictindex[2]))]
            ages2 = [ages[paticutypedictindex[3][i]] for i in range(len(paticutypedictindex[3]))]
            genders2 = [genders[paticutypedictindex[3][i]] for i in range(len(paticutypedictindex[3]))]
            ages = ages1 + ages2
            genders = genders1 + genders2
            realpaticutypedictindex = paticutypedictindex[2] + paticutypedictindex[3]            
        if icutype == 11:
            npatients = len(paticutypedictindex[2]) + len(paticutypedictindex[4])  
            ages1 = [ages[paticutypedictindex[2][i]] for i in range(len(paticutypedictindex[2]))]
            genders1 = [genders[paticutypedictindex[2][i]] for i in range(len(paticutypedictindex[2]))]
            ages2 = [ages[paticutypedictindex[4][i]] for i in range(len(paticutypedictindex[4]))]
            genders2 = [genders[paticutypedictindex[4][i]] for i in range(len(paticutypedictindex[4]))]
            ages = ages1 + ages2
            genders = genders1 + genders2      
            realpaticutypedictindex = paticutypedictindex[2] + paticutypedictindex[4]
        if icutype == 13 :
            npatients = len(paticutypedictindex[1]) + len(paticutypedictindex[2]) + len(paticutypedictindex[3])
            ages1 = [ages[paticutypedictindex[1][i]] for i in range(len(paticutypedictindex[1]))]
            genders1 = [genders[paticutypedictindex[1][i]] for i in range(len(paticutypedictindex[1]))]
            ages2 = [ages[paticutypedictindex[2][i]] for i in range(len(paticutypedictindex[2]))]
            genders2 = [genders[paticutypedictindex[2][i]] for i in range(len(paticutypedictindex[2]))]
            ages3 = [ages[paticutypedictindex[3][i]] for i in range(len(paticutypedictindex[3]))]
            genders3 = [genders[paticutypedictindex[3][i]] for i in range(len(paticutypedictindex[3]))]
            ages = ages1 + ages2 + ages3
            genders = genders1 + genders2 + ages3
            realpaticutypedictindex = paticutypedictindex[1] + paticutypedictindex[2] + paticutypedictindex[3]
        for i in range(numvars):
            if listofalldicts[i] == urinedict:
                urineflag= True
            (ovlapinputHmmallVars[i],inputHmmallVars[i],validpatientsindicesallVars[i])  = generateinputoutputbasedondict(npatients, resolutions[i], listofalldicts[i], paticutypedictindex, icutype,urineflag)
        validpatientsindices = validpatientsindicesallVars[0]
        for i in range(1,numvars):
            validpatientsindices = list(set(validpatientsindices).intersection(set(validpatientsindicesallVars[i])))
        realfeatmtrx = np.empty((len(validpatientsindices),numfeats))
        reallos = [0] * len(validpatientsindices)
        KNNfeats = np.column_stack((ages,genders))
        for i in range(len(validpatientsindices)):
            realfeatmtrx[i,:] = featmtrx[realpaticutypedictindex[validpatientsindices[i]],:]
            reallos[i] = los[realpaticutypedictindex[validpatientsindices[i]]]
    else:
        # all patients together
        npatients = np.shape(featmtrx)[0] 
        numfeats = np.shape(featmtrx)[1]
        ages = [int(age) for age in ages]
        for i in range(numvars):
            if listofalldicts[i] == urinedict:
                urineflag= True
            (ovlapinputHmmallVars[i],inputHmmallVars[i],validpatientsindicesallVars[i])  = generateinputoutputbasedondict(npatients, resolutions[i], listofalldicts[i], paticutypedictindex, icutype,urineflag)
        validpatientsindices = validpatientsindicesallVars[0]
        for i in range(1,numvars):
            validpatientsindices = list(set(validpatientsindices).intersection(set(validpatientsindicesallVars[i])))
        realfeatmtrx = np.empty((len(validpatientsindices),numfeats))
        reallos = [0] * len(validpatientsindices)
        KNNfeats = np.column_stack((ages,genders))
        for i in range(len(validpatientsindices)):
            if icutype < 5:
                realfeatmtrx[i,:] = featmtrx[paticutypedictindex[icutype][validpatientsindices[i]],:]
                reallos[i] = los[paticutypedictindex[icutype][validpatientsindices[i]]]
            else:
                realfeatmtrx[i,:] = featmtrx[[i],:]
                reallos[i] = los[validpatientsindices[i]]
    (testindices,trainindices) = stratifyBOlos(realfeatmtrx,reallos) 
    realfeatmtrxtrain = realfeatmtrx[trainindices,:]
    realfeatmtrxtest = realfeatmtrx[testindices,:]
    return (validpatientsindices,realfeatmtrxtrain,realfeatmtrxtest,KNNfeats,reallos,inputHmmallVars,ovlapinputHmmallVars,trainindices,testindices)
def reportfrequencyofchange(states): 
    '''
    Shows the frequency of change for various features
    '''
    npatients = np.shape(states)[0]
    ntimepoints = np.shape(states)[1]
    changecounts = [0] * npatients
    for i in range(npatients):
        for j in range(1,ntimepoints):
            if states[i,j] != states[i,j-1]:
                changecounts[i] += 1
    avgchangecounts = np.mean(np.array(changecounts))
    measureOfChange = float(avgchangecounts) / float (ntimepoints-1)
    return measureOfChange
def plotforoptimumpoints(everyxhours,nstates,ordAvgVarPatches,ordVarRadiiPatchesmean,ordVarRadiiPatchesmedian,ovlapAvgVarPatches,ovlapVarRadiiPatchesmean,ovlapVarRadiiPatchesmedian,ordscores,ovlapscores):
    '''
    Runs a grid search on time resolution and number of states and plots the result to assist with the best choice of paramerters
    '''
    plt.close()
    plt.plot(nstates,ordscores)
    plt.xlabel("Number of states")
    plt.ylabel("LogLikelihood Value")
    title1 = "nstateslikelihoodord" + str(everyxhours) + str(nstates[-1]) + ".png"
    plt.savefig(title1)
    plt.close()
    plt.plot(nstates,ovlapscores)
    title2 = "nstateslikelihoodoverlap" + str(everyxhours) + str(nstates[-1]) + ".png"
    plt.xlabel("Number of states")
    plt.ylabel("LogLikelihood Value")
    plt.savefig(title2)
    plt.close()
    plt.plot(nstates,ordAvgVarPatches)
    plt.xlabel("Number of states")
    plt.ylabel("Average Variance of LOS within each Patch nonoverlapping")
    title1 = "nstatesAvgVarord" + str(everyxhours) + str(nstates[-1]) + ".png"
    plt.savefig(title1)
    plt.close()
    plt.plot(nstates,ovlapAvgVarPatches)
    title2 = "nstatesAvgVarovlap" + str(everyxhours) + str(nstates[-1]) + ".png"
    plt.xlabel("Number of states")
    plt.ylabel("Average Variance of LOS within each Patch overlapping")
    plt.savefig(title2)
    plt.close()
    plt.plot(nstates,ordVarRadiiPatchesmean)
    plt.xlabel("Number of states")
    plt.ylabel("Variance of Mean of LOS across Patches nonoverlapping")
    title1 = "nstatesVarMeanord" + str(everyxhours) + str(nstates[-1]) + ".png"
    plt.savefig(title1)
    plt.close()
    plt.plot(nstates,ovlapVarRadiiPatchesmean)
    title2 = "nstatesVarMeanovlap" + str(everyxhours) + str(nstates[-1]) + ".png"
    plt.xlabel("Number of states")
    plt.ylabel("Variance of Mean of LOS across Patches overlapping")
    plt.savefig(title2)
    plt.close()    
    plt.plot(nstates,ordVarRadiiPatchesmedian)
    plt.xlabel("Number of states")
    plt.ylabel("Variance of median of LOS across Patches nonoverlapping")
    title1 = "nstatesVarMedianord" + str(everyxhours) + str(nstates[-1]) + ".png"
    plt.savefig(title1)
    plt.close()
    plt.plot(nstates,ovlapVarRadiiPatchesmedian)
    title2 = "nstatesVarMedianovlap" + str(everyxhours) + str(nstates[-1]) + ".png"
    plt.xlabel("Number of states")
    plt.ylabel("Variance of Median of LOS across Patches overlapping")
    plt.savefig(title2)
    plt.close()    
def generateinputoutputbasedondict(npatients,everyxhours,featdict,paticutypedictindex,icutype,urineflag):   
    '''
    Generates feature specific dictionary for a given type
    Input : overall number of patients, time resolution, the particular feature dictionary, patient ICU type dictionary, ICUtype, urine flag
    Output: Overlapping hmm input , non overlapping hmm input and valid patients who have at least one measure for that feature across timepoints
    ''' 
    dimensionality = int(math.floor(48.0/float(everyxhours)))
    overdimensionality = (2 * dimensionality) -1 
    inputhmm=np.zeros((npatients,dimensionality))
    ovlapinputhmm=np.zeros((npatients,overdimensionality))
    counts =  (np.ones((npatients,dimensionality)))
    ovcounts =  (np.ones((npatients,overdimensionality)))
    voidindices = []
    lastovlaptimepoint = 48 - everyxhours
    # aggregating for time points 
    if icutype < 5:
        for j in range(npatients):
            if len(featdict[paticutypedictindex[icutype][j]][0]) != 0 :
                ovlapinputhmmlistval = [[] for i in range(overdimensionality)]
                ovlapinputhmmtime = [[] for i in range(overdimensionality)]
                inputhmmtime = [[]  for i in range(dimensionality)]
                inputhmmlistval = [[] for i in range(dimensionality)]
                for (time,value )in featdict[paticutypedictindex[icutype][j]][0]:
                    indice = int(math.floor((time-1) / (everyxhours*60)))
                    if not (time < (everyxhours) * 60 or time > lastovlaptimepoint * 60 ):
                        ovindice =  2 * (int(math.floor((time + (everyxhours * 60) -1)/ (everyxhours * 60)))) - 1
                        ovlapinputhmmtime[ovindice].append(value)
                        ovlapinputhmmlistval[ovindice].append(value)  
                    inputhmmtime[indice].append(value)
                    inputhmmlistval[indice].append(value)
                    ovlapinputhmmtime[2 * indice].append(value) 
                    ovlapinputhmmlistval[2 * indice].append(value)  
                for i in range(dimensionality):
                    if len(inputhmmtime[i]) > 0:
                        maxind = inputhmmtime[i].index(max(inputhmmtime[i]))
                        inputhmm[j,i] = inputhmmlistval[i][maxind]
                for i in range(overdimensionality):
                    if len(ovlapinputhmmtime[i]) > 0:                    
                        maxind = ovlapinputhmmtime[i].index(max(ovlapinputhmmtime[i]))
                        ovlapinputhmm[j,i] = ovlapinputhmmlistval[i][maxind]
            else:
                voidindices.append(j)
        allzeros1 = list(set(list(np.where(~inputhmm.any(axis = 1))[0])))
        allzeros2 = list(set(list(np.where(~ovlapinputhmm.any(axis = 1))[0])))
        allzeros = []
        allzeros += allzeros1
        allzeros += allzeros2
        validpatientsindices = list(set(range(npatients)) - set(allzeros))
    elif icutype > 7:
        if icutype == 8:
            # type 1 and 2
            for j in range(npatients):
                if j >= len(paticutypedictindex[1]):
                    icupatientidx = paticutypedictindex[2]
                    k= j-len(paticutypedictindex[1])
                else:
                    icupatientidx = paticutypedictindex[1]
                    k= j
                if len(featdict[icupatientidx[k]][0]) != 0 :
                    ovlapinputhmmlistval = [[] for i in range(overdimensionality)]
                    ovlapinputhmmtime = [[] for i in range(overdimensionality)]
                    inputhmmtime = [[]  for i in range(dimensionality)]
                    inputhmmlistval = [[] for i in range(dimensionality)]
                    for (time,value )in featdict[icupatientidx[k]][0]:
                        indice = int(math.floor((time-1) / (everyxhours*60)))
                        if not (time < (everyxhours) * 60 or time > lastovlaptimepoint * 60 ):
                            ovindice =  2 * (int(math.floor((time + (everyxhours * 60) -1)/ (everyxhours * 60)))) - 1
                            ovlapinputhmmtime[ovindice].append(value)
                            ovlapinputhmmlistval[ovindice].append(value)  
                        inputhmmtime[indice].append(value)
                        inputhmmlistval[indice].append(value)
                        ovlapinputhmmtime[2 * indice].append(value) 
                        ovlapinputhmmlistval[2 * indice].append(value)  
                    for i in range(dimensionality):
                        if len(inputhmmtime[i]) > 0:
                            maxind = inputhmmtime[i].index(max(inputhmmtime[i]))
                            inputhmm[j,i] = inputhmmlistval[i][maxind]
                    for i in range(overdimensionality):
                        if len(ovlapinputhmmtime[i]) > 0:                    
                            maxind = ovlapinputhmmtime[i].index(max(ovlapinputhmmtime[i]))
                            ovlapinputhmm[j,i] = ovlapinputhmmlistval[i][maxind]
                else:
                    voidindices.append(j)
            allzeros1 = list(set(list(np.where(~inputhmm.any(axis = 1))[0])))
            allzeros2 = list(set(list(np.where(~ovlapinputhmm.any(axis = 1))[0])))
            allzeros = []
            allzeros += allzeros1
            allzeros += allzeros2
            validpatientsindices = list(set(range(npatients)) - set(allzeros))
        if icutype == 12:
            # type 3,4
            for j in range(npatients):
                if j >= len(paticutypedictindex[3]):
                    icupatientidx = paticutypedictindex[4]
                    k= j-len(paticutypedictindex[3])
                else:
                    icupatientidx = paticutypedictindex[3]
                    k= j
                if len(featdict[icupatientidx[k]][0]) != 0 :
                    ovlapinputhmmlistval = [[] for i in range(overdimensionality)]
                    ovlapinputhmmtime = [[] for i in range(overdimensionality)]
                    inputhmmtime = [[]  for i in range(dimensionality)]
                    inputhmmlistval = [[] for i in range(dimensionality)]
                    for (time,value )in featdict[icupatientidx[k]][0]:
                        indice = int(math.floor((time-1) / (everyxhours*60)))
                        if not (time < (everyxhours) * 60 or time > lastovlaptimepoint * 60 ):
                            ovindice =  2 * (int(math.floor((time + (everyxhours * 60) -1)/ (everyxhours * 60)))) - 1
                            ovlapinputhmmtime[ovindice].append(value)
                            ovlapinputhmmlistval[ovindice].append(value)  
                        inputhmmtime[indice].append(value)
                        inputhmmlistval[indice].append(value)
                        ovlapinputhmmtime[2 * indice].append(value) 
                        ovlapinputhmmlistval[2 * indice].append(value)  
                    for i in range(dimensionality):
                        if len(inputhmmtime[i]) > 0:
                            maxind = inputhmmtime[i].index(max(inputhmmtime[i]))
                            inputhmm[j,i] = inputhmmlistval[i][maxind]
                    for i in range(overdimensionality):
                        if len(ovlapinputhmmtime[i]) > 0:                    
                            maxind = ovlapinputhmmtime[i].index(max(ovlapinputhmmtime[i]))
                            ovlapinputhmm[j,i] = ovlapinputhmmlistval[i][maxind]
                else:
                    voidindices.append(j)
            allzeros1 = list(set(list(np.where(~inputhmm.any(axis = 1))[0])))
            allzeros2 = list(set(list(np.where(~ovlapinputhmm.any(axis = 1))[0])))
            allzeros = []
            allzeros += allzeros1
            allzeros += allzeros2
            validpatientsindices = list(set(range(npatients)) - set(allzeros))
        if icutype == 10:
            # type 2,3
            for j in range(npatients):
                if j >= len(paticutypedictindex[2]):
                    icupatientidx = paticutypedictindex[3]
                    k= j-len(paticutypedictindex[2])
                else:
                    icupatientidx = paticutypedictindex[2]
                    k= j
                if len(featdict[icupatientidx[k]][0]) != 0 :
                    ovlapinputhmmlistval = [[] for i in range(overdimensionality)]
                    ovlapinputhmmtime = [[] for i in range(overdimensionality)]
                    inputhmmtime = [[]  for i in range(dimensionality)]
                    inputhmmlistval = [[] for i in range(dimensionality)]
                    for (time,value )in featdict[icupatientidx[k]][0]:
                        indice = int(math.floor((time-1) / (everyxhours*60)))
                        if not (time < (everyxhours) * 60 or time > lastovlaptimepoint * 60 ):
                            ovindice =  2 * (int(math.floor((time + (everyxhours * 60) -1)/ (everyxhours * 60)))) - 1
                            ovlapinputhmmtime[ovindice].append(value)
                            ovlapinputhmmlistval[ovindice].append(value)  
                        inputhmmtime[indice].append(value)
                        inputhmmlistval[indice].append(value)
                        ovlapinputhmmtime[2 * indice].append(value) 
                        ovlapinputhmmlistval[2 * indice].append(value)  
                    for i in range(dimensionality):
                        if len(inputhmmtime[i]) > 0:
                            maxind = inputhmmtime[i].index(max(inputhmmtime[i]))
                            inputhmm[j,i] = inputhmmlistval[i][maxind]
                    for i in range(overdimensionality):
                        if len(ovlapinputhmmtime[i]) > 0:                    
                            maxind = ovlapinputhmmtime[i].index(max(ovlapinputhmmtime[i]))
                            ovlapinputhmm[j,i] = ovlapinputhmmlistval[i][maxind]
                else:
                    voidindices.append(j)
            if ~(urineflag):
                for i in range(npatients):
                    for j in range(dimensionality):
                        inputhmm[i,j] =  np.array([(float(float(inputhmm[i,j]) / float(counts[i,j])))])
                    for k in range(overdimensionality):
                        ovlapinputhmm[i,k] = np.array([(float(float(ovlapinputhmm[i,k]) / float(ovcounts[i,k])))])
            allzeros1 = list(set(list(np.where(~inputhmm.any(axis = 1))[0])))
            allzeros2 = list(set(list(np.where(~ovlapinputhmm.any(axis = 1))[0])))
            allzeros = []
            allzeros += allzeros1
            allzeros += allzeros2
            validpatientsindices = list(set(range(npatients)) - set(allzeros))
        if icutype == 11:
            # type 2 and 4
            for j in range(npatients):
                if j >= len(paticutypedictindex[2]):
                    icupatientidx = paticutypedictindex[4]
                    k= j-len(paticutypedictindex[2])
                else:
                    icupatientidx = paticutypedictindex[2]
                    k = j
                if len(featdict[icupatientidx[k]][0]) != 0 :
                    ovlapinputhmmlistval = [[] for i in range(overdimensionality)]
                    ovlapinputhmmtime = [[] for i in range(overdimensionality)]
                    inputhmmtime = [[]  for i in range(dimensionality)]
                    inputhmmlistval = [[] for i in range(dimensionality)]
                    for (time,value )in featdict[icupatientidx[k]][0]:
                        indice = int(math.floor((time-1) / (everyxhours*60)))
                        if not (time < (everyxhours) * 60 or time > lastovlaptimepoint * 60 ):
                            ovindice =  2 * (int(math.floor((time + (everyxhours * 60) -1)/ (everyxhours * 60)))) - 1
                            ovlapinputhmmtime[ovindice].append(value)
                            ovlapinputhmmlistval[ovindice].append(value)  
                        inputhmmtime[indice].append(value)
                        inputhmmlistval[indice].append(value)
                        ovlapinputhmmtime[2 * indice].append(value) 
                        ovlapinputhmmlistval[2 * indice].append(value)  
                    for i in range(dimensionality):
                        if len(inputhmmtime[i]) > 0:
                            maxind = inputhmmtime[i].index(max(inputhmmtime[i]))
                            inputhmm[j,i] = inputhmmlistval[i][maxind]
                    for i in range(overdimensionality):
                        if len(ovlapinputhmmtime[i]) > 0:                    
                            maxind = ovlapinputhmmtime[i].index(max(ovlapinputhmmtime[i]))
                            ovlapinputhmm[j,i] = ovlapinputhmmlistval[i][maxind]
                else:
                    voidindices.append(j)
            if ~(urineflag):
                for i in range(npatients):
                    for j in range(dimensionality):
                        inputhmm[i,j] =  np.array([(float(float(inputhmm[i,j]) / float(counts[i,j])))])
                    for k in range(overdimensionality):
                        ovlapinputhmm[i,k] = np.array([(float(float(ovlapinputhmm[i,k]) / float(ovcounts[i,k])))])
            allzeros1 = list(set(list(np.where(~inputhmm.any(axis = 1))[0])))
            allzeros2 = list(set(list(np.where(~ovlapinputhmm.any(axis = 1))[0])))
            allzeros = []
            allzeros += allzeros1
            allzeros += allzeros2
            validpatientsindices = list(set(range(npatients)) - set(allzeros))
        if icutype == 13:
            # type 1 and 2 and 3
            for j in range(npatients):
                if j >= len(paticutypedictindex[2]) + len(paticutypedictindex[1]):
                    icupatientidx = paticutypedictindex[3]
                    k= j- (len(paticutypedictindex[2]) + len(paticutypedictindex[1]))
                elif j >= len(paticutypedictindex[1]) :
                    icupatientidx = paticutypedictindex[2]
                    k= j - len(paticutypedictindex[1])
                else:
                    icupatientidx = paticutypedictindex[1]
                    k= j
                if len(featdict[icupatientidx[k]][0]) != 0 :
                    ovlapinputhmmlistval = [[] for i in range(overdimensionality)]
                    ovlapinputhmmtime = [[] for i in range(overdimensionality)]
                    inputhmmtime = [[]  for i in range(dimensionality)]
                    inputhmmlistval = [[] for i in range(dimensionality)]
                    for (time,value )in featdict[icupatientidx[k]][0]:
                        indice = int(math.floor((time-1) / (everyxhours*60)))
                        if not (time < (everyxhours) * 60 or time > lastovlaptimepoint * 60 ):
                            ovindice =  2 * (int(math.floor((time + (everyxhours * 60) -1)/ (everyxhours * 60)))) - 1
                            ovlapinputhmmtime[ovindice].append(value)
                            ovlapinputhmmlistval[ovindice].append(value)  
                        inputhmmtime[indice].append(value)
                        inputhmmlistval[indice].append(value)
                        ovlapinputhmmtime[2 * indice].append(value) 
                        ovlapinputhmmlistval[2 * indice].append(value)  
                    for i in range(dimensionality):
                        if len(inputhmmtime[i]) > 0:
                            maxind = inputhmmtime[i].index(max(inputhmmtime[i]))
                            inputhmm[j,i] = inputhmmlistval[i][maxind]
                    for i in range(overdimensionality):
                        if len(ovlapinputhmmtime[i]) > 0:                    
                            maxind = ovlapinputhmmtime[i].index(max(ovlapinputhmmtime[i]))
                            ovlapinputhmm[j,i] = ovlapinputhmmlistval[i][maxind]
                else:
                    voidindices.append(j)
            if ~(urineflag):
                for i in range(npatients):
                    for j in range(dimensionality):
                        inputhmm[i,j] =  np.array([(float(float(inputhmm[i,j]) / float(counts[i,j])))])
                    for k in range(overdimensionality):
                        ovlapinputhmm[i,k] = np.array([(float(float(ovlapinputhmm[i,k]) / float(ovcounts[i,k])))])
            allzeros1 = list(set(list(np.where(~inputhmm.any(axis = 1))[0])))
            allzeros2 = list(set(list(np.where(~ovlapinputhmm.any(axis = 1))[0])))
            allzeros = []
            allzeros += allzeros1
            allzeros += allzeros2
            validpatientsindices = list(set(range(npatients)) - set(allzeros))
    else:
        # meaning all patients
        for j in range(npatients-1):
            if j in featdict.keys():
                if len(featdict[j]) != 0 :
                    for (time,value ) in featdict[j][0]:
                        indice = int(math.floor((time-1) / (everyxhours*60)))
                        if not (time < (everyxhours * 60) or time > lastovlaptimepoint * 60 ):
                            ovindice =  2 * (int(math.floor((time + (everyxhours * 60) - 1)/ (everyxhours * 60)))) - 1
                            ovlapinputhmm [j,ovindice] += value
                            ovcounts[j,ovindice] += 1
                        ovlapinputhmm[j,2 * indice] += value                
                        inputhmm[j,indice] += value
                        counts[j,indice] +=1
                        ovcounts[j,2 * indice] +=1
                else:
                    voidindices.append(j)
        if ~(urineflag):
            for i in range(npatients-1):
                for j in range(dimensionality):
                    inputhmm[i,j] =  np.array([(float(float(inputhmm[i,j]) / float(counts[i,j])))])
                for k in range(overdimensionality):
                    ovlapinputhmm[i,k] = np.array([(float(float(ovlapinputhmm[i,k]) / float(ovcounts[i,k])))])
        allzeros = list(set(list(np.where(~inputhmm.any(axis = 1))[0])))
        validpatientsindices = list(set(range(npatients)) - set(allzeros))
    return (ovlapinputhmm,inputhmm,validpatientsindices)
def find2mostcertaintimepointsidx(indiv_chosenstate_probs,ovlapindiv_chosenstate_probs):
    '''
    Finds the probabilites belonging to the most certain time points and uses them as features, this did not work :D
    '''
    numpatients = np.shape(indiv_chosenstate_probs)[0]
    dimensionality = np.shape(indiv_chosenstate_probs)[1]
    overdimensionality = np.shape(ovlapindiv_chosenstate_probs)[1]    
    ordfirstmaxstateidx = [0] * (numpatients)
    ordsecondmaxstateidx = [0] * (numpatients)
    ovlapfirstmaxstateidx = [0] * (numpatients)
    ovlapsecondmaxstateidx = [0] * (numpatients)
    for i in range(numpatients):
        indlist = list(indiv_chosenstate_probs[i,:])
        ovlaplist = list(ovlapindiv_chosenstate_probs[i,:])
        ordfirstmaxstateidx[i] = indlist.index(max(indlist))
        ovlapfirstmaxstateidx[i] = ovlaplist.index(max(ovlaplist))
        ordexceptmax = max([indlist[j]  for j in range(dimensionality) if j !=ordfirstmaxstateidx[i]])
        ovlapexceptmax = max([ovlaplist[j]  for j in range(overdimensionality) if j !=ovlapfirstmaxstateidx[i]])        
        ordsecondmaxstateidx[i] = indlist.index(ordexceptmax)
        ovlapsecondmaxstateidx[i] = ovlaplist.index(ovlapexceptmax)
    return (ordfirstmaxstateidx,ordsecondmaxstateidx,ovlapfirstmaxstateidx,ovlapsecondmaxstateidx)
def applybestPCA(featmtrx):
    maxim = np.shape(featmtrx)[1]
    ncomponents = [2,maxim]
    i = 0
    explained = 0
    while(i <len(ncomponents) and explained < 0.95):
        pca = PCA(n_components=ncomponents[i])
        pca.fit(featmtrx)
        pcafeatmtrx = pca.fit_transform(featmtrx)
        explained = (sum(pca.explained_variance_ratio_))
        i += 1
    plt.close()
    ax = sns.heatmap(pca.components_)
    plt.xlabel("selected component")
    plt.ylabel("original feats")
    realtitle = "PCAheatmap" + ".png"
    (plt.savefig(realtitle))

    return ncomponents[i-1]      
def Linearregr(trainingx,testx,trainingy,testy,hmm,log,numofstate,everyxhours,icutype,model):
    '''
    runs the best linear regression model lasso on the probability features, or baselien feature,  to predict length of stay

    Input: probability train and test features, or baseline features and other model parameters
    Output: MSE error and predicted LOS
    '''
    if hmm:
        hmmtitile = "hmm"
    else:
        hmmtitile = ""
    if log:
        logtrainingy = logaritmizelos(trainingy)
        logtesty = logaritmizelos(testy)
    else :
        logtrainingy = trainingy
        logtesty = testy
    # scaling both training and test using the same scaler
    scaler = sklearn.preprocessing.StandardScaler().fit(trainingx)
    trainingx = scaler.transform(trainingx)
    testx = scaler.transform(testx)
    featmtrx = np.concatenate((trainingx,testx),axis = 0)
    totsamp = np.shape(featmtrx)[0]
    numtrain = np.shape(trainingx)[0]
    # applying pca based on the flag, only if the mode is not 
    if hmm:
        pcafeatmtrx = featmtrx
    else:
        ncomp = applybestPCA(featmtrx)
        pca = PCA(n_components=ncomp)
        pca.fit(featmtrx)
        pcafeatmtrx = pca.fit_transform(featmtrx)

    sapsdetect = len(np.shape(pcafeatmtrx))
    if sapsdetect != 1:
        trainingfeats = pcafeatmtrx[0:numtrain,:]
        testfeats = pcafeatmtrx[numtrain:,:]
    else:
        trainingfeats = pcafeatmtrx[0:numtrain]
        testfeats = pcafeatmtrx[numtrain:]   
        trainingfeats = trainingfeats.reshape((numtrain,1))
        testfeats = testfeats.reshape((totsamp-numtrain,1))
    # running cross validation for linear regresssion 
    bestalpha = lrcrossvalidation(trainingfeats,logtrainingy,'Lasso')
    testmodely = doLRandreport(trainingfeats, logtrainingy, testfeats, logtesty,bestalpha,'Lasso')
    plottitle = hmmtitile + "Lasso" + "numofstate=" + str(numofstate) + "resolution" + str(everyxhours) + "icutype" + str(icutype) + model
    plotscatter(logtesty, testmodely,plottitle,1)
    errors = (testmodely - logtesty) ** 2
    # shows the actual histogram 
    # histitle = "hist" + plottitle
    testmse =  np.sqrt(((float( np.sum( (testmodely - logtesty) ** 2 ))) / float(len(testmodely))))
    mseerror = testmse
    # plt.close()
    # plt.hist(errors, bins=10 )
    # filename = histitle + ".png"
    # plt.savefig(filename)
    # plt.close()
    return mseerror  
def startendlosvisualizerweighted(indivtrainstates,ovlapindivtrainstates,changestatindices,ovlapchangestatindices,nstates,ytrain,everyxhours,mean_chosenstate_prob_ord,mean_chosenstate_prob_ovlap,icutype):
    dimensionality = np.shape(indivtrainstates)[1]
    overdimensionality = np.shape(ovlapindivtrainstates)[1]
    ordfirststates = indivtrainstates[:,0]
    ovlapfirststates = ovlapindivtrainstates[:,0]
    ordlaststates = indivtrainstates[:,dimensionality-1]
    ovlaplaststates = ovlapindivtrainstates[:,overdimensionality-1]
    allpossiblepair = [(i,j) for i in range(nstates) for j in range(nstates) ]
    loschg =dict.fromkeys(allpossiblepair)
    losnonchg = dict.fromkeys(allpossiblepair)
    ordlos = dict.fromkeys(allpossiblepair)
    ordlosidx = dict.fromkeys(allpossiblepair)
    ovlaplosidx = dict.fromkeys(allpossiblepair)    
    ovlaploschg = dict.fromkeys(allpossiblepair)
    ovlaplosnonchg = dict.fromkeys(allpossiblepair)
    ovlaplos = dict.fromkeys(allpossiblepair)
    listak = dict.fromkeys(allpossiblepair,0)
    probchg =dict.fromkeys(allpossiblepair,0)
    probnonchg = dict.fromkeys(allpossiblepair,0)
    ordprob = dict.fromkeys(allpossiblepair,0)  
    ovlapprobchg = dict.fromkeys(allpossiblepair,0)
    ovlapprobnonchg = dict.fromkeys(allpossiblepair,0)
    ovlapprob = dict.fromkeys(allpossiblepair,0)
    for key in allpossiblepair:
        loschg[key] = list()
        losnonchg[key] = list()
        ordlos[key]= list()
        ordlosidx[key] = list()
        ovlaplosidx[key] = list()
        ovlaploschg[key] = list()
        ovlaplosnonchg[key]= list()
        ovlaplos [key]= list()
    for i in range(np.shape(indivtrainstates)[0]):
        numbord = float((ytrain[i])) * float(mean_chosenstate_prob_ord[i])
        numbovlap = float(ytrain[i]) *float( mean_chosenstate_prob_ovlap[i])
        keyak = tuple((int(ordfirststates[i]),int(ordlaststates[i])))
        keyakovlap = tuple((int(ovlapfirststates[i]),int(ovlaplaststates[i])))
        ordprob[keyak] += float(mean_chosenstate_prob_ord[i])        
        (ordlos[keyak]).append(numbord)
        (ordlosidx[keyak]).append(i)
        (ovlaplosidx[keyak]).append(i)
        listak[keyak] += 1
        (ovlaplos[keyakovlap]).append(numbovlap)
        ovlapprob [keyakovlap] += float( mean_chosenstate_prob_ovlap[i])
        if i in changestatindices:
            (loschg[keyak]).append(numbord)
            probchg [keyak]+= float(mean_chosenstate_prob_ord[i])
        else :
            (losnonchg[keyak]).append(numbord)
            probnonchg [keyak]+= float(mean_chosenstate_prob_ord[i])            
        if i in ovlapchangestatindices:
            (ovlaploschg[keyakovlap]).append(numbovlap)
            ovlapprobchg[keyakovlap] += float(mean_chosenstate_prob_ovlap[i])
        else:
            (ovlaplosnonchg[keyakovlap]).append(numbovlap)
            ovlapprobnonchg[keyakovlap] += float(mean_chosenstate_prob_ovlap[i])            
    radiiordchg = dict.fromkeys(allpossiblepair,0)
    radiiordnonchg = dict.fromkeys(allpossiblepair,0)
    radiiovlapchg = dict.fromkeys(allpossiblepair,0)
    radiiovlapnonchg = dict.fromkeys(allpossiblepair,0)
    radiiord = dict.fromkeys(allpossiblepair,0)
    radiioverlap = dict.fromkeys(allpossiblepair,0)
    radiioverlapvariance = dict.fromkeys(allpossiblepair,0)
    radiiordvariance = dict.fromkeys(allpossiblepair,0)
    radiiordmedian = dict.fromkeys(allpossiblepair,0)
    radiiovlapmedian = dict.fromkeys(allpossiblepair,0)
    for i in range(nstates):
        for j in range(nstates):
            if len(ordlos[(i,j)]) >= 1:                                 
                radiiord[(i,j)] = float(float(np.sum(ordlos[(i,j)])) / float(ordprob[i,j]))
                radiiordvariance[(i,j)] = np.var(ordlos[(i,j)])
                radiiordmedian[(i,j)] = np.median(ordlos[(i,j)])
            if len(loschg[(i,j)]) >= 1:                    
                radiiordchg[(i,j)] = float(float(np.sum(loschg[(i,j)])) / float(probchg[i,j]))
            if len(losnonchg[(i,j)]) >= 1:                    
                radiiordnonchg[(i,j)] = float(float(np.sum(losnonchg[(i,j)])) / float(probnonchg[i,j]))
            if len(ovlaplos[(i,j)]) >= 1:            
                radiioverlap[(i,j)] = float(float(np.sum(ovlaplos[(i,j)])) / float(ovlapprob[i,j]))
                radiioverlapvariance[(i,j)] = np.var(ovlaplos[(i,j)])
                radiiovlapmedian[(i,j)] = np.median(ovlaplos[(i,j)])
            if len(ovlaploschg[(i,j)]) >= 1:
                radiiovlapchg[(i,j)] = float(float(np.sum(ovlaploschg[(i,j)])) / float(ovlapprobchg[i,j]))
            if len(ovlaplosnonchg[(i,j)]) >= 1:            
                radiiovlapnonchg[(i,j)] = float(float(np.sum(ovlaplosnonchg[(i,j)])) / float(ovlapprobnonchg[i,j]))
    ordAvgVarPatches = np.mean(radiiordvariance.values())
    ordVarRadiiPatchesmean = np.var(radiiord.values())
    ordVarRadiiPatchesmedian = np.var(radiiordmedian.values())
    ovlapAvgVarPatches = np.mean(radiioverlapvariance.values())
    ovlapVarRadiiPatchesmean = np.var(radiioverlap.values())
    ovlapVarRadiiPatchesmedian = np.var(radiiovlapmedian.values())
    biggestloskeyord = keywithmaxval(radiiord)
    biggestloskeyovlap = keywithmaxval(radiioverlap)
    idxbiggestlosord = ordlosidx[biggestloskeyord]
    idxbiggestlosovlap = ovlaplosidx[biggestloskeyovlap]
    params = csv.writer(open("params.csv", "w")) 
    params.writerow([everyxhours,nstates,icutype])
    dicttocsv(radiiord,ordlos,"ordweighted"+ str(icutype))
    dicttocsv(radiiordchg,loschg,"ordchgweighted"+ str(icutype))
    dicttocsv(radiiordnonchg,losnonchg,"ordnonchgweighted"+ str(icutype))
    dicttocsv(radiioverlap,ovlaplos,"overlapweighted"+ str(icutype))
    dicttocsv(radiiovlapchg,ovlaploschg,"overlapchgweighted"+ str(icutype))
    dicttocsv(radiiovlapnonchg,ovlaplosnonchg,"overlapnonchgweighted"+ str(icutype))
    return (idxbiggestlosord,idxbiggestlosovlap,ordAvgVarPatches,ordVarRadiiPatchesmean,ordVarRadiiPatchesmedian,ovlapAvgVarPatches,ovlapVarRadiiPatchesmean,ovlapVarRadiiPatchesmedian)
def plotlongestlosbasedoncertainty(idxbiggestlosord,idxbiggestlosovlap,ytrain,indiv_chosenstate_probs,ovlapindiv_chosenstate_probs,everyxhours,nstates,indivprobs,ovlapindivprobs,weighted,icutype):
    '''
    plots long staying patient for the purpose of comparing models with different number of states and time resoltions
    '''
    
    if weighted:
        weight = "weighted"
    else:
        weight = ""
    losord = [ytrain[i] for i in idxbiggestlosord]
    losovlap = [ytrain[i] for i in idxbiggestlosovlap]
    dimensionality = np.shape(indiv_chosenstate_probs)[1]
    overdimensionality = np.shape(ovlapindiv_chosenstate_probs)[1] 
    realcertaintyord = np.empty((len(idxbiggestlosord),dimensionality,nstates))
    realcertaintyovlap = np.empty((len(idxbiggestlosovlap),overdimensionality,nstates))
    realcertaintyord = [indivprobs[i,:,:] for i in idxbiggestlosord]
    certaintyord = [indiv_chosenstate_probs[i,:] for i in idxbiggestlosord]
    certaintyovlap = [ovlapindiv_chosenstate_probs[i,:] for i in idxbiggestlosovlap]
    realcertaintyovlap = [ovlapindivprobs[i,:,:] for i in idxbiggestlosovlap]
    mean_certainty_ord = [np.mean(certaintyord[i]) for i in range(len(idxbiggestlosord))]
    mean_certainty_ovlap = [np.mean(certaintyovlap[i]) for i in range(len(idxbiggestlosovlap))]
    plt.close()
    plt.scatter(losord,mean_certainty_ord)
    plt.ylabel('Mean certainty over different timpoints for that patient')
    plt.xlabel('Length of stay of that patient')
    plt.title(weight +'Mean certainty of longest staying patient over timepoint w.r.t Length of stay nonoverlapping')   
    realtitle = weight + "ordlongestloscertaintyres" + str(everyxhours)  + str("numstates") + str(nstates) +"icutype" + str(icutype) +  ".png"
    plt.savefig((realtitle))
    plt.close()
    plt.scatter(losovlap,mean_certainty_ovlap)
    plt.ylabel('Mean certainty over different timpoints for that patient')
    plt.xlabel('Length of stay of that patient')
    plt.title(weight +'Mean certainty of longest staying patient over timepoint w.r.t Length of stay overlapping')   
    realtitle = weight + "ovlaplongestloscertaintyres" + str(everyxhours)  + str("numstates") + str(nstates) +"icutype" + str(icutype) +  ".png"
    plt.savefig((realtitle))
    plt.close()
    losord = [int(i) for i in losord]
    losovlap = [int(i) for i in losovlap]   
    if len(losord) >= 1 :
        ax1 =sns.violinplot(data = (certaintyord),cut = 0)
        ax1.set_xticklabels(losord)
        plt.xlabel("Length of stay of patients")
        plt.ylabel("Certainty for the chosen states across all timepoints")
        plt.title(weight +"Distributin of certainty of chosen states for longest staying patients ordinary")
        realtitle = weight + "Distrcertaintychosstateslongeststayordnumstate" + str(nstates) + "res =" + str(everyxhours) + "icutype" + str(icutype) +  ".png"
        (plt.savefig(realtitle))
        plt.clf()
    if len(losovlap) >= 1:
        ax2 = sns.violinplot(data = (certaintyovlap),cut = 0)
        ax2.set_xticklabels(losovlap)
        plt.xlabel("Length of stay of patients")
        plt.ylabel("Certainty for the chosen states across all timepoints")
        plt.title(weight +"Distributin of certainty of chosen states for longest staying patients overlapping")
        realtitle = weight + "Distrcertaintychosstateslongeststayovlapnumstate" + str(nstates) + "res =" + str(everyxhours) + "icutype" + str(icutype) + ".png"
        (plt.savefig(realtitle))
        plt.clf()
def keywithmaxval(di):
    k= list( di.keys())
    v = list(di.values())
    result  = k[v.index(max(v))]
    return result
def detectchangeinstates(indivtrainstates,ovlapindivtrainstates):
    '''
    how many patient do experience some form of state changing during their transition in time
    '''
    numchangestate = 0
    ovlapnumchangestate = 0
    changestatindices = []
    ovlapchangestatindices = []
    dimensionality = np.shape(indivtrainstates)[1]
    overdimensionality = np.shape(ovlapindivtrainstates)[1]
    numvalidpatients = np.shape(indivtrainstates)[0]
    for i in range(numvalidpatients):
        lennorm = len(set(indivtrainstates[i,0:dimensionality-1]))
        lenoverlap = len(set(ovlapindivtrainstates[i,0:overdimensionality-1]))
        if lennorm != 1:
            numchangestate +=1
            changestatindices.append(i)
        if lenoverlap != 1 :
            ovlapnumchangestate +=1
            ovlapchangestatindices.append(i)
    return (changestatindices,ovlapchangestatindices,numchangestate,ovlapnumchangestate)
def detectchangeinstatesfiltered(indivtrainstates,ovlapindivtrainstates):
    numchangestate = 0
    ovlapnumchangestate = 0
    changestatindices = []
    ovlapchangestatindices = []
    dimensionality = np.shape(indivtrainstates)[1]
    overdimensionality = np.shape(ovlapindivtrainstates)[1]
    for i in range(np.shape(indivtrainstates)[0]):
        lennorm = len(set(indivtrainstates[i,0:dimensionality-1]))
        if lennorm != 1:
            numchangestate +=1
            changestatindices.append(i)
    for i in range(np.shape(ovlapindivtrainstates)[0]):
        lenoverlap = len(set(ovlapindivtrainstates[i,0:overdimensionality-1]))
        if lenoverlap != 1 :
            ovlapnumchangestate +=1
            ovlapchangestatindices.append(i)
    return (changestatindices,ovlapchangestatindices,numchangestate,ovlapnumchangestate)
def filterbasedoncertainty(indivprobs,ovlapindivprobs,onlyfirstandlast):
    '''
    teasing apart patients for whom the first and last states probabilites is greater than 0.5
    '''
    npatients = np.shape(indivprobs)[0]
    ntimepoints = np.shape(indivprobs)[1]
    novlaptimepoints = np.shape(ovlapindivprobs)[1]                
    nstates  = np.shape(indivprobs)[2]
    orduncertainidx = []
    ovlapuncertainidx = []
    if ~onlyfirstandlast :
        for i in range(npatients):
            for j in range(ntimepoints):
                mostlklsateprob = max(indivprobs[i,j,:])
                if mostlklsateprob < 0.5 :
                    orduncertainidx.append(i)
                    break
            for k in range(novlaptimepoints):
                mostlklstateprob2 = max(ovlapindivprobs[i,k,:])  
                if mostlklstateprob2 < 0.5 :
                    ovlapuncertainidx.append(i)
                    break
    if onlyfirstandlast:
        for i in range(npatients):
            mostlklsateprobf = max(indivprobs[i,0,:])
            mostlklsateprobl = max(indivprobs[i,-1,:])        
            if (mostlklsateprobf < 0.5 ) or (mostlklsateprobl < 0.5 ) :
                orduncertainidx.append(i)
            mostlklstateprob2f = max(ovlapindivprobs[i,0,:])  
            mostlklstateprob2l = max(ovlapindivprobs[i,-1,:])  
            if (mostlklstateprob2f < 0.5) or  (mostlklstateprob2l < 0.5) :
                ovlapuncertainidx.append(i)
    return (orduncertainidx,ovlapuncertainidx)  
def dicttocsv(dictionnaire,dict2, title):
    '''
    Writes a dictionary to a csv file to be later used in the R script
    '''
    realtitle = title + ".csv"
    w = csv.writer(open(realtitle, "w"))
    for key, val in dictionnaire.items():
        if (~ np.isnan(val)) and val >=1:
            w.writerow([key[0],key[1] ,val,len(dict2[key]),float(len(dict2[key]) * val)])
def startendlosvisualizer(indivtrainstates,ovlapindivtrainstates,changestatindices,ovlapchangestatindices,nstates,ytrain,everyxhours,icutype):
    '''
    computes and plots the average LOS and count of patients falling into various start end state pairs
    '''
    # initializations
    dimensionality = np.shape(indivtrainstates)[1]
    overdimensionality = np.shape(ovlapindivtrainstates)[1]
    ordfirststates = indivtrainstates[:,0]
    ovlapfirststates = ovlapindivtrainstates[:,0]
    ordlaststates = indivtrainstates[:,dimensionality-1]
    ovlaplaststates = ovlapindivtrainstates[:,overdimensionality-1]
    allpossiblepair = [(i,j) for i in range(nstates) for j in range(nstates) ]
    loschg =dict.fromkeys(allpossiblepair)
    losnonchg = dict.fromkeys(allpossiblepair)
    ordlos = dict.fromkeys(allpossiblepair)
    ordlosidx = dict.fromkeys(allpossiblepair)
    ovlaplosidx = dict.fromkeys(allpossiblepair)    
    ovlaploschg = dict.fromkeys(allpossiblepair)
    ovlaplosnonchg = dict.fromkeys(allpossiblepair)
    ovlaplos = dict.fromkeys(allpossiblepair)
    listak = dict.fromkeys(allpossiblepair,0)
    for key in allpossiblepair:
        loschg[key] = list()
        losnonchg[key] = list()
        ordlos[key]= list()
        ordlosidx[key] = list()
        ovlaplosidx[key] = list()
        ovlaploschg[key] = list()
        ovlaplosnonchg[key]= list()
        ovlaplos [key]= list()
    # getting the statistics for the necessary measures
    for i in range( np.shape(indivtrainstates)[0]):
        numb = int(ytrain[i])
        keyak = tuple((int(ordfirststates[i]),int(ordlaststates[i])))
        keyakovlap = tuple((int(ovlapfirststates[i]),int(ovlaplaststates[i])))
        (ordlos[keyak]).append(numb)
        (ordlosidx[keyak]).append(i)
        (ovlaplosidx[keyak]).append(i)
        listak[keyak] += 1
        (ovlaplos[keyakovlap]).append(numb)
        if i in changestatindices:
            (loschg[keyak]).append(numb)
        else :
            (losnonchg[keyak]).append(numb)
        if i in ovlapchangestatindices:
            (ovlaploschg[keyakovlap]).append(numb)
        else:
            (ovlaplosnonchg[keyakovlap]).append(numb)
    radiiordchg = dict.fromkeys(allpossiblepair)
    radiiordnonchg = dict.fromkeys(allpossiblepair)
    radiiovlapchg = dict.fromkeys(allpossiblepair)
    radiiovlapnonchg = dict.fromkeys(allpossiblepair)
    radiiord = dict.fromkeys(allpossiblepair)
    radiioverlap = dict.fromkeys(allpossiblepair)
    # setting radious of circles to denote LOS
    for i in range(nstates):
        for j in range(nstates):
            radiiord[(i,j)] = np.mean(ordlos[(i,j)])
            radiiordchg[(i,j)] = np.mean(loschg[(i,j)])
            radiiordnonchg[(i,j)] = np.mean(losnonchg[(i,j)])
            radiioverlap[(i,j)] = np.mean(ovlaplos[(i,j)])
            radiiovlapchg[(i,j)] = np.mean(ovlaploschg[(i,j)])
            radiiovlapnonchg[(i,j)] = np.mean(ovlaplosnonchg[(i,j)])
    biggestloskeyord = keywithmaxval(radiiord)
    biggestloskeyovlap = keywithmaxval(radiioverlap)
    idxbiggestlosord = ordlosidx[biggestloskeyord]
    idxbiggestlosovlap = ovlaplosidx[biggestloskeyovlap]
    # saving the statistics for each pair to later generate plots using plt.
    params = csv.writer(open("params.csv", "w")) 
    params.writerow([everyxhours,nstates,icutype])
    dicttocsv(radiiord,ordlos,"ord" + str(icutype))
    dicttocsv(radiiordchg,loschg,"ordchg"+ str(icutype))
    dicttocsv(radiiordnonchg,losnonchg,"ordnonchg"+ str(icutype))
    dicttocsv(radiioverlap,ovlaplos,"overlap"+ str(icutype))
    dicttocsv(radiiovlapchg,ovlaploschg,"overlapchg"+ str(icutype))
    dicttocsv(radiiovlapnonchg,ovlaplosnonchg,"overlapnonchg"+ str(icutype))
    return (idxbiggestlosord,idxbiggestlosovlap,ovlaplosidx)
def startendlosvisualizerfiltered(indivtrainstates,ovlapindivtrainstates,certainordidx,certaiovlapdidx,changestatindices,ovlapchangestatindices,nstates,ytrain,everyxhours,icutype):
    dimensionality = np.shape(indivtrainstates)[1]
    overdimensionality = np.shape(ovlapindivtrainstates)[1]
    ordfirststates = indivtrainstates[:,0]
    ovlapfirststates = ovlapindivtrainstates[:,0]
    ordlaststates = indivtrainstates[:,dimensionality-1]
    ovlaplaststates = ovlapindivtrainstates[:,overdimensionality-1]
    allpossiblepair = [(i,j) for i in range(nstates) for j in range(nstates) ]
    loschg =dict.fromkeys(allpossiblepair)
    losnonchg = dict.fromkeys(allpossiblepair)
    ordlos = dict.fromkeys(allpossiblepair)
    ovlaploschg = dict.fromkeys(allpossiblepair)
    ovlaplosnonchg = dict.fromkeys(allpossiblepair)
    ovlaplos = dict.fromkeys(allpossiblepair)
    listak = dict.fromkeys(allpossiblepair,0)
    for key in allpossiblepair:
        loschg[key] = list()
        losnonchg[key] = list()
        ordlos[key]= list()
        ovlaploschg[key] = list()
        ovlaplosnonchg[key]= list()
        ovlaplos [key]= list()
    ytrainordfiltered = [ytrain[i] for i in certainordidx]
    ytrainovlapfiltered = [ ytrain[i] for i in certaiovlapdidx]
    for i in range( np.shape(indivtrainstates)[0]):
        numb = int(ytrainordfiltered[i])
        keyak = tuple((int(ordfirststates[i]),int(ordlaststates[i])))
        (ordlos[keyak]).append(numb)
        listak[keyak] += 1
        if i in changestatindices:
            (loschg[keyak]).append(numb)
        else :
            (losnonchg[keyak]).append(numb)
    for i in range( np.shape(ovlapindivtrainstates)[0]):
        numb = int(ytrainovlapfiltered[i])
        keyakovlap = tuple((int(ovlapfirststates[i]),int(ovlaplaststates[i])))
        (ovlaplos[keyakovlap]).append(numb)
        if i in ovlapchangestatindices:
            (ovlaploschg[keyakovlap]).append(numb)
        else:
            (ovlaplosnonchg[keyakovlap]).append(numb)
    radiiordchg = dict.fromkeys(allpossiblepair)
    radiiordnonchg = dict.fromkeys(allpossiblepair)
    radiiovlapchg = dict.fromkeys(allpossiblepair)
    radiiovlapnonchg = dict.fromkeys(allpossiblepair)
    radiiord = dict.fromkeys(allpossiblepair)
    radiioverlap = dict.fromkeys(allpossiblepair)
    for i in range(nstates):
        for j in range(nstates):
            radiiord[(i,j)] = np.mean(ordlos[(i,j)])
            radiiordchg[(i,j)] = np.mean(loschg[(i,j)])
            radiiordnonchg[(i,j)] = np.mean(losnonchg[(i,j)])
            radiioverlap[(i,j)] = np.mean(ovlaplos[(i,j)])
            radiiovlapchg[(i,j)] = np.mean(ovlaploschg[(i,j)])
            radiiovlapnonchg[(i,j)] = np.mean(ovlaplosnonchg[(i,j)])
    params = csv.writer(open("params.csv", "w")) 
    params.writerow([everyxhours,nstates,icutype])
    dicttocsv(radiiord,ordlos,"ordfiltered" + str(icutype))
    dicttocsv(radiiordchg,loschg,"ordchgfiltered"+ str(icutype))
    dicttocsv(radiiordnonchg,losnonchg,"ordnonchgfiltered"+ str(icutype))
    dicttocsv(radiioverlap,ovlaplos,"overlapfiltered"+ str(icutype))
    dicttocsv(radiiovlapchg,ovlaploschg,"overlapchgfiltered"+ str(icutype))
    dicttocsv(radiiovlapnonchg,ovlaplosnonchg,"overlapnonchgfiltered"+ str(icutype))
def drawcircircllos(radii,title,everyxhours,nstates,icutype):
    '''
    Initial circle plot using the python packages, later R was used to draw the circle plots
    '''
    plt.close()
    numb = len(radii.keys())
    realradii = []
    keys = []
    circle = [0] * numb
    for key in radii.keys():
        if (~ np.isnan(radii[key])):
            keys.append(key)
            realradii.append(radii[key])
        else:
            numb -= 1
    fig, ax = plt.subplots()
    sizes =[ float(realradii[i]/ float(max(realradii) * nstates * 0.5 ))  for i in range(numb)] 
    xy = [(keys[i][0] + 1,keys[i][1] + 1 ) for i in range(numb)]
    patches = [plt.Circle(center, size) for center, size in zip(xy, sizes)]
    coll = matplotlib.collections.PatchCollection(patches, facecolors='black')
    ax.add_collection(coll)
    plt.ylim(ymin = 0, ymax = nstates + 1)
    plt.xlim(xmin = 0, xmax = nstates + 1 )
    plt.xlabel("Start State")
    plt.ylabel("End State")  
    realtitle = title + "res" + str(everyxhours) + "numstate=" + str(nstates) + "icutype=" + str(icutype) + ".png"
    plt.title(realtitle)      
    fig.savefig(realtitle)
def imputefeatures(trainindices,testindices,inputhmm,KNNtrain,KNNtest,dimensionality,numvars):
    '''
    Impute missing time points using a carry forward methodif the first one is missing, we first impute 
    it using measures based on KNN constructed on age and gender of first time point available measures

    Input: input hmm, train and test indices, numvars is number of features included

    Output: Imputed overlapping and non-overlapping input for HMM
    '''
    inputhmmtrain = np.empty((len(trainindices),dimensionality,numvars))
    inputhmmtest  = np.empty((len(testindices),dimensionality,numvars))
    for i in range(len(trainindices)):
        inputhmmtrain[i,:,:] = inputhmm[trainindices[i]][:]
    for i in range(len(testindices)):
        inputhmmtest[i,:,:] = inputhmm[testindices[i]][:]
    for i in range(numvars):
        missingfirsttrain =  (np.where( 0 in inputhmmtrain[:,0,i] == 0)[0].tolist())
        missingfirsttest = (np.where( inputhmmtest[:,0,i] == 0)[0].tolist())
        nonmistrain = list(set(range(len(trainindices))) - set(missingfirsttrain))
        nonmistest = list(set(range(len(testindices))) - set(missingfirsttest))
        if len(missingfirsttrain) > 0:
            neightrain = KNeighborsRegressor(n_neighbors=5)
            neightrain.fit(KNNtrain[nonmistrain,:],inputhmmtrain[nonmistrain,0,i])
            inputhmmtrain[missingfirsttrain,0,i] = neightrain.predict(KNNtrain[missingfirsttrain,:])
        if len(missingfirsttest) > 0:
            neightest = KNeighborsRegressor(n_neighbors=5)
            neightest.fit(KNNtest[nonmistest,:],inputhmmtest[nonmistest,0,i])
            inputhmmtest[missingfirsttest,0,i] = neightest.predict(KNNtest[missingfirsttest,:])
    # push forward the rest of the missing time points
    inputhmmtrain = pushforward(range(np.shape(inputhmmtrain)[0]),dimensionality,inputhmmtrain)
    inputhmmtest = pushforward(range(np.shape(inputhmmtest)[0]),dimensionality,inputhmmtest)
    return (inputhmmtrain,inputhmmtest)
def pushforward(indexes,dimensionality,inputhmm):
    '''
    Pushes forward to impute missing time point measures
    '''
    numvars = np.shape(inputhmm)[2]
    for idx in range(len(indexes)):
        if  float(0) not in inputhmm[idx,0,:]:
            for i in range(1,dimensionality):
                for j in range(numvars):
                    if inputhmm[idx,i,j] == float(0):
                        inputhmm[idx,i,j] = inputhmm[idx,i-1,j]
    return inputhmm
def fxn():
    '''
    Get's rid of annying deprecation warning because of hmm modue implementation
    '''
    warnings.warn("deprecated", DeprecationWarning)
def hmmgrid(nstate,inputtrain,trainlengths,meanfeat,varfeat,maxfeat):
    meanfeat = float(meanfeat)
    mincovars = list(np.linspace(0.01,0.4,10))
    covartypes = ["full"]
    algorithms = ["viterbi","map"]
    means_priors = [[[(meanfeat - 20.0)]] * nstate ,[[(meanfeat - 10.0)] ]* nstate,[[meanfeat]] * nstate , [[meanfeat + 10.0]] * nstate, [[meanfeat + 20]] * nstate]
    transmat_priors = [np.full((nstate,nstate),float ( 1.0 / float (nstate))), np.diagflat(1.0 * nstate)]
    covars_priors = [float( 1.0 / nstate) * np.tile(1, (nstate, 1)),float( float(varfeat) / float(maxfeat)) * np.tile(1, (nstate, 1))]
    prevscore = float("-inf")
    selmincovar = mincovars[0]
    selcovartype = covartypes[0]
    selalg = algorithms[0]
    selmeanp= means_priors[0]
    seltransprior = transmat_priors[0]
    selcovarp = covars_priors[0]
    for mincovar in mincovars:
        for alg in algorithms:
            for i in range(len(means_priors)):
                for transprior in transmat_priors :
                    for covar in covars_priors:
                            hmmmodel = hmm.GaussianHMM(algorithm=alg,n_components= int(nstate), covariance_type="full",min_covar=mincovar
                                ,  n_iter=1000,covars_prior = covar, transmat_prior = transprior ,means_prior = means_priors[i]).fit(inputtrain,lengths = trainlengths)
                            score = hmmmodel.score
                            if score > prevscore :
                                    selmincovar = mincovar
                                    selalg = alg
                                    selmeanp= means_priors[i]
                                    seltransprior = transprior
                                    selcovarp =covar
                                    model = hmmmodel
                            prevscore = score
    return(model,selcovartype,selalg,selmeanp,seltransprior,selcovarp)
def hmmcompactgrid(nstate,inputtrain,trainlengths,ninits,dimensionality):
    '''
    Acts as a validation procedure to find the best hyper parameters available for the data
    Input : input of hmm model
    output : Best hmm model based on log likelihood value
    '''
    bestscore = float("-inf")
    allscores = []
    # running a grid search with various parameters
    for i in range( ninits):
        nsamp = np.shape(inputtrain)[1]
        lengthaks = [dimensionality] * nsamp
        print "check shape"
        print np.shape(inputtrain)
        nfeats = np.shape(inputtrain)[0]
        
        exmodel = init_gaussian.hmmgaussian(nstate,4,dimensionality,nsamp,nfeats, False)

        (seq_states,deltas,likelihood)= get_seq_statescont.getseqofstatescont(nstate,inputtrain,exmodel)
        likelihood = np.mean(likelihood)
        if likelihood > bestscore :
            bestscore = likelihood
        allscores.append(likelihood)
    maxscore =max(allscores)
    minscore = min(allscores)
    absdiff = float(abs(minscore)- abs(maxscore))
    reldiff = (absdiff/abs(maxscore))
    print "the change in score is "
    print absdiff
    print reldiff
    # saving the best model
    return (seq_states,deltas)
def correlationanalyzer(featmtrx1):
    '''
    analyzed the pearson coefficients between the initial features
    '''
    prefixes = ["first","min","mean", "last","max"]
    corefeats = ["GCS","Temp", "HR","WBC","Glucose","Urine","NIDiasABP"]
    finallabels = []
    for corf in corefeats:
            for pref in prefixes:
                finallabels.append(pref + " " + corf)
    featdataframe = dict.fromkeys(finallabels)
    meanfeatmtrx = np.zeros((np.shape(featmtrx1)[0],7))
    for i in range(7):
        meanfeatmtrx[:,i] = featmtrx1[:,(5 * i) + 2]
        for j in range(5):
            featdataframe[finallabels[(5 * i) + j]] = featmtrx1[:,(5 * i) + j]
    df = pd.DataFrame(featdataframe, range(np.shape(featmtrx1)[0]), finallabels)
    corrmtrx = np.corrcoef(featmtrx1,rowvar = False)[1,0]
    accrcorrmtrx = np.corrcoef(meanfeatmtrx,rowvar = False)[1,0]
def plothistogram(los,c,title,ylimak):
    plt.close()
    plt.hist(los, bins=10, color = c )
    realtitile = "Histogram of length of Stay" + title
    plt.title(realtitile)
    plt.xlabel("Length of Stay")
    plt.ylabel("Frequency")
    plt.ylim(0,ylimak)
    plt.xlim(0,90)
    filename = title + ".png"
    plt.savefig(filename)
def plotresult(testy,testmodely,method,i):
    plt.close() 
    plt.plot(range(len(testy)),testy,'r')
    if method == 'poisson' or method =='negbinom':
        scale = "norm"
    else:
        scale = "log"
    namestr = scale + " " + method
    plottile = namestr + " " + "blue model prediction, red real values"
    namestrexten = "trend" + namestr + str(i) + ".png"
    plt.plot(range(len(testy)),testmodely,'b')
    plt.title(plottile)
    plt.xlabel("sample index")
    plt.ylabel("Length of stay value")
    plt.savefig(namestrexten)
def dosvmandreport(trainingx,trainingy,testx,testy,bestc,besteps,bestgamma,bestkernel):
    if bestkernel == 'linear':
        clf = svm.LinearSVR(C=bestc, epsilon=besteps,loss = 'epsilon_insensitive')
    else :
        clf = svm.SVR(C=bestc, epsilon=besteps,gamma = bestgamma,kernel = bestkernel)
    clf.fit(trainingx, trainingy)
    testmodely = clf.predict(testx)
    return testmodely
def svmcrossval(trainingx,trainingy):
    '''
    runs cross validation for svm initally tested to find the optimum c value
    '''
    cvals = np.linspace(10,100000,10)
    epsvals = np.linspace(0.0000001,0.1,10)
    hypscores = []
    for cval in cvals: 
        for epsval in epsvals: 
            model = svm.LinearSVR(C = cval,epsilon= epsval, loss = 'epsilon_insensitive' )
            scores = cross_val_score(model, trainingx, trainingy, cv=10)
            mse_scores = (-scores)
            hypscores.append(mse_scores.mean())
    dumind= hypscores.index((min(hypscores)))
    dummyc = int(float(math.floor(dumind) )/ float(10))
    bestc = int(cvals[dummyc])
    besteps = epsvals[ dumind - (dummyc * 10) ]
    return (bestc,besteps)
def doLRandreport(trainingx,trainingy,testx,testy,bestalpha,method):
    '''
    Does the actual regression based ont he best provided parameters and returns predictions
    '''
    if method == 'Ridge':
        realmodel = linear_model.Ridge(alpha = bestalpha, max_iter = 10000,fit_intercept=True)
    if method == 'Lasso':
        realmodel = linear_model.Lasso(alpha = bestalpha, max_iter = 10000,fit_intercept=True)
    realmodel.fit(trainingx,trainingy)
    testmodely = realmodel.predict(testx)
    for i in range(len(testy)):
        testmodely[i] = (testmodely[i])
        testy[i] = (testy[i])
    diff =  []
    for i in range(len(testy)):
        diff.append(abs(testmodely[i] - testy[i]))
        bestpatidx = diff.index(min(diff))
        worstpatidx = diff.index(max(diff))
    coefs = realmodel.coef_
    besty = np.inner(testx[bestpatidx,:],coefs)
    realbesty = testy[bestpatidx]
    worsty = np.inner(testx[worstpatidx,:],coefs)
    realworsty = testy[worstpatidx]
    return testmodely
def lrcrossvalidation(trainingx,trainingy,method):
    '''
    finds and returns the best alpha or regularization parameter for the given mode based on the lowest mse score
    '''
    alphs = np.linspace(math.pow(2,-5),math.pow(2,3),num=140) 
    hypscores = []
    for alph in alphs:
        if method == 'Ridge':
            model = linear_model.Ridge(alpha = alph, fit_intercept=True,max_iter =10000)
        if method == 'Lasso':
            model = linear_model.Lasso(alpha = alph, fit_intercept=True,max_iter =10000)
        scores = cross_val_score(model, trainingx, trainingy, scoring="neg_mean_absolute_error", cv=10)
        mse_scores = (-scores)
        hypscores.append(mse_scores.mean())
    bestalpha = alphs[hypscores.index((min(hypscores)))]
    return bestalpha
def logaritmizelos(ar):
    losak = [ float(math.log10(i)) for i in ar ]
    return losak
def xtracttopfeat():
    '''
    Extract top features from the trainng files in physionet dataset and returns feature matrices for all and specific ICU 
    Input : None, assumes you are already within the 'train' folder containing all the 4000 patients data
    Output : Aggregate matrices, specific ICU type matrices, each feature dictionary containing with its key being patient id and 
    and its value being the time points and corresponding value
    '''
    featset = ["GCS","Temp","HR","WBC","Glucose","Urine","NIDiasABP"]
    numbfeats = len(featset)
    # initializations
    featmtrx = np.empty((5,4000,5 * numbfeats))  
    featmtrx1st24 = np.empty((5,4000,5 * numbfeats))  
    featmtrx2nd24 = np.empty((5,4000,5 * numbfeats))  
    icutype = 0
    ages = []
    genders = []
    recids = []
    icutypes = [1,2,3,4]
    paticutype = dict.fromkeys(icutypes) 
    for type in icutypes:
        paticutype[type] = []
    meanfeat = [0] * (len(featset) )
    meanfeat1st24 = [0] * (len(featset))
    meanfeat2nd24 = [0] * (len(featset))
    combkeys = [(1,1),(2,1), (3,1),(4,1), (1,2),(2,2),(3,2), (4,2)]
    dumbak = [0] * 7
    meanfeaticu = dict.fromkeys(combkeys,dumbak)
    meanfeaticualltime = dict.fromkeys(icutypes,dumbak)
    index = -1
    heartratedict = dict.fromkeys(range(4000)) 
    WBCdict = dict.fromkeys(range(4000)) 
    tempdict = dict.fromkeys(range(4000)) 
    GCSdict = dict.fromkeys(range(4000)) 
    urinedict = dict.fromkeys(range(4000)) 
    glucosedict = dict.fromkeys(range(4000)) 
    NIDSdict = dict.fromkeys(range(4000)) 
    changesdictmedians = dict.fromkeys(featset)
    changesdictmeans = dict.fromkeys(featset)
    for featak in featset:
        changesdictmedians[featak] = []
        changesdictmeans[featak] = []
    for d in range(4000):
        heartratedict[d] = []
        WBCdict[d] = []
        tempdict[d] = []
        GCSdict[d] = []
        urinedict[d] = []
        glucosedict[d] = []
        NIDSdict[d] = []
    # reading each patients data, and putting it into the dictionary
    for ind in range(132539,142674):
        filename = str(ind) + ".txt"
        try:
            f = open(filename,'r')
            recids.append(ind)
            index += 1
        except IOError as e:
            pass
        featdict = dict.fromkeys(featset,[])
        featdict1st24 = dict.fromkeys(featset,[])
        featdict2nd24 = dict.fromkeys(featset,[])
        for key in featdict.keys():
            featdict[key] = []
            featdict1st24[key] = []
            featdict2nd24[key] = []
        while(True):
            line = f.readline()
            if not line:
                break
            curline = line.split(",")
            feat = curline[1]
            if  feat in featset:
                timecomp = curline[0].split(":")
                time = (int(timecomp[0]) * 60 ) + int(timecomp[1]) 
                leng = len(curline[2])
                val = curline[2][0:leng-1]
                featdict[feat].append(((time),float(val)))
                if time <= (24 * 60):
                    featdict1st24[feat].append(((time),float(val)))
                else:
                    featdict2nd24[feat].append(((time),float(val)))
            if feat == "Age":
                ages.append(curline[2].strip('\n'))
            if feat == "Gender":
                genders.append(curline[2].strip('\n'))
            if feat == "ICUType" :
                leng = len(curline[2])
                paticutype[int(curline[2][0:leng-1])].append(index)
                icutype = int(curline[2][0:leng-1])
                f.readline()
        for i in range(len(featset)):
            tuplesak = sorted (featdict[featset[i]], key = lambda x : x[0])
            if len(tuplesak) > 0:
                tuplevalues = [indie[1] for indie in tuplesak if math.isnan(indie[1]) == False]
                tupletimes = [indi[0] for indi in tuplesak if math.isnan(indi[1]) == False]                
                diffvalues = []
                ij = 1
                while (ij<len(tuplevalues)):
                    nextij = ij
                    while nextij < len(tuplevalues) :
                        if tuplevalues[nextij] == tuplevalues[nextij-1] :
                            nextij += 1
                        else:
                            break
                    if nextij < len(tuplevalues):
                        diffvalues.append(tupletimes[nextij] - tupletimes[ij-1] )
                        ij = nextij + 1
                    else:
                        diffvalues.append(tupletimes[nextij-1] - tupletimes[ij-1] )
                        ij = nextij + 1
                meandiffvalues = np.mean(np.array(diffvalues))
                mediandiffvalues = np.median(np.array(diffvalues))
                changesdictmeans[featset[i]].append(meandiffvalues)    
                changesdictmedians[featset[i]].append(mediandiffvalues)  
            if featset[i] == "HR":
                tuples = sorted (featdict[featset[i]], key = lambda x : x[0])                                         
                heartratedict[index].append(tuples)
            if featset[i] == "WBC":
                tuples = sorted (featdict[featset[i]], key = lambda x : x[0])                  
                WBCdict[index].append(tuples)
            if featset[i] == "GCS":
                tuples = sorted (featdict[featset[i]], key = lambda x : x[0])                   
                GCSdict[index].append(tuples)
            if featset[i] == "Temp":
                tuples = sorted (featdict[featset[i]], key = lambda x : x[0])                   
                tempdict[index].append(tuples)
            if featset[i] == "Glucose":
                tuples = sorted (featdict[featset[i]], key = lambda x : x[0])                  
                glucosedict[index].append(tuples)
            if featset[i] == "Urine":
                tuples = sorted (featdict[featset[i]], key = lambda x : x[0]) 
                for innn in range(len(tuples)):
                    tuples = [( tuples[innn][0] , sum([tuples[j][1] for j in range(innn+1) if tuples[j][1] < 10000 ])) for innn in range(len(tuples))]              
                urinedict[index].append(tuples)
            if featset[i] == "NIDiasABP":
                tuples = sorted (featdict[featset[i]], key = lambda x : x[0])               
                NIDSdict[index].append(tuples)          
            if len(featdict[featset[i]]) > 0: 
                if len(featdict1st24[featset[i]]) > 0: 
                    featmtrx1st24[icutype,index,5 * i] = min(featdict1st24[featset[i]], key= lambda t : t[0])[1]
                    featmtrx1st24[icutype,index,(5 * i) + 1] = min(featdict1st24[featset[i]], key= lambda t : t[1])[1]
                    vals = [v[1] for v in featdict1st24[featset[i]] if v[1]!= -1]
                    featmtrx1st24[icutype,index,(5 * i) + 2] = float(sum(vals) / float(len(vals)) )
                    meanfeat1st24[i] += featmtrx1st24[icutype,index,(5 * i) + 2]
                    featmtrx1st24[icutype,index,(5 * i) + 3] = max(featdict1st24[featset[i]], key= lambda t : t[1])[1]
                    featmtrx1st24[icutype,index,(5 * i) + 4] = max(featdict1st24[featset[i]], key= lambda t : t[0])[1]
                    (meanfeaticu[(icutype,1)])[i] += featmtrx1st24[icutype,index,(5 * i) + 2]
                if len(featdict2nd24[featset[i]]) > 0: 
                    featmtrx2nd24[icutype,index,5 * i] = min(featdict2nd24[featset[i]], key= lambda t : t[0])[1]
                    featmtrx2nd24[icutype,index,(5 * i) + 1] = min(featdict2nd24[featset[i]], key= lambda t : t[1])[1]
                    vals = [v[1] for v in featdict2nd24[featset[i]] if v[1]!= -1]
                    featmtrx2nd24[icutype,index,(5 * i) + 2] = float(sum(vals) / float(len(vals)) )
                    meanfeat2nd24[i] += featmtrx2nd24[icutype,index,(5 * i) + 2]
                    featmtrx2nd24[icutype,index,(5 * i) + 3] = max(featdict2nd24[featset[i]], key= lambda t : t[1])[1]
                    featmtrx2nd24[icutype,index,(5 * i) + 4] = max(featdict2nd24[featset[i]], key= lambda t : t[0])[1]
                    (meanfeaticu[(icutype,2)])[i] += featmtrx2nd24[icutype,index,(5 * i) + 2]
                featmtrx[icutype,index,5 * i] = min(featdict[featset[i]], key= lambda t : t[0])[1]
                featmtrx[icutype,index,(5 * i) + 1] = min(featdict[featset[i]], key= lambda t : t[1])[1]
                vals = [v[1] for v in featdict[featset[i]] if v!= -1]
                featmtrx[icutype,index,(5 * i) + 2] = float(sum(vals) / float(len(vals)) )
                meanfeat[i] += featmtrx[icutype,index,(5 * i) + 2]
                meanfeaticu[(icutype,2)][i] += featmtrx[icutype,index,(5 * i) + 2]
                featmtrx[icutype,index,(5 * i) + 3] = max(featdict[featset[i]], key= lambda t : t[1])[1]
                featmtrx[icutype,index,(5 * i) + 4] = max(featdict[featset[i]], key= lambda t : t[0])[1]
    meanfeat = [float(i) / 4000.00 for i in meanfeat]
    meanfeat1st24 = [float(i) / 4000.00 for i in meanfeat1st24]
    meanfeat2nd24 = [float(i) / 4000.00 for i in meanfeat2nd24]
    for kk in combkeys:
        numero = float(len(paticutype[kk[0]]))
        meanfeaticu[kk] = [ float(float(m) / numero)  for m in meanfeaticu[kk]]
    for type in icutypes:
        dummylist = np.array([meanfeaticu[(type,1)],meanfeaticu[(type,2)]])
        meanfeaticualltime[type] = np.average(dummylist)
    # computing mean across features for imputation purposes
    for j in range(4000):
        for k in range(numbfeats):
            for l in range(5):
                for type in icutypes:
                    if featmtrx[type, j,(5 * k )+ l] == -1 or math.isnan(featmtrx[type, j,(5 * k )+ l]) == True:
                        featmtrx[type, j,(5 * k ) +l] = meanfeaticualltime[type][(5 * k ) +l]
                    if featmtrx1st24[type, j,(5 * k )+ l] == -1 or  math.isnan(featmtrx1st24[type, j,(5 * k )+ l]) == True :
                        featmtrx1st24[type, j,(5 * k ) +l] = meanfeaticu[(type,1)][(5 * k ) +l]
                    if featmtrx2nd24[type, j,(5 * k )+ l] == -1 or math.isnan(featmtrx2nd24[type, j,(5 * k )+ l]) == True:
                        featmtrx2nd24[type, j,(5 * k ) +l] = meanfeaticu[(type,2)][(5 * k ) +l]
    # constructing feature matrices containing only features , removing ICU type as an additional axis
    realfeat = [0] * 5
    realfeat1st24 = [0] * 5
    realfeat2nd24 = [0] * 5
    for type in icutypes:
        realfeat[type] = np.empty((len(paticutype[type]),5*numbfeats))
        realfeat[type] = featmtrx[type,paticutype[type],:]
        realfeat1st24[type] = np.empty((len(paticutype[type]),5*numbfeats))
        realfeat1st24[type] = featmtrx1st24[type,paticutype[type],:]
        realfeat2nd24[type] = np.empty((len(paticutype[type]),5*numbfeats))
        realfeat2nd24[type] = featmtrx2nd24[type,paticutype[type],:]
    # computing average mean of change for future purposes
    AVGmeanchangefeats = dict.fromkeys(featset)
    AVGmedianchangefeats = dict.fromkeys(featset)
    for fea in featset:
        AVGmeanchangefeats[fea] = np.nanmean((changesdictmeans[fea]))
        AVGmedianchangefeats[fea] = np.nanmedian((changesdictmedians[fea]))  
    wholefeat = np.empty((4000,5*numbfeats))
    for type in icutypes:
        wholefeat[paticutype[type],:] = realfeat[type]
    return (wholefeat,realfeat[1], realfeat[2],\
     realfeat[3],realfeat[4],realfeat1st24[1],\
     realfeat1st24[2],realfeat1st24[3],realfeat1st24[4],realfeat2nd24[1],realfeat2nd24[2],\
     realfeat2nd24[3],realfeat2nd24[4],paticutype,heartratedict,ages,genders,WBCdict,tempdict,GCSdict,glucosedict,NIDSdict,urinedict,AVGmeanchangefeats,AVGmedianchangefeats,recids)
def xtrlenofstay(paticutypedictindex):
    '''
    Extracts length of stay of patients
    Input : Change name of file and folder name accordingly 
    Output: length of stay of aggregate and particular ICU types, record id of patients with no LOS
    '''
    invalids = []
    recids = []
    los = np.empty((4000,1))
    os.chdir("..")
    os.chdir(os.path.abspath(os.curdir) + "/HMMphys")
    f = open("Outcomes-a.txt",'r')
    f.readline() 
    i = 0
    for line in f:
        lenofstay = int(line.split(",")[3])
        recid = int(line.split(",")[0])
        recids.append(recid)
        if lenofstay >= 0:
            los[i] = int((float(lenofstay)))
        else :
            invalids.append(recid)
        i +=1
    types = [1,2,3,4]
    lengs = dict.fromkeys(types)
    for type in types:
        lengs[type]= []
    for i in range(1,5):
        lengs[i] = [los[j] for j in paticutypedictindex[i]]
    return (los,lengs[1],lengs[2],lengs[3],lengs[4],invalids,recids)
def stratifyBOlos(featmtrx,los):
    '''
    Stratifies training and test sets to make sure that similar popoulation of patients are put into both groups in terms of length of stay
    '''

    # buckets for cutoff of the LOS days
    losbuckets = list(range(0,14))
    losbuckets.append(15)
    losbuckets.append(18)
    losbuckets.append(21)
    losbuckets.append(28)
    losbuckets.append(35)
    strataindices = []
    for i in range(19):
        strataindices.append([])
    for i in range(len(los)):
        for j in range(len(losbuckets) -1):
            if los[i] >= losbuckets[j] and los[i] < losbuckets[j+1] :
                strataindices[j].append(i)
                break
        if los[i] >= losbuckets[len(losbuckets) -1] :
            strataindices[18].append(i)
    lengstrata = []
    for ll in range(19):
        lengstrata.append(len(strataindices[ll]))
    testindices = []
    trainindices = []
    for j in range(19):
        testindices.append([])
        trainindices.append([])
    for i in range(0,19):
        numtest = int(math.ceil(float(len(strataindices[i])  * 0.2 )))
        if len(strataindices[i]) > 0 and numtest >=1:
            randindices = np.random.randint(0, len(strataindices[i]), numtest)
            for index in randindices:
                testindices[i].append(strataindices[i][index])
            for indice in strataindices[i]:
                if indice not in testindices[i]:
                    trainindices[i].append(indice)
    testindex = []
    trainindex = []
    for indie in range(19):
        testindex += ( testindices[indie])
        trainindex += ( trainindices[indie])
    trainindex2 = list(set(trainindex))
    testindex2 = list(set(testindex))
    sum1 = []
    sum2 = []
    sum1 += trainindex
    sum1 += testindex
    sum2 += trainindex2
    sum2 += testindex2
    diff = [i for i in sum1 if i not in sum2]
    return (testindex2,trainindex2)
def plotscatter(testy,testmodely,method,i):
    if method == 'poisson' or method =='negbinom':
        scale = "norm"
    else:
        scale = "log"
    namestr = scale + " " + method
    plottile = namestr + " " + "blue model prediction, red real values"
    namestrexten = "scatter " + namestr  + ".png"
    plt.close()
    xr = np.linspace(min(min(testmodely),min(testy)),max(max(testmodely),max(testy)),100)
    yr = np.linspace(min(min(testmodely),min(testy)),max(max(testmodely),max(testy)),100)
    plt.scatter(testy,testmodely,c = 'k')
    plt.plot(xr,yr,'r')
    plt.title(plottile)
    plt.xlabel("testy")
    plt.ylabel("testmodely")
    plt.savefig(namestrexten)
def svmrandomcrossval(trainingx,trainingy):
    param_dist = {"C": scipy.stats.expon(scale=100),
              "epsilon": np.linspace(0,4,2000),
              "kernel":['rbf'],
              "gamma":scipy.stats.expon(scale=.1),
              }
    model = svm.SVR(C=1.0, epsilon=0.2)
    random_search = RandomizedSearchCV(model, param_distributions=param_dist,n_iter = 5500,cv = 5)
    random_search.fit(trainingx,trainingy)
    results = random_search.cv_results_
    bestdict = random_search.best_params_ 
    bestc = bestdict["C"]
    besteps = bestdict["epsilon"]
    bestgamma = bestdict["gamma"]
    bestkernel = bestdict["kernel"]
    return (bestc,besteps,bestgamma,bestkernel)
def poissregr(trainingx,trainingy,testx):
    poi_model = sm.GLM(trainingy,trainingx, family=sm.families.Poisson())
    poi_results = poi_model.fit()
    paramet = poi_results.params
    testmodelymu= poi_results.predict(testx)
    testmodely = np.empty((len(testmodelymu),1))
    for i in range(len(testmodelymu)) :
        sample = int(stats.poisson.rvs(testmodelymu[i]))
        testmodely[i] = sample
    return testmodely
def negbinomregr(trainingx, trainingy,testx):
    neg_model = sm.GLM(trainingy,trainingx, family=sm.families.NegativeBinomial())
    neg_results = neg_model.fit()
    paramet = neg_results.params
    testmodelymu = neg_results.predict(testx)
    testmodely = np.empty((len(testmodelymu),1))
    for i in range(len(testmodelymu)) :
        sample = int(stats.nbinom.rvs(testmodelymu[i],1,1))
        testmodely[i] = sample
    return testmodely
def performbaseline(trainingx,ytrain,testx,ytest):
    scaler = sklearn.preprocessing.StandardScaler().fit(trainingx)
    trainingx = scaler.transform(trainingx)
    testx = scaler.transform(testx)
    featmtrx = np.concatenate((trainingx,testx),axis = 0)
    numtrain = np.shape(trainingx)[0]
    if hmm:
        pcafeatmtrx = featmtrx
    else:
        ncomp = applybestPCA(featmtrx)
        pca = PCA(n_components=ncomp)
        pca.fit(featmtrx)
        pcafeatmtrx = pca.fit_transform(featmtrx)
    trainingfeats = pcafeatmtrx[0:numtrain,:]
    testfeats = pcafeatmtrx[numtrain:,:]
    bestalpha = lrcrossvalidation(trainingfeats,ytrain,'Lasso')
    testmodely = doLRandreport(trainingfeats, ytrain, testfeats, ytest,bestalpha,'Lasso')
    testmse =  np.sqrt((float( np.sum( (testmodely - ytest) ** 2 ))) / float(len(testmodely)))
    return testmse
def learnhmm(validpatientsindices,KNNfeats,reallos,inputHmmallVars,ovlapinputHmmallVars,trainindices,testindices,dictindices,everyxhours,nstates,icutype,over,model):
    '''
    Learns the HMM model using the aleady computed input from data and outputs probability features for future use
    Input: valid patient indices, input to HMM model, time resoultuon , configuration of the desired model like overlapping or non overlapping model
    Output: Train and test feature matrices showing the probability of belonging to the first and last most likely states 

    '''

    # initializations
    numvars = len(dictindices)
    featnames = {1 : "HeartRate",2 : "WBC", 3: "Temperature" ,4 : "GCS" ,5 : "Glucose" ,  6 : "NIDS" , 7 : "Urine"   }
    npatients = np.shape(KNNfeats)[0]
    dimensionality = int(math.floor(48.0/float(everyxhours)))
    overdimensionality = (2 * dimensionality) -1 
    inputhmm=np.zeros((npatients,dimensionality,numvars))
    ovlapinputhmm=np.zeros((npatients,overdimensionality,numvars))
    # reshaping input for hmm to conform to the hmm module
    for i in range((npatients)):
        for j in range(numvars):
            inputhmm[i,:,j] = (inputHmmallVars[dictindices[j]])[i,:]
            ovlapinputhmm[i,:,j] = (ovlapinputHmmallVars[dictindices[j]])[i,:]
    inputhmm =[ inputhmm[idx,:] for idx in validpatientsindices]
    ovlapinputhmm = [ovlapinputhmm[idx,:,:] for idx in validpatientsindices ]
    KNNtrain = np.empty((len(trainindices),2))
    KNNtest = np.empty((len(testindices),2))
    for i in range(len(trainindices)):
        KNNtrain[i,:] = KNNfeats[trainindices[i],:]
    for i in range(len(testindices)):
        KNNtest[i,:] = KNNfeats[testindices[i],:]
    # impute featuers usign KNN features of age and gender
    (inputhmmtrain,inputhmmtest) = imputefeatures(trainindices, testindices, inputhmm,KNNtrain,KNNtest,dimensionality,numvars)
    (ovlapinputhmmtrain,ovlapinputhmmtest) = imputefeatures(trainindices, testindices, ovlapinputhmm,KNNtrain,KNNtest,overdimensionality,numvars) 
    inputtrain = inputhmmtrain
    ovlapinputtrain = ovlapinputhmmtrain  
    inputtest = inputhmmtest
    ovlapinputtest =  ovlapinputhmmtest
    realinputtrain = swapaxes(inputtrain)
    realovlapinputtrain = swapaxes(ovlapinputtrain)
    realinputtest= swapaxes(inputtest)
    realovlapinputtest = swapaxes(ovlapinputtest)

    print "Viz an Viz"
    # print np.shape(ovlapinputtest)
    # preparing the input hmm to be fed into appropriate form , flattening it
    # if numvars == 1:
    #     inputtrain = inputtrain.flatten()
    #     inputtest = inputtest.flatten()
    #     ovlapinputtest = ovlapinputtest.flatten()    
    #     ovlapinputtrain = ovlapinputtrain.flatten() 
    #     realinputtrain = [[inputtrain[i]] for i in range(len(inputtrain))] 
    #     realovlapinputtrain = [[ovlapinputtrain[i]] for i in range(len(ovlapinputtrain))] 
    #     realinputtest = [[inputtest[i]] for i in range(len(inputtest))] 
    #     realovlapinputtest = [[ovlapinputtest[i]] for i in range(len(ovlapinputtest))] 
    # else:
    #     inputtrain = inputtrain.reshape((len(trainindices)* dimensionality,numvars))
    #     inputtest = inputtest.reshape((len(testindices) * dimensionality,numvars))
    #     normalovlapinputtest = ovlapinputtest.reshape((len(testindices) ,overdimensionality,numvars))  
    #     ovlapinputtest = ovlapinputtest.reshape((len(testindices) * overdimensionality,numvars))  
    #     ovlapinputtrain = ovlapinputtrain.reshape((len(trainindices) * overdimensionality,numvars))  
    #     realinputtrain = inputtrain
    #     realovlapinputtrain = ovlapinputtrain
    #     realinputtest = inputtest
    #     realovlapinputtest = ovlapinputtest
    reallos = [numb * 24 for numb in reallos ]
    ytrain = [reallos[i] for i in trainindices]
    ytest = [reallos[i] for i in testindices]
    trainlengths =(np.array((len(trainindices) * numvars,1))).fill(dimensionality)
    ovlaptrainlengths =(np.array((len(trainindices) * numvars,1))).fill(overdimensionality)
    testlengths = (np.array((len(testindices) * numvars,1))).fill(dimensionality)
    ovlaptestlengths = (np.array((len(testindices) * numvars,1))).fill(overdimensionality)    
    noinits = 2
    # running the actual hmm model here
    (seq_states,deltas) = hmmcompactgrid(nstates,realinputtrain,trainlengths,noinits,dimensionality)
    (ovlapseq_states,ovlapdeltas) = hmmcompactgrid(nstates, realovlapinputtrain, ovlaptrainlengths, noinits,overdimensionality)
    # loadign the best model to get the data
    name = "bestmodelak " + str(dimensionality) + ".pkl"
    ovlapname = "bestmodelak " + str(overdimensionality) + ".pkl"
    chosenmodel = joblib.load(name)
    ovlapchosenmodel = joblib.load(ovlapname)
    # getting the probabilities and states
    probs = chosenmodel.predict_proba(realinputtrain)
    probstest = chosenmodel.predict_proba(realinputtest)
    ovlapprobs = ovlapchosenmodel.predict_proba(realovlapinputtrain)
    ovlapprobstest = ovlapchosenmodel.predict_proba(realovlapinputtest)
    ordtransmat= chosenmodel.transmat_
    ovlaptransmat= ovlapchosenmodel.transmat_
    ordpii = chosenmodel.startprob_
    ovlappii = ovlapchosenmodel.startprob_
    plt.close()
    # plotting the heatmap for patients who ended up in different state end state pairs
    ax = sns.heatmap(chosenmodel.transmat_)
    plt.xlabel("State Index")
    plt.ylabel("State Index")
    realtitle = "transitionheatmap" + str(nstates) + "res =" + str(everyxhours) + "icutype" + str(icutype) +  ".png"
    (plt.savefig(realtitle))
    trainstates = list(chosenmodel.predict(realinputtrain))
    ovlaptrainstates = list(ovlapchosenmodel.predict(realovlapinputtrain))
    maxcounttrain = max(set(trainstates), key = trainstates.count)
    ovlapmaxcounttrain = max(set(ovlaptrainstates), key = ovlaptrainstates.count)    
    maxcounttrain = trainstates.count(maxcounttrain)
    ovlapmaxcounttrain = trainstates.count(ovlapmaxcounttrain)    
    teststates = list(chosenmodel.predict(realinputtest,lengths = testlengths))
    ovlapteststates = list(ovlapchosenmodel.predict(realovlapinputtest,lengths = ovlaptestlengths))    
    maxcounttest = max(set(teststates), key = teststates.count) 
    ovlapmaxcounttest = max(set(ovlapteststates), key = ovlapteststates.count) 
    maxcounttest = teststates.count(maxcounttest)
    maxcounttest = teststates.count(ovlapmaxcounttest)
    numpatients = len(trainindices)
    indivtrainstates = np.reshape(trainstates,(len(trainindices),dimensionality))
    trainstatescount = np.zeros((len(trainindices),nstates))
    ovlaptrainstatescount = np.zeros((len(trainindices),nstates))
    indivteststates = np.reshape(teststates,(len(testindices),dimensionality))
    teststatescount = np.zeros((len(testindices),nstates))
    ovlapteststatescount = np.zeros((len(testindices),nstates))
    # Sepratign individual probablities
    indivprobs = np.reshape(probs,(len(trainindices),dimensionality,nstates))
    ovlapindivtrainstates = np.reshape(ovlaptrainstates,(len(trainindices),overdimensionality)) 
    ovlapindivteststates = np.reshape(ovlapteststates,(len(testindices),overdimensionality))     
    ovlapindivprobs = np.reshape(ovlapprobs,(len(trainindices),overdimensionality,nstates))
    ovlapindivprobstest =np.reshape(ovlapprobstest,(len(testindices),overdimensionality,nstates)) 
    indivprobstest =np.reshape(probstest,(len(testindices),dimensionality,nstates)) 
    indiv_chosenstate_probs = np.empty((numpatients,dimensionality))
    ovlapindiv_chosenstate_probs = np.empty((numpatients,overdimensionality))
    indiv_chosenstate_probstest = np.empty((len(testindices),dimensionality))
    ovlapindiv_chosenstate_probstest = np.empty((len(testindices),overdimensionality))
    # quantifying number of patients fallilng into different groups
    for i in range(len(trainindices)):
        for j in range(nstates):
            trainstatescount[i,j] = np.sum(j == indivtrainstates[i,:])
            ovlaptrainstatescount[i,j] = np.sum(j == ovlapindivtrainstates[i,:])
    for i in range(len(testindices)):
        for j in range(nstates):
            teststatescount[i,j] = np.sum(j == indivteststates[i,:])
            ovlapteststatescount[i,j] = np.sum(j == ovlapindivteststates[i,:])
    for i in range(numpatients):
        for j in range(dimensionality) :
            indiv_chosenstate_probs[i,j] = max(indivprobs[i,j,:])
        for k in range(overdimensionality):
            ovlapindiv_chosenstate_probs[i,k] = max(ovlapindivprobs[i,k,:])
    for i in range(len(testindices)):
        for j in range(dimensionality) :
            indiv_chosenstate_probstest[i,j] = max(indivprobstest[i,j,:])
        for k in range(overdimensionality):
            ovlapindiv_chosenstate_probstest[i,k] = max(ovlapindivprobstest[i,k,:])
    firstovlapindivtrainstates = ovlapindivtrainstates[:,0]
    firstovlapindivteststates = ovlapindivteststates[:,0]
    firsttraindic = dict.fromkeys(range(nstates),0)
    firsttestdic = dict.fromkeys(range(nstates),0)
    for state in firstovlapindivtrainstates:
        firsttraindic[state] += 1
    for stated in firstovlapindivteststates:
        firsttestdic[stated] += 1
    numbaktr = len(trainindices)
    numbaktes= len(testindices)
    trainstateperc = [ float (item /float(numbaktr)) for item in [firsttraindic[stateidx] for  stateidx in range(nstates)]] 
    teststateperc = [ float (it /float(numbaktes)) for it in [firsttestdic[staten] for  staten in range(nstates)]]
    plt.clf()
    # plotting related to choosing the best number of states based on higher yieldign probabilities
    probtitle = "probheatmap" + str(nstates) + "res =" + str(everyxhours) + "icutype" + str(icutype) +  ".png"
    ax = sns.heatmap([teststateperc])
    plt.title("Starting State")
    (plt.savefig(probtitle))
    plt.clf()
    plt.close()
    plt.bar(range(nstates),trainstateperc )
    plt.xlabel("State idx")
    plt.ylabel("Percentage Frequency")
    filename = "trainingfirststates" + str(icutype) +  ".png"
    plt.savefig(filename)
    plt.close()
    plt.bar(range(nstates),teststateperc )
    plt.xlabel("State idx")
    plt.ylabel("Percentage Frequency")
    filename = "testfirststates" + str(icutype) +  ".png"
    plt.savefig(filename)
    plt.close()
    plt.close()
    mean_chosenstate_prob_ord = [np.mean(indiv_chosenstate_probs[:,i]) for i in range(dimensionality)]
    mean_chosenstate_prob_ordovertimepoints = [np.mean(indiv_chosenstate_probs[i,:]) for i in range((numpatients))]
    mean_chosenstate_prob_ovlapovertimepoints = [np.mean(ovlapindiv_chosenstate_probs[i,:]) for i in range((numpatients))]
    plt.plot(range(dimensionality),mean_chosenstate_prob_ord)
    plt.xlabel("Time point")
    plt.ylabel("Mean probability of the most probable state across all patients")
    title = "Certatinty change througout the time, non-overlapping, with res =  " + str(everyxhours) + "and nstates = " + str(nstates) 
    plt.title(title)
    filenm = "certaintychangeovertimenonovlap" + str(everyxhours) + str(nstates) + ".png"
    plt.savefig(filenm)
    plt.close()
    mean_chosenstate_prob_ovlap = [np.mean(ovlapindiv_chosenstate_probs[:,i]) for i in range(overdimensionality)]    
    plt.plot(range(overdimensionality),mean_chosenstate_prob_ovlap)
    plt.xlabel("Time point")
    plt.ylabel("Mean probability of the most probable state across all patients")
    title = "Certatinty change througout the time, overlapping, with res =  " + str(everyxhours) + "and nstates = " + str(nstates) 
    plt.title(title)
    filenm = "certaintychangeovertimeovlap" + str(everyxhours) + str(nstates) + ".png"
    plt.savefig(filenm)
    # Playing with patients who fall into certain start end paris to analyze theier charachteristics
    (orduncertainidx,ovlapuncertainidx) = filterbasedoncertainty(indivprobs,ovlapindivprobs,onlyfirstandlast =False)
    percentage_uncertain_ord = float(float(len(orduncertainidx)) / float(len(trainindices)))
    percentage_uncertain_ovlap= float(float(len(ovlapuncertainidx)) / float(len(trainindices)))
    certainordidx = list(set(range(len(trainindices))) - set (orduncertainidx))
    certaiovlapdidx = list(set(range(len(trainindices))) - set (ovlapuncertainidx))
    indivtrainstatesfiltered = indivtrainstates[certainordidx]
    ovlapindivtrainstatesfiltered = ovlapindivtrainstates[certaiovlapdidx]
    (changestatindices,ovlapchangestatindices,numchangestate,ovlapnumchangestate) = detectchangeinstates(indivtrainstates,ovlapindivtrainstates)
    (idxbiggestlosord,idxbiggestlosovlap,ovlaplosidx) = startendlosvisualizer(indivtrainstates,ovlapindivtrainstates,changestatindices,ovlapchangestatindices,nstates,ytrain,everyxhours,icutype)
    (idxbiggestlosordtest,idxbiggestlosovlaptest,ovlaplosidxtest) = startendlosvisualizer(indivteststates,ovlapindivteststates,changestatindices,ovlapchangestatindices,nstates,ytrain,everyxhours,icutype)
    heatmapofvitalschange(ovlaplosidxtest,nstates,normalovlapinputtest)
    weighted = False
    plotlongestlosbasedoncertainty(idxbiggestlosord,idxbiggestlosovlap,ytrain,indiv_chosenstate_probs,ovlapindiv_chosenstate_probs,everyxhours,nstates,indivprobs,ovlapindivprobs,weighted,icutype)
    (changestatindicesfitered,ovlapchangestatindicesfiltered,numchangestatefiltered,ovlapnumchangestatefiltered) = detectchangeinstatesfiltered(indivtrainstatesfiltered,ovlapindivtrainstatesfiltered)
    startendlosvisualizerfiltered(indivtrainstatesfiltered,ovlapindivtrainstatesfiltered,certainordidx,certaiovlapdidx,changestatindicesfitered,ovlapchangestatindicesfiltered,nstates,ytrain,everyxhours,icutype)
    (idxbiggestlosordweight,idxbiggestlosovlapweight,ordAvgVarPatches,ordVarRadiiPatchesmean,ordVarRadiiPatchesmedian,ovlapAvgVarPatches,ovlapVarRadiiPatchesmean,ovlapVarRadiiPatchesmedian)= startendlosvisualizerweighted(indivtrainstates,ovlapindivtrainstates,changestatindices,ovlapchangestatindices,nstates,ytrain,everyxhours,mean_chosenstate_prob_ordovertimepoints,mean_chosenstate_prob_ovlapovertimepoints,icutype)
    weighted = True
    # plot long staying patients
    plotlongestlosbasedoncertainty(idxbiggestlosordweight,idxbiggestlosovlapweight,ytrain,indiv_chosenstate_probs,ovlapindiv_chosenstate_probs,everyxhours,nstates,indivprobs,ovlapindivprobs,weighted,icutype)
    loschg1stst = [[] for i in range(nstates)]
    losnonchg1stst = [[] for i in range(nstates)]
    ovlaploschg1stst = [[] for i in range(nstates)]
    ovlaplosnonchg1stst = [[] for i in range(nstates)]
    d1 = []
    for i in range(nstates):
        d1.append([x - y for x, y in zip(loschg1stst[i], losnonchg1stst[i])])
    d2 = []
    for i in range(nstates):
        d2.append([x - y for x, y in zip(ovlaploschg1stst[i], ovlaplosnonchg1stst[i])])
    numvalidpatients = len(trainindices)
    numtestpatients = len(testindices)
    traininghmmfeats = np.zeros((numpatients, 2 * nstates))
    testhmmfeats = np.zeros((numtestpatients, 2 * nstates))
    if model == "firstlastprobs":
        # this was the better model and was used in the paper
        if ~ (over) :
            for i in range(numtestpatients):
                testhmmfeats[i,0:nstates ] = indivprobstest[i,0,:] 
                testhmmfeats[i,nstates:2 * nstates ] = indivprobstest[i,-1,:] 
            for i in range(numpatients):
                traininghmmfeats[i,0:nstates ] = indivprobs[i,0,:] 
                traininghmmfeats[i,nstates:2 * nstates ] = indivprobs[i,-1,:] 
        if over :
            for i in range(numtestpatients):
                testhmmfeats[i,0:nstates ] = ovlapindivprobstest[i,0,:] 
                testhmmfeats[i,nstates:2 * nstates ] = ovlapindivprobstest[i,-1,:] 
            for i in range(numpatients):
                traininghmmfeats[i,0:nstates ] = ovlapindivprobs[i,0,:] 
                traininghmmfeats[i,nstates:2 * nstates ] = ovlapindivprobs[i,-1,:] 
    if model == "twomostprobs":
        (ordfirstmaxstateidx,ordsecondmaxstateidx,ovlapfirstmaxstateidx,ovlapsecondmaxstateidx) = find2mostcertaintimepointsidx(indiv_chosenstate_probs,ovlapindiv_chosenstate_probs)
        (ordfirstmaxstateidxtest,ordsecondmaxstateidxtest,ovlapfirstmaxstateidxtest,ovlapsecondmaxstateidxtest) = find2mostcertaintimepointsidx(indiv_chosenstate_probstest,ovlapindiv_chosenstate_probstest)
        if ~ (over) :
            for i in range(numtestpatients):
                testhmmfeats[i,0:nstates ] = indivprobstest[i,ordfirstmaxstateidxtest[i],:] 
                testhmmfeats[i,nstates:2 * nstates ] = indivprobstest[i,ordsecondmaxstateidxtest[i],:] 
            for i in range(numpatients):
                traininghmmfeats[i,0:nstates ] = indivprobs[i,ordfirstmaxstateidx[i],:] 
                traininghmmfeats[i,nstates:2 * nstates ] = indivprobs[i,ordsecondmaxstateidx[i],:] 
        if over :
            for i in range(numtestpatients):
                testhmmfeats[i,0:nstates ] = ovlapindivprobstest[i,ovlapfirstmaxstateidxtest[i],:] 
                testhmmfeats[i,nstates:2 * nstates ] = ovlapindivprobstest[i,ovlapsecondmaxstateidxtest[i],:] 
            for i in range(numpatients):
                traininghmmfeats[i,0:nstates ] = ovlapindivprobs[i,ovlapfirstmaxstateidx[i],:] 
                traininghmmfeats[i,nstates:2 * nstates ] = ovlapindivprobs[i,ovlapsecondmaxstateidx[i],:] 
    return ( ordscore,ordselalg,ordselcovartype,ovlapscore,ovlapselalg,ovlapselcovartype , \
    traininghmmfeats,testhmmfeats,ytrain,ytest,ordAvgVarPatches,ordVarRadiiPatchesmean,ordVarRadiiPatchesmedian, \
    ovlapAvgVarPatches,ovlapVarRadiiPatchesmean,ovlapVarRadiiPatchesmedian ,ordtransmat, ovlaptransmat , ordpii , ovlappii ) 
def plotvariance(nstates,los,title):
    plt.close()
    var = [0] * nstates
    for i in range(nstates):
        var[i] = (np.var(los[i]))
    fig = plt.bar(range(nstates),var,align='center', alpha=0.1)
    for i in range(nstates):
        fig[i].set_color('r')
    plt.xlabel("firat state ")
    plt.ylabel("variance of length of stay")
    plt.title(title)
    realtitle = title + ".png"
    plt.savefig(realtitle)
main()
