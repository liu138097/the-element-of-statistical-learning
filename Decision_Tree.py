# -*- coding: utf-8 -*- 
import numpy as np
from math import log

#ID3算法
def calcShannonEnt(dataSet):
    numEntries=len(dataSet)
    labelCounts={}
    for featVec in dataSet:
        label=featVec[-1]
        labelCounts[label]=labelCounts.get(label,0)+1
    shannonEnt=-sum([p/numEntries *np.log2(p/numEntries) for p in labelCounts.values()])
    return shannonEnt

def createDataSet():
    dataSet=[[1,1,'yes'],
             [1,1,'yes'],
             [1,0,'no'],
             [0,1,'no'],
             [0,1,'no']]
    labels=['no surfacing','flippers']
    return dataSet,labels
myDat,labels=createDataSet()
print(myDat)
print(calcShannonEnt(myDat))

def splitDataSet(dataSet,axis,value):
    retDataSet=[]
    for featVec in dataSet:
        if featVec[axis]==value:
            reducedFeatVec=featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

print(splitDataSet(myDat,0,1))

def chooseBestFeatureToSplit(dataSet):
    numFeatures=len(dataSet[0])-1
    baseEntropy=calcShannonEnt(dataSet)
    bestinfoGain=0.0
    for i in range(numFeatures):
        featList=[example[i] for example in dataSet]
        uniqueVals=set(featList)
        newEntropy=0.0
        for value in uniqueVals:
            subDataSet=splitDataSet(dataSet,i,value)
            prob=len(subDataSet)/float(len(dataSet))
            newEntropy+=prob*calcShannonEnt(subDataSet)
        infoGain=baseEntropy-newEntropy
        if(infoGain>bestinfoGain):
            bestinfoGain=infoGain
            bestFeature=i
    return bestFeature

print(chooseBestFeatureToSplit(myDat))

import operator
def majorityCnt(classList):
    classCount={}
    for vote in classList:
        classCount[vote]=classCount.get(vote,0)+1
    sortedClassCount=sorted(classCount.items(),key=lambda x:x[-1],reverse=True)
#    sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

classList=['yes','yes','yes','no','no']
a=majorityCnt(classList)
print(a)


def createTree(dataSet,labels):

    classList=[example[-1] for example in dataSet]
    if classList.count(classList[0])==len(classList):
        return classList[0]
    if len(dataSet[0])==1:
        return majorityCnt(classList)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   

    bestFeat =chooseBestFeatureToSplit(dataSet)
    bestFeatLabel=labels[bestFeat]

    myTree={bestFeatLabel:{}}
    del(labels[bestFeat])
    featValues=[example[bestFeat] for example in dataSet]
    uniqueVals=set(featValues)

    for value in uniqueVals:
        subLabels=labels[:]
        myTree[bestFeatLabel][value]=createTree(splitDataSet(dataSet,bestFeat,value),subLabels)
    return myTree
myTree=createTree(myDat,labels)
print(myTree)

import matplotlib.pyplot as plt
decisionNode=dict(boxstyle='sawtooth',function='0.8')
leafNode=dict(boxstyle='sawtooth',function='0.8')
arrow_args=dict(arrowstyle='<-')

#def plotNode(nodeTxt, centerPt, parentPt, nodeType):
#    createPlot.ax1.annotate(nodeTxt, xy=parentPt,  xycoords='axes fraction',
#             xytext=centerPt, textcoords='axes fraction',
#             va="center", ha="center", bbox=nodeType, arrowprops=arrow_args )

#def createPlot():
#    fig=plt.figure(1,facecolor='white')
#    fig.clf()
#    createPlot.ax1=plt.subplot(111,frameon=False)
#    plotNode('决策节点',(0.5,0.1),(0.1,0.5),decisionNode)
#    plotNode('叶节点',(0.8,0.1),(0.3,0.8),leafNode)
#    plt.show()

def getNumLeafs(myTree):
    numLeafs=0
    firstStr=list(myTree.keys())[0]
    secondDict=myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            numLeafs+=getNumLeafs(secondDict[key])
        else:
            numLeafs+=1
    return numLeafs

#num=getNumLeafs(myTree)
          
def getTreeDepth(myTree):

    maxDepth=0

    firstStr=list(myTree.keys())[0]
    secondDict=myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            thisDepth=1+getTreeDepth(secondDict[key])
        else:
            thisDepth=1
        if thisDepth>maxDepth: maxDepth=thisDepth
    return maxDepth

print(getTreeDepth(myTree))

def classify(inputTree,featLabels,testVec):
    firstStr=list(inputTree.keys())[0]
    secondDict=inputTree[firstStr]
    featIndex=featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex]==key:
            if type(secondDict[key]).__name__=='dict':
                classLabel=classify(secondDict[key],featLabels,testVec)
            else:
                classLabel=secondDict[key]
    return classLabel

myDat,labels=createDataSet()
print(classify(myTree,labels,[1,1]))

def storeTree(inputTree,filename):
    import pickle
    fw=open(filename,'wb')
    pickle.dump(inputTree,fw)
    fw.close()

def grabTree(filename):
    import pickle
    fr=open(filename,'rb')
    return pickle.load(fr)

storeTree(myTree,'classifierStorage.txt')
print(grabTree('classifierStorage.txt'))

fr=open('lenses.txt')
lenses=[inst.strip().split('\t')  for inst in fr.readlines()]
lensesLabels=['age','prescript','astigmatic','tearRate']
lensesTree=createTree(lenses,lensesLabels)
print(lensesTree)




