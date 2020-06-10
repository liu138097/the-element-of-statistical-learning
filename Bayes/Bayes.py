import numpy as np
import random
def loadDataSet():
    postingList=[['my','dog','has','flea','problems','help','please'],
                 ['maybe','not','take','him','to','dog','park','stupid'],
                 ['my','dalmation','is','so','cute','I','love','him'],
                 ['stop','posting','stupid','worthless','garbage'],
                 ['mr','licks','ate','my','steak','how','to','stop','him'],
                 ['quit','buying','worthless','dog','food','stupid']]
    classVec=[0,1,0,1,0,1]
    return postingList,classVec

def createVocabList(dataset):
    vocabSet=set([])
    for document in dataset:
        vocabSet=vocabSet|set(document)
    return list(vocabSet)

def setOfWords2Vec(vocabList,inputSet):
    returnVec=[0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
             returnVec[vocabList.index(word)]=1
        else:
            print('the word : %s is not in my Vocabulary'%word)
    return returnVec


def trainNB0(trainMatrix,trainCategory):
    numTrainDocs=len(trainMatrix) #文档数目
    numWords=len(trainMatrix[0])
    pAbusive=sum(trainCategory)/float(numTrainDocs) #文档中属于侮辱类的概率，等于1才能算，0是非侮辱类
    p0Num = np.ones(numWords)
    p1Num = np.ones(numWords)
    p0Denom = 2.0
    p1Denom = 2.0
    for i in range(numTrainDocs):#遍历每个文档
        if trainCategory[i]==1: #一旦某个词出现在某个文档中出现（出现为1，不出现为0）
            p1Num+=trainMatrix[i]  #该词数加1
            p1Denom+=sum(trainMatrix[i]) #文档总词数加1
        else: #另一个类别
            p0Num+=trainMatrix[i]
            p0Denom+=sum(trainMatrix[i])
    p1Vec =np.log(p1Num/p1Denom)
    p0Vec =np.log(p0Num/p0Denom)
    return p0Vec, p1Vec, pAbusive  #返回p0Vec，p1Vec都是矩阵，对应每个词在文档总体中出现概率，pAb对应文档属于1的概率


def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
    p1=sum(vec2Classify*p1Vec)+np.log(pClass1)
    p0=sum(vec2Classify*p0Vec)+np.log(1.0-pClass1)
    if p1>p0:
        return 1
    else:
        return 0

def testingNB():
    listOPosts,listClasses=loadDataSet()
    myVocabList=createVocabList(listOPosts)
    trainMat=[]
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList,postinDoc))
    p0V,p1V,pAb=trainNB0(trainMat,listClasses)
    testEntry=['love','my','dalmation']
    thisDoc=np.array(setOfWords2Vec(myVocabList,testEntry))
    print(testEntry,' classified as: ',classifyNB(thisDoc, p0V,p1V,pAb))
    testEntry=['stupid','garbage']
    thisDoc=np.array(setOfWords2Vec(myVocabList,testEntry))
    print(testEntry,' classified as: ',classifyNB(thisDoc, p0V,p1V,pAb))
    return 0

#testingNB()

def bagOfWords2VecMN(vocabList,inputSet):
    returnVec=[0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)]+=1
    return returnVec

def textParse(bigString):
    import re
    listOfTokens=re.split(r'\W+',bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]

def spamTest():
    docList=[];classList=[];fullText=[]
    for i in range(1,26):
        wordList=textParse(open('email/spam/%d.txt'%i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)

        wordList=textParse(open('email/ham/%d.txt'%i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList=createVocabList(docList)
    trainingSet=list(range(50));testSet=[]
    for i in range(10):
        randIndex=int(random.uniform(0.1,len(trainingSet)-0.1))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat=[];trainClasses=[]
    for docIndex in trainingSet:
#        trainMat.append(setOfWords2Vec(vocabList,docList[docIndex]))
        trainMat.append(bagOfWords2VecMN(vocabList,docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam=trainNB0(trainMat,trainClasses)
    errCount=0
    for docIndex in testSet: 
#        wordVector=setOfWords2Vec(vocabList,docList[docIndex])
        wordVector=bagOfWords2VecMN(vocabList,docList[docIndex])
        if classifyNB(np.array( wordVector),p0V,p1V,pSpam)!=classList[docIndex]:
            errCount+=1
    print("the error rate is : ",float(errCount)/len(testSet))
    return float(errCount)/len(testSet)

#交叉验证取平均值
for i in range(10):
    error=[]
    error.append(spamTest())
error_mean=np.sum(error)/10.0
print(error_mean)

import feedparser

#取前60个高频词汇,30个太少，测试集按0.3比例取，也就是12个
#经测验去掉前60个高频词汇，取测试集12个，采用词袋模型，error最低
def calcMostFreq(vocavList,fullText):
    import operator
    freqDict={}
    for token in vocavList:
        freqDict[token]=fullText.count(token)
    sortedFreq=sorted(freqDict.items(),key=operator.itemgetter(1),reverse=True)
    return sortedFreq[:60]

def localWords(feed1,feed0):
    import feedparser
    docList=[];classList=[];fullText=[]
    minLen=min(len(feed1['entries']),len(feed0['entries']))
    for i in range(minLen):
        wordList=textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)

        wordList=textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)

    vocabList=createVocabList(docList)
    top60Words=calcMostFreq(vocabList,fullText)
    for pairW in top60Words:
        if pairW[0] in vocabList:
            vocabList.remove(pairW[0])
    trainingSet=list(range(2*minLen));testSet=[]
    for i in range(12):
        randIndex=int(random.uniform(0.1,len(trainingSet)-0.1))#这里取值不加0.1范围【0,50】，假设trainingSet共50个数据，那么如果随机取到50，就会超出索引界限，因此加上0.1限制一下范围
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat=[];trainClasses=[]
    for docIndex in trainingSet:
#词集模型与词袋模型
#        trainMat.append(setOfWords2Vec(vocabList,docList[docIndex]))
        trainMat.append(bagOfWords2VecMN(vocabList,docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam=trainNB0(trainMat,trainClasses)
    errCount=0
    for docIndex in testSet: 
#        wordVector=setOfWords2Vec(vocabList,docList[docIndex])
        wordVector=bagOfWords2VecMN(vocabList,docList[docIndex])
        if classifyNB(np.array( wordVector),p0V,p1V,pSpam)!=classList[docIndex]:
            errCount+=1
    print("the error rate is : ",float(errCount)/len(testSet))
    return vocabList,p0V,p1V

ny=feedparser.parse('http://www.nasa.gov/rss/dyn/image_of_the_day.rss')
sf=feedparser.parse('http://sports.yahoo.com/nba/teams/hou/rss.xml')
#vocabList,pSF,pNY=localWords(ny,sf)

def getTopWords(ny,sf):
    import operator
    vocabList,p0V,p1V=localWords(ny,sf)
    topNY=[];topSF=[]
    for i in range(len(p0V)):
        if(p0V[i] > -6.0):
            topSF.append((vocabList[i],p0V[i]))
        if(p1V[i] > -6.0):
            topNY.append((vocabList[i],p1V[i]))
    sortedSF=sorted(topSF,key=lambda x:x[-1],reverse=True)
    print("SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**")
    for item in sortedSF[:20]:
        print(item[0])
    sortedNY=sorted(topNY,key=lambda x:x[-1],reverse=True)
    print("NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**")
    for item in sortedNY[:20]:
        print(item[0])
    return 0

getTopWords(ny,sf)
