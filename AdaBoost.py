import numpy as np
import matplotlib.pyplot as plt
import math
import copy
def loadSimpleData():
    dataMat=np.mat([[1.,2.1],
                    [2.,1.1],
                    [1.3,1.],
                    [1.,1.],
                    [2.,1.]])
    classLabels=[1.0,1.0,-1.0,-1.0,1.0]
    return dataMat,classLabels

#dataMat,classLabels=loadSimpleData()
#print(dataMat,classLabels)
#true=np.where(np.array(classLabels)>0.0)
#false=np.where(np.array(classLabels)<0.0)
#plt.scatter(np.array(dataMat[true,0]),np.array(dataMat[true,1]),c='r',marker='o')
#plt.scatter(np.array(dataMat[false,0]),np.array(dataMat[false,1]),c='r',marker='x')
#plt.show()


def stumClassify(dataMatrix,dimen,threshVal,threshIneq):
    retArray=np.ones((np.shape(dataMatrix)[0],1))
    if threshIneq=='lt':
        retArray[dataMatrix[:,dimen]<=threshVal]=-1.0
    else:
        retArray[dataMatrix[:,dimen]>threshVal]=-1.0
    return retArray

def buildStump(dataArr,classLabels,D):
    dataMatrix=np.mat(dataArr);labelMat=np.mat(classLabels).T
    m,n=np.shape(dataMatrix)
    numSteps=10.0;bestStump={};bestClassEst=np.mat(np.zeros((m,1)))
    minError=math.inf
    for i in range(n):
        rangeMin=np.min(dataMatrix[:,i]);rangeMax=np.max(dataMatrix[:,i])
        stepSize=(rangeMax-rangeMin)/numSteps
        for j in range(-1,int(numSteps)+1):
            for inequal in ['lt','gt']:
                threshVal=rangeMin+stepSize*float(j)
                predictedVals=stumClassify(dataMatrix,i,threshVal,inequal)
                errArr=np.mat(np.ones((m,1)))
                errArr[predictedVals==labelMat]=0
                weightedError=D.T*errArr
                print("split: dim %d,thresh %.2f,thresh iequal: %s,the weight error is %.3f"\
                    %(i,threshVal,inequal,weightedError))
                if weightedError<minError:
                    minError=weightedError
                    bestClassEst=copy.copy( predictedVals)
                    bestStump['dim']=i
                    bestStump['thresh']=threshVal
                    bestStump['ineq']=inequal
    return bestStump,minError,bestClassEst

#D=np.mat(np.ones((5,1))/5)
#stump,error,classlf=buildStump(dataMat,classLabels,D)
#print(stump,error,classlf)


def adaBoostTrainDS(dataArr,classLabels,numIt=40):
    weakClassArr=[]
    m=np.shape(dataArr)[0]
    D=np.mat(np.ones((m,1))/m)
    aggClassEst=np.mat(np.zeros((m,1)))
    for i in range(numIt):
        bestStump,error,classEst=buildStump(dataArr,classLabels,D)
        print("D: ",D.T)
        alpha = float(0.5*np.log((1.0-error)/max(error,1e-16)))
        bestStump['alpha']=alpha
        weakClassArr.append(bestStump)
        print("classEst: ",classEst.T)
        expon=np.multiply(-1*alpha*np.mat(classLabels).T,classEst)
        D=np.multiply(D,np.exp(expon))
        D=D/np.sum(D)
        aggClassEst+=alpha*classEst
        print("aggClassEst: ",aggClassEst.T)
        aggErrors=np.multiply(np.sign(aggClassEst)!=np.mat(classLabels).T,np.ones((m,1)))
        errorRate=np.sum(aggErrors)/m
        print("total error: ",errorRate,"\n")
        if errorRate==0.0:break
    return weakClassArr
#classifierArray=adaBoostTrainDS(dataMat,classLabels,9)
#print(classifierArray)

def adaClassify(datToClass,classifierArr):
    dataMatrix=np.mat(datToClass)
    m=np.shape(dataMatrix)[0]
    aggClassEst=np.mat(np.zeros((m,1)))
    for i in range(len(classifierArr)):
        classEst=stumClassify(dataMatrix,classifierArr[i]['dim'],\
            classifierArr[i]['thresh'],classifierArr[i]['ineq'])
        aggClassEst+=classifierArr[i]['alpha']*classEst
        print(aggClassEst)
    return np.sign(aggClassEst)

#dataMat,classLabels=loadSimpleData()
#classifierArr=adaBoostTrainDS(dataMat,classLabels,numIt=30)
#yHat=adaClassify([0,0],classifierArr)
#yHat=adaClassify([[5,5],[0,0]],classifierArr)
#print(yHat)

def loadDataSet(filename):
    numFeat=len(open(filename).readline().split('\t'))
    dataMat=[];labelMat=[]
    fr=open(filename)
    for line in fr.readlines():
        lineArr=[]
        curLine=line.strip().split('\t')
        for i in range(numFeat-1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat
dataArr,labelArr=loadDataSet('horseColicTraining2.txt')
classifierArr=adaBoostTrainDS(dataArr,labelArr,10)

testArr,testLabelArr=loadDataSet('horseColicTest2.txt')
prediction10=adaClassify(testArr,classifierArr)
errArr=np.mat(np.ones((67,1)))
errorRate=errArr[prediction10!=np.mat(testLabelArr).T].sum()/np.shape(testArr)[0]
print("error rate:",errorRate)

def plotROC(predStrengths,classLabels):
    cur=(1.0,1.0)
    ySum=0.0
    numPosClas=sum(np.array(classLabels)==1.0)
    yStep=1/float(numPosClas)
    xStep=1/float(len(classLabels)-numPosClas)
    sortedIndicies=np.argsort(predStrengths,axis=0)

    sortedIndicies=np.array(sortedIndicies).flatten()

    fig=plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    for index in sortedIndicies.tolist():
        if classLabels[index] == 1.0:
            delX = 0; delY = yStep;
        else:
            delX = xStep; delY = 0;
            ySum += cur[1]
        ax.plot([cur[0],cur[0]-delX],[cur[1],cur[1]-delY], c='b')
        cur = (cur[0]-delX,cur[1]-delY)
    ax.plot([0,1],[0,1],'b--')
    plt.xlabel('False positive rate'); plt.ylabel('True positive rate')
    plt.title('ROC curve for AdaBoost horse colic detection system')
    ax.axis([0,1,0,1])
    plt.show()
    print ("the Area Under the Curve is: ",ySum*xStep)
plotROC(prediction10,testLabelArr)