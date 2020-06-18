import numpy as np
import matplotlib.pyplot as plt
import random
def loadDataSet(filename):
    dataMat=[];labelMat=[]
    fr=open(filename)
    for line in fr.readlines():
        currentLine=line.strip().split('\t')
        dataMat.append([1.0,float(currentLine[0]),float(currentLine[1])])
        labelMat.append(int(currentLine[2]))
    return dataMat,labelMat

def loadDataSet_horse(filename):
    trainingSet=[];trainingLabels=[]
    fr=open(filename)
    for line in fr.readlines():
        currentLine=line.strip().split('\t')
        lineArr=[]
        lineArr.append(1.0)
        for i in range(21):
            lineArr.append(float(currentLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currentLine[21]))
    return trainingSet,trainingLabels

def sigmoid(inX):
    return 1.0/(1+np.exp(-inX))

def gradAscent(dataMatIn,classLabels,iterations=500):
    dataMatrix=np.mat(dataMatIn)
    labelMat=np.mat(classLabels).transpose()
    alpha=0.001
    m,n=np.shape(dataMatrix)
    weights=np.ones((n,1))
    for i in range(iterations):
        h=sigmoid(dataMatrix*weights)
        error=labelMat-h
        weights=weights+alpha*dataMatrix.transpose()*error
    return np.array(weights)

dataArr,labelMat=loadDataSet('testSet.txt')
#print(dataArr[:5])
w=gradAscent(dataArr,labelMat)
#print(grad)

def plotBestFit(data,label,weights):
    dataArr=np.array(data)
    labelArr=np.array(label)
    label1=np.where(labelArr==1)

    plt.figure()
    plt.scatter(dataArr[label1,1],dataArr[label1,2],marker='o',color='r',label='True')

    label0=np.where(labelArr==0)
    plt.scatter(dataArr[label0,1],dataArr[label0,2],marker='x',color='b',label='False')

    x=np.arange(-3,3,0.1)
    y=-(weights[0]+weights[1]*x)/weights[2]
    plt.plot(x,y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.legend()
    plt.show()

plotBestFit(dataArr,labelMat,w)

def stocGradAscent(data,classLabels):
    dataArr=np.array(data)
    m,n=np.shape(dataArr)
    alpha=0.01
    weights=np.ones(n)
    for i in range(m):
        h=sigmoid(sum(dataArr[i]*weights))
        error=classLabels[i]-h
        weights=weights+alpha*dataArr[i]*error
    return weights

w_stoc=stocGradAscent(dataArr,labelMat)
plotBestFit(dataArr,labelMat,w_stoc)

def stocGradAscent1(data,classLabels,iterations=150):
    dataArr=np.array(data)
    m,n=np.shape(dataArr)
    weights=np.ones(n)
    for j in range(iterations):
        dataIndex=list(range(m))
        for i in range(m):
            alpha=4/(1.0+j+i)+0.01
            randIndex=int(random.uniform(0,len(dataIndex)))
            h=sigmoid(sum(dataArr[randIndex]*weights))
            error=classLabels[randIndex]-h
            weights=weights+alpha*dataArr[randIndex]*error
            del(dataIndex[randIndex])
    return weights

w_stoc1=stocGradAscent1(dataArr,labelMat)
plotBestFit(dataArr,labelMat,w_stoc1)
print(w_stoc1)


def classifyVector(inX,weights):
    prob=sigmoid(sum(inX*weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0

def colicTest():
    trainingSet,trainingLabels=loadDataSet_horse('horseColicTraining.txt')
    trainingWeights=stocGradAscent1(trainingSet,trainingLabels,iterations=300)

    testSet,testLabels=loadDataSet_horse('horseColicTest.txt')
#    one=np.ones((len(testSet),1))
#    testSet=np.hstack((one,testSet))
    testLabelsHat=np.mat(testSet)*np.mat(trainingWeights).T
    testLabelsHat=sigmoid(testLabelsHat)

    testHat=np.ones(np.shape(testLabelsHat)[0])
    label0=np.where(testLabelsHat<=0.5)[0]
    testHat[label0]=0

    accurate=sum(testHat==testLabels)
    print('the accurate of this test is %f'%(accurate/len(testSet)))
    return accurate/len(testSet)

def multiTest():
    numTest=10;accurateSum=0.0
    for k in range(numTest):
        accurateSum+=colicTest()
    print('after %d iterations the average accurate is ; %f'%(numTest,accurateSum/float(numTest)))

multiTest()

