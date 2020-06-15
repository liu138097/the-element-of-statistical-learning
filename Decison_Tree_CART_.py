import numpy as np
from numpy import inf
def loadDataSet(filename):
    dataMat=[]
    fr=open(filename)
    for line in fr.readlines():
        curLine=line.strip().split('\t')
        fltLine=list(map(float,curLine))
        dataMat.append(fltLine)
    return dataMat

def binSplitDataSet(dataSet,feature,value):
    mat0=dataSet[np.nonzero(dataSet[:,feature]>value)[0],:]
    mat1=dataSet[np.nonzero(dataSet[:,feature]<=value)[0],:]
    return mat0,mat1

def regLeaf(dataSet):
    return np.mean(dataSet[:,-1])

def regErr(dataSet):
    return np.var(dataSet[:,-1])*np.shape(dataSet)[0]

def chooseBestSplit(dataSet,leafType=regLeaf,errType=regErr,ops=(1,4)):
    tolS=ops[0];tolN=ops[1]
    if len(set(dataSet[:,-1].T.tolist()[0]))==1:
        return None,leafType(dataSet)
    m,n=np.shape(dataSet)
    bestS=inf;bestIndex=0;bestValue=0
    S=errType(dataSet)
    for featIndex in range(n-1):
        for splitVal in set(dataSet[:,featIndex].T.A.tolist()[0]):
            mat0,mat1=binSplitDataSet(dataSet,featIndex,splitVal)
            if (np.shape(mat0)[0]<tolN) or (np.shape(mat1)[0]<tolN): continue
            newS=errType(mat0)+errType(mat1)
            if newS<bestS:
                bestS=newS
                bestIndex=featIndex
                bestValue=splitVal
    if (S-bestS)<tolS:
        return None,leafType(dataSet)
    mat0,mat1=binSplitDataSet(dataSet,bestIndex,bestValue)
    if (np.shape(mat0)[0]<tolN) or (np.shape(mat1)[0]<tolN):
        return None,leafType(dataSet)
    return bestIndex,bestValue

def createTree(dataSet,leafType=regLeaf,errType=regErr,ops=(1,4)):
    feat,val=chooseBestSplit(dataSet,leafType,errType,ops)
    if feat==None: return val
    retTree={}
    retTree['spInd']=feat
    retTree['spVal']=val
    lSet,rSet=binSplitDataSet(dataSet,feat,val)
    retTree['left']=createTree(lSet,leafType,errType,ops)
    retTree['right']=createTree(rSet,leafType,errType,ops)
    return retTree
 
myDat=loadDataSet('ex00.txt')
myMat=np.mat(myDat)
myTree=createTree(myMat)
print(myTree)

myDat1=loadDataSet('ex0.txt')
myMat1=np.mat(myDat1)
myTree1=createTree(myMat1)
print(myTree1)

myDat2=loadDataSet('ex2.txt')
myMat2=np.mat(myDat2)
myTree2=createTree(myMat2,ops=(10000,1))
print(myTree2)

def isTree(obj):
    return (type(obj).__name__=='dict')

def getMean(tree):
    if isTree(tree['right']): tree['right']=getMean(tree['right'])
    if isTree(tree['left']): tree['left']=getMean(tree['left'])
    return (tree['left']+tree['right'])/2.0

#print(getMean(myTree))
#print(np.mean(myMat,axis=0))

def prune(tree,testData):
    if np.shape(testData)[0]==0 :return getMean(tree)
    if (isTree(tree['right']) or isTree(tree['left'])) :
         lSet,rSet=binSplitDataSet(testData,tree['spInd'],tree['spVal'])
    if isTree(tree['left']):tree['left']=prune(tree['left'],lSet)
    if isTree(tree['right']):tree['right']=prune(tree['right'],lSet)
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet,rSet=binSplitDataSet(testData,tree['spInd'],tree['spVal'])
        errorNomerge=sum(np.power(lSet[:,-1]-tree['left'],2))+sum(np.power(rSet[:,-1]-tree['right'],2))
        treeMean=(tree['left']+tree['right'])/2.0
        errorMerge=sum(np.power(testData[:,-1]-treeMean,2))
        if errorMerge<=errorNomerge:
            print('merging')
            return treeMean
        else:
            return tree
    else:
        return tree
#myDatTest2=loadDataSet('ex2test.txt')
#myMatTest2=np.mat(myDatTest2)
#prune_tree2=prune(myTree2,myMatTest2)
#print(prune_tree2)


def linearSolve(dataSet):
    m,n=np.shape(dataSet)
    X=np.mat(np.ones((m,n)));Y=np.mat(np.ones((m,1)))
    X[:,1:n]=dataSet[:,0:n-1];Y=dataSet[:,-1]
    xTx=X.T*X
    if np.linalg.det(xTx)==0.0:
        raise NameError('This matrix is singular,cannot do inverse,try increasing the second value of ops')
    ws=xTx.I*(X.T*Y)
    return ws,X,Y

def modelLeaf(dataSet):
    ws,X,Y=linearSolve(dataSet)
    return ws

def modelErr(dataSet):
    ws,X,Y=linearSolve(dataSet)
    yHat=X*ws
    return sum(np.power(Y-yHat,2))

myDatap=loadDataSet('exp2.txt')
myMatp=np.mat(myDatap)
tree2=createTree(myMatp,modelLeaf,modelErr,ops=(1,10))
print(tree2)



def regTreeEval(model,inData):
    return float(model)

def modelTreeEval(model,inData):
    n=np.shape(inData)[1]
    X=np.mat(np.ones((1,n+1)))
    X[:,1:n+1]=inData
    return float(X*model)

def treeForeCast(tree,inData,modelEval=regTreeEval):
    if not isTree(tree): return regTreeEval(tree,inData)
    if inData[tree['spInd']]>tree['spVal']:
        if isTree(tree['left']):
            return treeForeCast(tree['left'],inData,modelEval)
        else:
            return modelEval(tree['left'],inData)
    else:
        if isTree(tree['right']):
            return treeForeCast(tree['right'],inData,modelEval)
        else:
            return modelEval(tree['right'],inData)

def createForeCast(tree,testData,modelEval=regTreeEval):
    m=len(testData)
    yHat=np.mat(np.zeros((m,1)))
    for i in range(m):
        yHat[i]=treeForeCast(tree,np.mat(testData[i]),modelEval)
    return yHat

trainMat=np.mat(loadDataSet('bikeSpeedVsIq_train.txt'))
testMat=np.mat(loadDataSet('bikeSpeedVsIq_test.txt'))
mytree=createTree(trainMat,ops=(1,20))
yHat=createForeCast(mytree,testMat[:,0])
#print(yHat)
accurate=np.corrcoef(yHat,testMat[:,1],rowvar=0)[0,1]
print(accurate)

mytree1=createTree(trainMat,modelLeaf,modelErr,ops=(1,20))
yHat1=createForeCast(mytree1,testMat[:,0],modelTreeEval)
#print(yHat1)
accurate1=np.corrcoef(yHat1,testMat[:,1],rowvar=0)[0,1]
print(accurate1)

ws,X,Y=linearSolve(trainMat)
m,n=np.shape(testMat)
yHat2=np.mat(np.zeros((m,1)))
for i in range(m):
    yHat2[i]=testMat[i,0]*ws[1,0]+ws[0,0]
#print(yHat2)
accurate2=np.corrcoef(yHat2,testMat[:,1],rowvar=0)[0,1]
print(accurate2)