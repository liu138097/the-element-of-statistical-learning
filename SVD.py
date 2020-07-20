import numpy as np

def rebuildMatrix(U,sigma,V):
    a=np.dot(U,sigma)
    a=np.dot(a,np.transpose(V))
    return a

def sortByEigenValue(EigenValues,EigenVectors):
    index=np.argsort(-1*EigenValues)
    EigenValues=EigenValues[index]
    EigenVectors=EigenVectors[:,index]
    return EigenValues,EigenVectors

def SVD(matrixA,NumOfLeft=None):
    matrixAT_matrixA=np.dot(np.transpose(matrixA),matrixA)
    lambda_V,X_V=np.linalg.eig(matrixAT_matrixA) #np.linalg.eig() 计算目标矩阵奇异值和右奇异值
    lambda_V,X_V=sortByEigenValue(lambda_V,X_V)
    sigmas=lambda_V
    sigmas=list(map(lambda x:np.sqrt(x) if x>0 else 0,sigmas))
    sigmas=np.array(sigmas)
    sigmasMatrix=np.diag(sigmas)
    if not NumOfLeft:
        rankOfSigmasMatrix=len(list(filter(lambda x:x>0,sigmas)))
    else:
        rankOfSigmasMatrix=NumOfLeft
    X_U=np.zeros((matrixA.shape[0],rankOfSigmasMatrix))
    for i in range(rankOfSigmasMatrix):
        X_U[:,i]=np.dot(matrixA,X_V[:,i])/sigmas[i]
    X_V=X_V[:,0:rankOfSigmasMatrix]
    sigmasMatrix=sigmasMatrix[0:rankOfSigmasMatrix,0:rankOfSigmasMatrix]
    return X_U,sigmasMatrix,X_V

A = np.array([[1, 1, 1, 2, 2], [0, 0, 0, 3, 3], [0, 0, 0, 1, 1], [1, 1, 1, 0, 0],
              [2, 2, 2, 0, 0], [5, 5, 5, 0, 0], [1, 1, 1, 0, 0]])
X_U, sigmasMatrix, X_V = SVD(A,NumOfLeft=3)
print("U: ",X_U)
print("sigmas: ",sigmasMatrix)
print("X_V: ",X_V)

A_rebuild=rebuildMatrix(X_U, sigmasMatrix, X_V)
print("A_rebuild: ",A_rebuild)