# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

iris=load_iris()
df=pd.DataFrame(iris.data,columns=iris.feature_names)
df['label']=iris.target #添加一列label
df.columns=['speal length','speal width','petal length','petal width','label']
df.label.value_counts()
plt.scatter(df[:50]['speal length'],df[0:50]['speal width'],label='-1')
plt.scatter(df[50:100]['speal length'],df[50:100]['speal width'],label='1')
plt.xlabel('speal length')
plt.ylabel('speal width')
plt.show()

data=np.array(df.iloc[:100,[0,1,-1]])#取第列，第二列，最后一列前100个数据
X,y=data[:,:-1],data[:,-1]#100*2,100*1

loc=np.where(y==0)[0]
y[loc]=-1
#print(y[:10])
w=np.random.randn(3,1)
X_train=np.hstack((np.ones((X.shape[0],1)), X))
#原始感知机
def fit(X_train,y_train,w,rate=1.0):
    fault=True
    i=0
    while fault:
        yhat=np.dot(X_train,w)
        y_predict=np.ones_like(y_train)
        loc_n=np.where(yhat<0)[0]
        y_predict[loc_n]=-1
        num_fault=len(np.where(y_train!=y_predict)[0])
#        print(i,num_fault)
        if num_fault==0:
            fault=False
        else:
            t=np.where(y_train!=y_predict)[0][0]
            w+=rate*y_train[t]*X_train[t,:].reshape((3,1))
            i+=1
    return w
w=fit(X_train,y,w,rate=1)
print(w)
x_points=np.linspace(4, 7, 10)
y_points=-(w[1]*x_points+w[0])/w[2]
plt.plot(x_points,y_points)
plt.plot(data[:50, 0], data[:50, 1], 'bo', color='blue', label='-1')
plt.plot(data[50:100, 0], data[50:100, 1], 'bo', color='orange', label='1')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend(loc = 'upper left')
plt.show()

#对偶感知机
def dual_fit(X_train,y_train,rate=1.0):
    fault=True
    num=0
    w=np.zeros((2,1))
    a=np.zeros((y_train.shape))
    b=0
    Gram=np.dot(X_train,X_train.T)
    while fault:
        yhat=np.dot(Gram,a*y_train)+b
        y_predict=np.ones_like(y_train)
        loc_n=np.where(yhat<0)[0]
        y_predict[loc_n]=-1
        num_fault=len(np.where(y_train!=y_predict)[0])
#        print(num,num_fault)
        if num_fault==0:
            fault=False
        else:
            t=np.where(y_train!=y_predict)[0][0]
            a[t]+=1*rate
            b+=y_train[t]*rate
            num+=1
    for i in range(y_train.shape[0]):
        w[0]+=a[i]*X_train[i][0]*y_train[i]
        w[1]+=a[i]*X_train[i][1]*y_train[i]
    return w,b
w_dual,b=dual_fit(X,y)
print(w_dual,b)
x_points=np.linspace(4, 7, 10)
y_points=-(w_dual[0]*x_points+b)/w_dual[1]
plt.plot(x_points,y_points)
plt.plot(X[:50,0],X[:50,1],'bo',color='blue',label='-1')
plt.plot(X[50:100, 0], X[50:100, 1], 'bo', color='orange', label='1')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend(loc = 'upper left')
plt.show()