import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from collections import Counter

iris=load_iris()
df=pd.DataFrame(iris.data,columns=iris.feature_names)
df['label']=iris.target
df.columns=['speal length','speal width','petal length','petal width','label']
#print(df.head())

plt.scatter(df[:50]['speal length'],df[:50]['speal width'],label='0')
plt.scatter(df[50:100]['speal length'],df[50:100]['speal width'],label='1')
plt.xlabel('speal length')
plt.ylabel('speal width')
plt.legend(loc='best')
plt.show()

data=np.array(df.iloc[:100,[0,1,-1]])
print(data.shape)

X=data[:,:-1]
y=data[:,-1]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

#暴力法
class KNN:
    def __init__(self,X_train,y_train,n_neighbours=3,p=2):
        self.n=n_neighbours
        self.p=p
        self.X_train=X_train
        self.y_train=y_train
    def predict(self,X):
        knn_list=[]
        for i in range(self.n):
            dist=np.linalg.norm(X-self.X_train[i],ord=self.p)
            knn_list.append((dist,self.y_train[i]))
        for i in range(self.n,len(self.X_train)):
            max_index=knn_list.index(max(knn_list,key=lambda x:x[0]))
            dist=np.linalg.norm(X-self.X_train[i],ord=self.p)
            if knn_list[max_index][0]>dist:
                knn_list[max_index]=(dist,self.y_train[i])
        knn = [k[-1] for k in knn_list]#返回y_train
        count_pairs=Counter(knn)
        max_count=sorted(count_pairs.items(),key=lambda x:x[-1])[-1][0] #sorted返回list
        return max_count
    def score(self,X_test,y_test):
        right_count = 0
        n = 10
        yhat=[]
        for X, y in zip(X_test, y_test):
            label = self.predict(X)
            yhat.append(label)
            if label == y:
                right_count += 1
        return right_count / len(X_test),yhat
clf = KNN(X_train, y_train)
accurate,y_testhat=clf.score(X_test, y_test)
#print(accurate)
#print(y_testhat)

test_point = [6.0, 3.0]
print('Test Point: {}'.format(clf.predict(test_point)))

plt.scatter(df[:50]['speal length'],df[:50]['speal width'],label='0')
plt.scatter(df[50:100]['speal length'], df[50:100]['speal width'], label='1')
plt.plot(test_point[0], test_point[1], 'bo', label='test_point')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend(loc='best')
plt.show()

#kd树法(k近似算法）
class Node(object):
    def __init__(self,data,left=None,right=None):
        self.data=data #节点样本
        self.left=left
        self.right=right

class KdTree(object):
    def __init__(self):
        self.kdtree=None
    def create(self,dataset,depth):
        if not dataset:
            return None
        m,n=np.shape(dataset)
        axis=depth%n
        median=m//2
        sortedDataset=sorted(dataset,key=lambda x:x[axis])

        leftDataset=sortedDataset[:median]
        rightDataset=sortedDataset[median+1:]
        print(leftDataset)
        print(rightDataset)

        node=Node(sortedDataset[median])
        node.left=self.create(leftDataset,depth+1)
        node.right=self.create(rightDataset,depth+1)
        return node

    #检查kd数能否正常排序
    def preOrder(self, node):
        if node != None:
            print("tttt->%s" % node.data)
            self.preOrder(node.left)
            self.preOrder(node.right)
    def search(self, tree, x, k=1):
        self.nearestPoint = []    #保存最近的点
        self.nearestValue = []   #保存最近的值
        print(self.nearestPoint)
        def travel(node, depth = 0):    #递归搜索
            if node != None:    #递归终止条件
                n = len(x)  #特征数
                axis = depth % n    #计算轴
                if x[axis] < node.data[axis]:   #如果数据小于结点，则往左结点找
                    travel(node.left, depth+1)
                else:
                    travel(node.right,depth+1)

                #以下是递归完毕后，往父结点方向回朔
                distNodeAndX = self.dist(x, node.data)  #目标和节点的距离判断
                if (len(self.nearestPoint)< k): #确定当前点，更新最近的点和最近的值
                    self.nearestPoint.append(node.data)
                    self.nearestValue.append(distNodeAndX)
                elif (max(self.nearestValue) > distNodeAndX):
                    max_index=self.nearestValue.index(max(self.nearestValue))
                    self.nearestPoint[max_index] = node.data
                    self.nearestValue[max_index] = distNodeAndX

                print(node.data, depth, self.nearestValue, node.data[axis], x[axis])
                if (abs(x[axis] - node.data[axis]) < max(self.nearestValue)):  #确定是否需要去子节点的区域去找（圆的判断）
                    if x[axis] < node.data[axis]:
                        travel(node.right, depth+1)
                    else:
                        travel(node.left,  depth+1)
        travel(tree)
        return self.nearestPoint

    def dist(self, x1, x2): #欧式距离的计算
        return ((np.array(x1) - np.array(x2)) ** 2).sum() ** 0.5


dataSet = [[2, 3],
           [5, 4],
           [9, 6],
           [4, 7],
           [8, 1],
           [7, 2]]
x=[4,4]
kdtree = KdTree()
tree = kdtree.create(dataSet, 0)
kdtree.preOrder(tree)
print(kdtree.search(tree, x ,k=3))