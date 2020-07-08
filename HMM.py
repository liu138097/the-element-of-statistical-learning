# -*- coding: utf-8 -*-
import numpy as np
class HiddenMarkov:
    def forward(self,Q,V,A,B,O,PI):
        N=len(Q)
        M=len(O)
        T=M
        alphas=np.zeros((N,M))
        for t in range(T):#第t个状态
            indexO=V.index(O[t])#第t个状态对应的观测
            for i in range(N):#第t个状态的N个选择
                if t==0:
                    alphas[i][t]=PI[i]*B[i][indexO]
                else:
                    alphas[i][t]=np.dot([alpha[t-1] for alpha in alphas],[a[i] for a in A])*B[i][indexO]
        P=np.sum([alpha[T-1] for alpha in alphas])
        print(P)
    def backward(self,Q,V,A,B,O,PI):
        N=len(Q)
        M=len(O)
        T=M
        bletas=np.ones((N,M))
        for t in range(T-2,-1,-1):
            indexO=V.index(O[t+1])
            for i in range(N):
                bletas[i][t]=np.dot(np.multiply(A[i],[b[indexO] for b in B]),[bleta[t+1] for bleta in bletas])
        index0=V.index(O[0])
        P=np.dot(np.multiply(PI,[b[index0] for b in B]),[bleta[0] for bleta in bletas])
        print(P)
    def viterbi(self, Q, V, A, B, O, PI):
        N = len(Q)  #可能存在的状态数量
        M = len(O)  # 观测序列的大小
        deltas = np.zeros((N, M))
        psis = np.zeros((N, M))
        I = np.zeros((1, M))
        for t in range(M):
            indexO=V.index(O[t])
            for i in range(N):
                if t==0:
                    deltas[i][t]=PI[i]*B[i][indexO]
                    psis[i][t]=0
                else:
                    deltas[i][t]=np.max(np.multiply([delta[t-1] for delta in deltas],[a[i] for a in A])*B[i][indexO])
                    psis[i][t]=np.argmax(np.multiply([delta[t-1] for delta in deltas],[a[i] for a in A]))+1
        I[0][M-1]=np.argmax([delta[M-1] for delta in deltas])+1
        for t in range(M-2,-1,-1):
            I[0][t]=psis[int(I[0][t+1])-1][t+1]
        print(I)

#习题10.1
Q = [1, 2, 3]
V = ['红', '白']
A = [[0.5, 0.2, 0.3], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]]
B = [[0.5, 0.5], [0.4, 0.6], [0.7, 0.3]]
# O = ['红', '白', '红', '红', '白', '红', '白', '白']
O = ['红', '白', '红', '白']    #习题10.1的例子
PI = [0.2, 0.4, 0.4]

HMM = HiddenMarkov()
# HMM.forward(Q, V, A, B, O, PI)
# HMM.backward(Q, V, A, B, O, PI)
HMM.viterbi(Q, V, A, B, O, PI)

Q = [1, 2, 3]
V = ['红', '白']
A = [[0.5, 0.2, 0.3], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]]
B = [[0.5, 0.5], [0.4, 0.6], [0.7, 0.3]]
O = ['红', '白', '红', '红', '白', '红', '白', '白']
PI = [0.2, 0.3, 0.5]
HMM.forward(Q, V, A, B, O, PI)
HMM.backward(Q, V, A, B, O, PI)