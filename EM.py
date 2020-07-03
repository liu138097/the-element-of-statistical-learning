import numpy as np
import math

class EM:
    def __init__(self,prob):
        self.prob_A,self.prob_B,self.prob_C=prob
    def pmf(self,i):
        prob_1=self.prob_A*np.power(self.prob_B,data[i])*np.power((1.0-self.prob_B),(1.0-data[i]))
        prob_2=prob_1+(1.0-self.prob_A)*np.power(self.prob_C,data[i])*np.power((1-self.prob_C),(1-data[i]))
        return float(prob_1/prob_2)
    def fit(self,data,iter=2):
        count=len(data)
        print('init prob:{}, {}, {}'.format(self.prob_A, self.prob_B,self.prob_C))
        for i in range(iter):
            pmf_sum=[self.pmf(k) for k in range(count)]
            prob_A=np.sum(pmf_sum)/count
            prob_B=np.sum([pmf_sum[k]*data[k] for k in range(count)])/np.sum(pmf_sum)
            prob_C=np.sum([(1-pmf_sum[k])*data[k] for k in range(count)])/np.sum([1-pmf_sum[k] for k in range(count)])
            print('{}/{}  pro_a:{:.3f}, pro_b:{:.3f}, pro_c:{:.3f}'.format(i + 1, iter, prob_A, prob_B, prob_C))
            self.prob_A=prob_A
            self.prob_B=prob_B
            self.prob_C=prob_C

data=[1,1,0,1,0,0,1,0,1,1]
#em=EM(prob=[0.5,0.5,0.5])
em=EM(prob=[0.4,0.6,0.7])
f=em.fit(data)


