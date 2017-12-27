#encoding=utf-8
import numpy as np
from weak_class import WeakClass
def sign(X):
    return [  1 if k >= 0 else -1  for k in X  ]
class AdaBoost:
    def __init__(self,X,Y,Weaker=WeakClass):
        self.X = X
        self.Y = np.array(Y)
        self.Weaker = Weaker
    def train(self,weak_num = 4):
        self.weakers = {}
        self.weaker_weights = np.ones(weak_num)
        sample_weights = np.ones(self.X.shape[1])/self.X.shape[1]
        
        cul_pred = np.zeros(self.Y.shape)
        for i in range(0,weak_num):
            self.weakers[i] = self.Weaker()
            error = self.weakers[i].train(self.X,self.Y,sample_weights)
            self.weaker_weights[i] = 1.0/2 * np.log((1-error)/error)
            print error,'classifier:',i,self.weaker_weights[i]
            tmp = self.Y*self.weakers[i].pred(self.X).T
            sample_weights  = sample_weights*np.exp(-self.weaker_weights[i] * tmp)
            sample_weights = sample_weights/np.sum(sample_weights)
            
            cul_pred = cul_pred + self.weakers[i].pred(self.X).flatten(1)*self.weaker_weights[i]
            if 0 == (sign(cul_pred) != self.Y).sum():
                print i+1," classifiers enough !"
                break
        self.num = i+1
    def pred(self,test_X):
        sums = np.zeros((test_X.shape[1],1))
        for i in range(self.num):
            sums = sums + self.weakers[i].pred(test_X)*self.weaker_weights[i]
        return sign(sums)
