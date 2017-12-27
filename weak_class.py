#encoding=utf-8
import numpy as np
class WeakClass:
    '''
    X:N*M,N is dim number ,M is sample number,
    Y:M,array
    '''
    def train(self,X,Y,W,iters = 100):
        self.point = 0
        self.label = 0
        self.dims = X.shape[0]
        last_error = 100000.0
        last_point = 0
        last_label = 0
        last_dim = 0
        for n in range(0,self.dims):
            for label in [-1,1]:
                error,point = self.min_error(X,Y,n,label,W,iters)
                if  error < last_error:
                    last_error = error
                    last_point = point
                    last_label = label
                    last_dim = n
        self.point = last_point
        self.direction = last_label
        self.dim = last_dim
        return last_error
    def min_error(self,X,Y,i,label,W,iters):
        min_x = np.min(X[i,:])
        max_x = np.max(X[i,:])
        step = ( max_x - min_x )*1.0/ iters
        last_error = 10000000000.0
        last_point = 0
        for point in np.arange(min_x,max_x,step):
            gt = np.ones((np.array(X).shape[1],1))
            gt[X[i,:]*label<point*label]=-1
            error = np.sum((gt.T != Y) * W )
            if error < last_error:
                last_error = error
                last_point = point
        return last_error,last_point
    def pred(self,test_X):
         gt = np.ones((np.array(test_X).shape[1],1))
         gt[test_X[self.dim,:]*self.direction<self.point*self.direction]=-1
         return gt
    
        
