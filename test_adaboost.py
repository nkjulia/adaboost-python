import numpy as np
from adaboost import AdaBoost
X=np.array([i for i in range(0,10)]).reshape(1,10)
Y=[1,1,1,-1,-1,-1,1,1,1,-1]
ada = AdaBoost(X,Y)
ada.train(4)
print ada.pred(X) == np.array(Y)
