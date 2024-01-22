from numpy.random import randint
from sklearn.base import BaseEstimator, TransformerMixin
import joblib
import numpy as np
import pandas as pd


class QKernelTransformer(BaseEstimator, TransformerMixin):
    def __init__(self,puntajeriesgo=np.array([1,3,1,15,1,15,6,3,1,25,1,1,1.5,1,1,1,1,3,1.5,1,15,1]),
      valormaxacep=np.array([[0,200],[0,0.2],[0,60],[0.3,2],[0,250],[0,0],[0,15],[0,5],[0,300],[0,0],[0,1],[0,0.5],[0,0.3],[0,36],[0,0.1],[0,0.07],[0,10],[0,0.1],[6.5,9],[0,250],[0,2],[0,3]])):
      self.puntajeriesgo=puntajeriesgo
      self.valormaxacep=valormaxacep


    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Perform arbitary transformation
        df=pd.DataFrame(X)
        for i in range(0,22):
          df[i+44]=df[[i,i+22]].apply(self.nuevo2,axis=1)

        for i in range(0,22):
          df[i+66]=df[[i,i+22]].apply(self.nuevo,axis=1)

        X0=df.iloc[:,0:22].to_numpy()
        X1=df.iloc[:,22:44].to_numpy()
        X2=df.iloc[:,44:66].to_numpy()
        X3=df.iloc[:,66:88].to_numpy()

        X=np.concatenate([(X0*X2).sum(axis=1).reshape((-1,1)),(X0*X3).sum(axis=1).reshape((-1,1))],axis=1)
        return X


    def nuevo(self,x):
      i1=x.index[0]
      i2=x.index[1]
      if  x[i2]>=self.valormaxacep[i1][0] and x[i2]<=self.valormaxacep[i1][1]:
        return 0
      return self.puntajeriesgo[i1]

    def nuevo2(self,x):
      i1=x.index[0]
      return self.puntajeriesgo[i1]