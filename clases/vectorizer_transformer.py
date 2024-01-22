
from numpy.random import randint
from sklearn.base import BaseEstimator, TransformerMixin
import joblib
import numpy as np
import pandas as pd

class VectorizerTransformer(BaseEstimator, TransformerMixin):

    def __init__(self,parametrosKeys=np.array(['alcalinidad', 'aluminio', 'calcio', 'cloro residual', 'cloruros',
       'coliformes totales', 'color', 'cot', 'dureza', 'escherichia coli',
       'fluoruros', 'fosfatos', 'hierro', 'magnesio', 'manganeso',
       'molibdeno', 'nitratos', 'nitritos', 'ph', 'sulfatos', 'turbiedad',
       'zinc'])):
      self.parametrosKeys=parametrosKeys

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Perform arbitary transformation
        X=X.to_numpy()
        X=self.getDataset([X[:,0],X[:,1]])
        return X
    def vectorizar(self,word):
      return word.replace("[","").replace("]","").split(",")
    def getDataset(self,data):


      parametros=data[0]
      parametrosv=list(map(self.vectorizar,parametros))
      valoresv=list(map(self.vectorizar,data[1]))
      valoresv=[list(map(float,x)) for x in valoresv]

      parametros_final=[]
      valores_final=[]
      for i,parametro in enumerate(parametrosv):
        paramr=np.zeros(22)
        valorr=np.zeros(22)
        for j,p in enumerate(parametro):
          try:
            index=self.parametrosKeys.tolist().index(p)
            paramr[index]=1
            valorr[index]=valoresv[i][j]
          except:
            continue
        parametros_final.append(paramr)
        valores_final.append(valorr)
      parametros_final=np.array(parametros_final)
      valores_final=np.array(valores_final)
      X=np.concatenate((parametros_final,valores_final),axis=1)
      return X