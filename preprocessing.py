import numpy as np
from feature import AbstractFeature
from scipy.misc import imresize

class Resize(AbstractFeature):
    def __init__(self, size):
        AbstractFeature.__init__(self)
        self._size = size
        
    def compute(self,X,y):
        Xp = []
        for xi in X:
            Xp.append(self.extract(xi))
        return Xp
        
    def extract(self,X):
        return imresize(X, self._size)
    
    def __repr__(self):
        return "Resize (size=%s)" % (self._size,)
