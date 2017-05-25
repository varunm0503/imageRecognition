import numpy as np

class AbstractDistance(object):
    def __init__(self, name):
        self._name = name
        
    def __call__(self,p,q):
        raise NotImplementedError("Every AbstractDistance must implement the __call__ method.")
        
    @property
    def name(self):
        return self._name

    def __repr__(self):
        return self._name
        
class EuclideanDistance(AbstractDistance):
    def __init__(self):
        AbstractDistance.__init__(self,"EuclideanDistance")

    def __call__(self, p, q):
        p = np.asarray(p).flatten()
        q = np.asarray(q).flatten()
        return np.sqrt(np.sum(np.power((p-q),2)))

class NormalizedCorrelation(AbstractDistance):
    """
        Calculates the NormalizedCorrelation Coefficient for two vectors.
    
        Literature:
            "Multi-scale Local Binary Pattern Histogram for Face Recognition". PhD (2008). Chi Ho Chan, University Of Surrey.
    """
    def __init__(self):
        AbstractDistance.__init__(self,"NormalizedCorrelation")
    
    def __call__(self, p, q):
        p = np.asarray(p).flatten()
        q = np.asarray(q).flatten()
        pmu = p.mean()
        qmu = q.mean()
        pm = p - pmu
        qm = q - qmu
        return 1.0 - (np.dot(pm, qm) / (np.sqrt(np.dot(pm, pm)) * np.sqrt(np.dot(qm, qm)))) 
