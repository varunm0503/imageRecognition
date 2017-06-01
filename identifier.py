import operator as op
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

class AbstractClassifier(object):

    def compute(self,X,y):
        raise NotImplementedError("Every AbstractClassifier must implement the compute method.")
    
    def predict(self,X):
        raise NotImplementedError("Every AbstractClassifier must implement the predict method.")

    def update(self,X,y):
        raise NotImplementedError("This Classifier is cannot be updated.")

class NearestNeighbor(AbstractClassifier):

    def __init__(self, dist_metric=EuclideanDistance(), k=1):
        AbstractClassifier.__init__(self)
        self.k = k
        self.dist_metric = dist_metric
        self.X = []
        self.y = np.array([], dtype=np.int32)

    def update(self, X, y):
        self.X.append(X)
        self.y = np.append(self.y, y)

    def compute(self, X, y):
        self.X = X
        self.y = np.asarray(y)
    
    def predict(self, q):
        distances = []
        for xi in self.X:
            xi = xi.reshape(-1,1)
            d = self.dist_metric(xi, q)
            distances.append(d)
        if len(distances) > len(self.y):
            raise Exception("More distances than classes. Is your distance metric correct?")
        distances = np.asarray(distances)
        idx = np.argsort(distances)
        sorted_y = self.y[idx]
        sorted_distances = distances[idx]
        sorted_y = sorted_y[0:self.k]
        sorted_distances = sorted_distances[0:self.k]
        hist = dict((key,val) for key, val in enumerate(np.bincount(sorted_y)) if val)
        predicted_label = max(iter(hist.items()), key=op.itemgetter(1))[0]
        return [predicted_label, { 'labels' : sorted_y, 'distances' : sorted_distances }]
        
    def __repr__(self):
        return "NearestNeighbor (k=%s, dist_metric=%s)" % (self.k, repr(self.dist_metric))
