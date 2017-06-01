import logging
import math as math
import random as random
from builtins import range

import numpy as np

from method import PredictableModel

def shuffle(X, y):
    idx = np.argsort([random.random() for i in range(len(y))])
    y = np.asarray(y)
    X = [X[i] for i in idx]
    y = y[idx]
    return (X, y)
    
def slice_2d(X,rows,cols):
    return [X[i][j] for j in cols for i in rows]

def precision(true_positives, false_positives):
    return accuracy(true_positives, 0, false_positives, 0)
    
def accuracy(true_positives, true_negatives, false_positives, false_negatives, description=None):
    true_positives = float(true_positives)
    true_negatives = float(true_negatives)
    false_positives = float(false_positives)
    false_negatives = float(false_negatives)
    if (true_positives + true_negatives + false_positives + false_negatives) < 1e-15:
       return 0.0
    return (true_positives+true_negatives)/(true_positives+false_positives+true_negatives+false_negatives)

class ValidationResult(object):
    def __init__(self, true_positives, true_negatives, false_positives, false_negatives, description):
        self.true_positives = true_positives
        self.true_negatives = true_negatives
        self.false_positives = false_positives
        self.false_negatives = false_negatives
        self.description = description
        
    def __repr__(self):
        res_precision = precision(self.true_positives, self.false_positives) * 100
        res_accuracy = accuracy(self.true_positives, self.true_negatives, self.false_positives, self.false_negatives) * 100
        return "ValidationResult (Description=%s, Precision=%.2f%%, Accuracy=%.2f%%)" % (self.description, res_precision, res_accuracy)
    
class ValidationStrategy(object):
    def __init__(self, model):
        if not isinstance(model,PredictableModel):
            raise TypeError("Validation can only validate the type PredictableModel.")
        self.model = model
        self.validation_results = []
    
    def add(self, validation_result):
        self.validation_results.append(validation_result)
        
    def validate(self, X, y, description):
        raise NotImplementedError("Every Validation module must implement the validate method!")
        
    
    def print_results(self):
        print(self.model)
        for validation_result in self.validation_results:
            print(validation_result)

    def __repr__(self):
        return "Validation Kernel (model=%s)" % (self.model)
        
class KFoldCrossValidation(ValidationStrategy):
    def __init__(self, model, k=10):
        super(KFoldCrossValidation, self).__init__(model=model)
        self.k = k
        self.logger = logging.getLogger("facerec.validation.KFoldCrossValidation")

    def validate(self, X, y, description="ExperimentName"):
        X,y = shuffle(X,y)
        c = len(np.unique(y))
        foldIndices = []
        n = np.iinfo(np.int).max
        for i in range(0,c):
            idx = np.where(y==i)[0]
            n = min(n, idx.shape[0])
            foldIndices.append(idx.tolist()); 

        if n < self.k:
            self.k = n

        foldSize = int(math.floor(n/self.k))
        
        true_positives, false_positives, true_negatives, false_negatives = (0,0,0,0)
        for i in range(0,self.k):
        
            self.logger.info("Processing fold %d/%d." % (i+1, self.k))
                
            l = int(i*foldSize)
            h = int((i+1)*foldSize)
            testIdx = slice_2d(foldIndices, cols=range(l,h), rows=range(0, c))
            trainIdx = slice_2d(foldIndices,cols=range(0,l), rows=range(0,c))
            trainIdx.extend(slice_2d(foldIndices,cols=range(h,n),rows=range(0,c)))

            Xtrain = [X[t] for t in trainIdx]
            ytrain = y[trainIdx]
                        
            self.model.compute(Xtrain, ytrain)

            for j in testIdx:
                prediction = self.model.predict(X[j])[0]
                if prediction == y[j]:
                    true_positives = true_positives + 1
                else:
                    false_positives = false_positives + 1
                    
        self.add(ValidationResult(true_positives, true_negatives, false_positives, false_negatives, description))
    
    def __repr__(self):
        return "k-Fold Cross Validation (model=%s, k=%s)" % (self.model, self.k)
