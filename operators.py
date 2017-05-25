from feature import AbstractFeature
import numpy as np

class FeatureOperator(AbstractFeature):
    """
    A FeatureOperator operates on two feature models.
    
    Args:
        model1 [AbstractFeature]
        model2 [AbstractFeature]
    """
    def __init__(self,model1,model2):
        if (not isinstance(model1,AbstractFeature)) or (not isinstance(model2,AbstractFeature)):
            raise Exception("A FeatureOperator only works on classes implementing an AbstractFeature!")
        self.model1 = model1
        self.model2 = model2
    
    def __repr__(self):
        return "FeatureOperator(" + repr(self.model1) + "," + repr(self.model2) + ")"
    
class ChainOperator(FeatureOperator):
    """
    The ChainOperator chains two feature extraction modules:
        model2.compute(model1.compute(X,y),y)
    Where X can be generic input data.
    
    Args:
        model1 [AbstractFeature]
        model2 [AbstractFeature]
    """
    def __init__(self,model1,model2):
        FeatureOperator.__init__(self,model1,model2)
        
    def compute(self,X,y):
        X = self.model1.compute(X,y)
        return self.model2.compute(X,y)
        
    def extract(self,X):
        X = self.model1.extract(X)
        return self.model2.extract(X)
    
    def __repr__(self):
        return "ChainOperator(" + repr(self.model1) + "," + repr(self.model2) + ")"
