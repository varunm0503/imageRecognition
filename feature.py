import numpy as np
from builtins import range

def asColumnMatrix(X):
    """
    Creates a column-matrix from multi-dimensional data items in list l.
    
    X [list] List with multi-dimensional data.
    """
    if len(X) == 0:
        return np.array([])
    total = 1
    for i in range(0, np.ndim(X[0])):
        total = total * X[0].shape[i]
    mat = np.empty([total, 0], dtype=X[0].dtype)
    for col in X:
        mat = np.append(mat, col.reshape(-1,1), axis=1) # same as hstack
    return np.asmatrix(mat)

class AbstractFeature(object):

    def compute(self,X,y):
        raise NotImplementedError("Every AbstractFeature must implement the compute method.")

    def extract(self,X):
        raise NotImplementedError("Every AbstractFeature must implement the extract method.")

    def save(self):
        raise NotImplementedError("Not implemented yet (TODO).")

    def load(self):
        raise NotImplementedError("Not implemented yet (TODO).")

    def __repr__(self):
        return "AbstractFeature"

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

class Identity(AbstractFeature):
    """
    Simplest AbstractFeature you could imagine. It only forwards the data and does not operate on it, 
    probably useful for learning a Support Vector Machine on raw data for example!
    """
    def __init__(self):
        AbstractFeature.__init__(self)
        
    def compute(self,X,y):
        return X
    
    def extract(self,X):
        return X
    
    def __repr__(self):
        return "Identity"
        
class PCA(AbstractFeature):
    def __init__(self, num_components=0):
        AbstractFeature.__init__(self)
        self._num_components = num_components
        
    def compute(self,X,y):
        # build the column matrix
        XC = asColumnMatrix(X)
        y = np.asarray(y)
        # set a valid number of components
        if self._num_components <= 0 or (self._num_components > XC.shape[1]-1):
            self._num_components = XC.shape[1]-1
        # center dataset
        self._mean = XC.mean(axis=1).reshape(-1,1)
        XC = XC - self._mean
        # perform an economy size decomposition (may still allocate too much memory for computation)
        self._eigenvectors, self._eigenvalues, variances = np.linalg.svd(XC, full_matrices=False)
        # sort eigenvectors by eigenvalues in descending order
        idx = np.argsort(-self._eigenvalues)
        self._eigenvalues, self._eigenvectors = self._eigenvalues[idx], self._eigenvectors[:,idx]
        # use only num_components
        self._eigenvectors = self._eigenvectors[0:,0:self._num_components].copy()
        self._eigenvalues = self._eigenvalues[0:self._num_components].copy()
        # finally turn singular values into eigenvalues 
        self._eigenvalues = np.power(self._eigenvalues,2) / XC.shape[1]
        # get the features from the given data
        features = []
        for x in X:
            xp = self.project(x.reshape(-1,1))
            features.append(xp)
        return features
    
    def extract(self,X):
        X = np.asarray(X).reshape(-1,1)
        return self.project(X)
        
    def project(self, X):
        X = X - self._mean
        return np.dot(self._eigenvectors.T, X)

    def reconstruct(self, X):
        X = np.dot(self._eigenvectors, X)
        return X + self._mean

    @property
    def num_components(self):
        return self._num_components

    @property
    def eigenvalues(self):
        return self._eigenvalues
        
    @property
    def eigenvectors(self):
        return self._eigenvectors

    @property
    def mean(self):
        return self._mean
        
    def __repr__(self):
        return "PCA (num_components=%d)" % (self._num_components)
        
class LDA(AbstractFeature):

    def __init__(self, num_components=0):
        AbstractFeature.__init__(self)
        self._num_components = num_components

    def compute(self, X, y):
        # build the column matrix
        XC = asColumnMatrix(X)
        y = np.asarray(y)
        # calculate dimensions
        d = XC.shape[0]
        c = len(np.unique(y))        
        # set a valid number of components
        if self._num_components <= 0:
            self._num_components = c-1
        elif self._num_components > (c-1):
            self._num_components = c-1
        # calculate total mean
        meanTotal = XC.mean(axis=1).reshape(-1,1)
        # calculate the within and between scatter matrices
        Sw = np.zeros((d, d), dtype=np.float32)
        Sb = np.zeros((d, d), dtype=np.float32)
        for i in range(0,c):
            Xi = XC[:,np.where(y==i)[0]]
            meanClass = np.mean(Xi, axis = 1).reshape(-1,1)
            Sw = Sw + np.dot((Xi-meanClass), (Xi-meanClass).T)
            Sb = Sb + Xi.shape[1] * np.dot((meanClass - meanTotal), (meanClass - meanTotal).T)
        # solve eigenvalue problem for a general matrix
        self._eigenvalues, self._eigenvectors = np.linalg.eig(np.linalg.inv(Sw)*Sb)
        # sort eigenvectors by their eigenvalue in descending order
        idx = np.argsort(-self._eigenvalues.real)
        self._eigenvalues, self._eigenvectors = self._eigenvalues[idx], self._eigenvectors[:,idx]
        # only store (c-1) non-zero eigenvalues
        self._eigenvalues = np.array(self._eigenvalues[0:self._num_components].real, dtype=np.float32, copy=True)
        self._eigenvectors = np.matrix(self._eigenvectors[0:,0:self._num_components].real, dtype=np.float32, copy=True)
        # get the features from the given data
        features = []
        for x in X:
            xp = self.project(x.reshape(-1,1))
            features.append(xp)
        return features
        
    def project(self, X):
        return np.dot(self._eigenvectors.T, X)

    def reconstruct(self, X):
        return np.dot(self._eigenvectors, X)

    @property
    def num_components(self):
        return self._num_components

    @property
    def eigenvectors(self):
        return self._eigenvectors
    
    @property
    def eigenvalues(self):
        return self._eigenvalues
    
    def __repr__(self):
        return "LDA (num_components=%d)" % (self._num_components)
        
class Fisherfaces(AbstractFeature):

    def __init__(self, num_components=0):
        AbstractFeature.__init__(self)
        self._num_components = num_components
    
    def compute(self, X, y):
        # turn into numpy representation
        Xc = asColumnMatrix(X)
        y = np.asarray(y)
        # gather some statistics about the dataset
        n = len(y)
        c = len(np.unique(y))
        # define features to be extracted
        pca = PCA(num_components = (n-c))
        lda = LDA(num_components = self._num_components)
        # fisherfaces are a chained feature of PCA followed by LDA
        model = ChainOperator(pca,lda)
        # computing the chained model then calculates both decompositions
        model.compute(X,y)
        # store eigenvalues and number of components used
        self._eigenvalues = lda.eigenvalues
        self._num_components = lda.num_components
        # compute the new eigenspace as pca.eigenvectors*lda.eigenvectors
        self._eigenvectors = np.dot(pca.eigenvectors,lda.eigenvectors)
        # finally compute the features (these are the Fisherfaces)
        features = []
        for x in X:
            xp = self.project(x.reshape(-1,1))
            features.append(xp)
        return features

    def extract(self,X):
        X = np.asarray(X).reshape(-1,1)
        return self.project(X)

    def project(self, X):
        return np.dot(self._eigenvectors.T, X)
    
    def reconstruct(self, X):
        return np.dot(self._eigenvectors, X)

    @property
    def num_components(self):
        return self._num_components
        
    @property
    def eigenvalues(self):
        return self._eigenvalues
    
    @property
    def eigenvectors(self):
        return self._eigenvectors

    def __repr__(self):
        return "Fisherfaces (num_components=%s)" % (self.num_components)

class LocalDescriptor(object):
    def __init__(self, neighbors):
        self._neighbors = neighbors

    def __call__(self,X):
        raise NotImplementedError("Every LBPOperator must implement the __call__ method.")

    @property
    def neighbors(self):
        return self._neighbors

    def __repr__(self):
        return "LBPOperator (neighbors=%s)" % (self._neighbors)

class OriginalLBP(LocalDescriptor):
    def __init__(self):
        LocalDescriptor.__init__(self, neighbors=8)

    def __call__(self,X):
        X = np.asarray(X)
        X = (1<<7) * (X[0:-2,0:-2] >= X[1:-1,1:-1]) \
            + (1<<6) * (X[0:-2,1:-1] >= X[1:-1,1:-1]) \
            + (1<<5) * (X[0:-2,2:] >= X[1:-1,1:-1]) \
            + (1<<4) * (X[1:-1,2:] >= X[1:-1,1:-1]) \
            + (1<<3) * (X[2:,2:] >= X[1:-1,1:-1]) \
            + (1<<2) * (X[2:,1:-1] >= X[1:-1,1:-1]) \
            + (1<<1) * (X[2:,:-2] >= X[1:-1,1:-1]) \
            + (1<<0) * (X[1:-1,:-2] >= X[1:-1,1:-1])
        return X

    def __repr__(self):
        return "OriginalLBP (neighbors=%s)" % (self._neighbors)

class ExtendedLBP(LocalDescriptor):
    def __init__(self, radius=1, neighbors=8):
        LocalDescriptor.__init__(self, neighbors=neighbors)
        self._radius = radius

    def __call__(self,X):
        X = np.asanyarray(X)
        ysize, xsize = X.shape
        # define circle
        angles = 2*np.pi/self._neighbors
        theta = np.arange(0,2*np.pi,angles)
        # calculate sample points on circle with radius
        sample_points = np.array([-np.sin(theta), np.cos(theta)]).T
        sample_points *= self._radius
        # find boundaries of the sample points
        miny=min(sample_points[:,0])
        maxy=max(sample_points[:,0])
        minx=min(sample_points[:,1])
        maxx=max(sample_points[:,1])
        # calculate block size, each LBP code is computed within a block of size bsizey*bsizex
        blocksizey = np.ceil(max(maxy,0)) - np.floor(min(miny,0)) + 1
        blocksizex = np.ceil(max(maxx,0)) - np.floor(min(minx,0)) + 1
        # coordinates of origin (0,0) in the block
        origy =  0 - np.floor(min(miny,0))
        origx =  0 - np.floor(min(minx,0))
        # calculate output image size
        dx = xsize - blocksizex + 1
        dy = ysize - blocksizey + 1
        # get center points
        C = np.asarray(X[origy:origy+dy,origx:origx+dx], dtype=np.uint8)
        result = np.zeros((dy,dx), dtype=np.uint64)
        for i,p in enumerate(sample_points):
            # get coordinate in the block
            y,x = p + (origy, origx)
            # Calculate floors, ceils and rounds for the x and y.
            fx = np.floor(x)
            fy = np.floor(y)
            cx = np.ceil(x)
            cy = np.ceil(y)
            # calculate fractional part    
            ty = y - fy
            tx = x - fx
            # calculate interpolation weights
            w1 = (1 - tx) * (1 - ty)
            w2 =      tx  * (1 - ty)
            w3 = (1 - tx) *      ty
            w4 =      tx  *      ty
            # calculate interpolated image
            N = w1*X[fy:fy+dy,fx:fx+dx]
            N += w2*X[fy:fy+dy,cx:cx+dx]
            N += w3*X[cy:cy+dy,fx:fx+dx]
            N += w4*X[cy:cy+dy,cx:cx+dx]
            # update LBP codes        
            D = N >= C
            #print D
            np.add(result,(1<<i)*D,out=result,casting="unsafe")
            #result += (1<<i)*D
        return result

    @property
    def radius(self):
        return self._radius

    def __repr__(self):
        return "ExtendedLBP (neighbors=%s, radius=%s)" % (self._neighbors, self._radius)

class SpatialHistogram(AbstractFeature):
    def __init__(self, lbp_operator=ExtendedLBP(), sz = (8,8)):
        AbstractFeature.__init__(self)
        if not isinstance(lbp_operator, LocalDescriptor):
            raise TypeError("Only an operator of type facerec.lbp.LocalDescriptor is a valid lbp_operator.")
        self.lbp_operator = lbp_operator
        self.sz = sz

    def compute(self,X,y):
        features = []
        for x in X:
            x = np.asarray(x)
            h = self.spatially_enhanced_histogram(x)
            features.append(h)
        return features

    def extract(self,X):
        X = np.asarray(X)
        return self.spatially_enhanced_histogram(X)

    def spatially_enhanced_histogram(self, X):
        # calculate the LBP image
        L = self.lbp_operator(X)
        # calculate the grid geometry
        lbp_height, lbp_width = L.shape
        grid_rows, grid_cols = self.sz
        py = int(np.floor(lbp_height/grid_rows))
        px = int(np.floor(lbp_width/grid_cols))
        E = []
        for row in range(0,grid_rows):
            for col in range(0,grid_cols):
                C = L[row*py:(row+1)*py,col*px:(col+1)*px]
                H = np.histogram(C, bins=2**self.lbp_operator.neighbors, range=(0, 2**self.lbp_operator.neighbors), normed=True)[0]
                # probably useful to apply a mapping?
                E.extend(H)
        return np.asarray(E)

    def __repr__(self):
        return "SpatialHistogram (operator=%s, grid=%s)" % (repr(self.lbp_operator), str(self.sz))

