import re
import numpy
import operator
from matplotlib import pyplot
from os import listdir
from os.path import isfile, join

def weights(mypath):
	evalues , evectors = kEigenVectors(mypath)
	#evectors = evectors.transpose() 
	L = normalize(mypath)
	evectors = evectors * L
	norms = numpy.linalg.norm(evectors,axis = 0)
	evectors = evectors / norms
	return evectors

def kEigenVectors(mypath):
	c = coVariance(mypath) 
	evalues, evectors = numpy.linalg.eig(c)
	sort_indices = evalues.argsort()[::-1] 
	evalues = evalues[sort_indices]                               # puttin the evalues in that order 
	evectors = evectors[sort_indices]

	evalues_sum = sum(evalues[:])  
	evalues_count = 0
	evalues_energy = 0.0

	for evalue in evalues: 
		evalues_count += 1 
		evalues_energy += evalue / evalues_sum 
		if evalues_energy >= 0.85: 
			break 

	evalues = evalues[0:evalues_count]                            # reduce the number of eigenvectors/values to consider 
	evectors = evectors[0:evalues_count] 
	return evalues,evectors

#not using it anymore, skip
def sortEigenValues(mypath):
	eigValues, eigVectors = eigenVectors(mypath)
	eigPositionList = []
	for i in range(len(eigValues)):
		eigPositionList.append(i)
	eigValuePositionList = zip(eigValues,eigPositionList)
	eigValuePositionList.sort(key=operator.itemgetter(0))
	eigValuePositionList.reverse()
	return eigValuePositionList
		

#not using it anymore, skip
#first column is first eigenVector not first row
#confirm
def eigenVectors(mypath):
	K = coVariance(mypath)
	eigenvalues, eigenvectors = numpy.linalg.eig(K)
	return eigenvalues , eigenvectors

def coVariance(mypath):
	normVectorList = normalize(mypath)
	aMatrix = [list(i) for i in zip(*normVectorList)]
	aTransMat = numpy.matrix(normVectorList)
	aMat = numpy.matrix(aMatrix)
	c = aTransMat * aMat 
	onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
	l = len(onlyfiles)
	c/=l
	return c

def normalize(mypath):
	vectorList = convertTestSetToVectors(mypath)
	#print vectorList[0][150]
	meanVector = meanImage(mypath)
	#print meanVector[150]
	l = len(meanVector)
	normalisedVectorList = []
	for v in vectorList:
		normalisedVector = [0]*l
		for i in range(l):
			normalisedVector[i] = int(v[i]) - int(meanVector[i])
		normalisedVectorList.append(normalisedVector)
	#print normalisedVectorList[0][150]
	#print len(normalisedVectorList)
	return normalisedVectorList

def getImage(mypath):
	imageVector = meanImage(mypath)
	im1d = numpy.asarray(imageVector)
	im2d = im1d.reshape(112,92)
	pyplot.imshow(im2d, pyplot.cm.gray)
	pyplot.show()

def meanImage(mypath):
	vectorList = convertTestSetToVectors(mypath)
	l = len(vectorList[0])
	meanImageVector = [0] * l
	den = len(vectorList)
	for i in range(l):
		for v in vectorList:
			meanImageVector[i] += v[i] 
	for i in range(l):
		meanImageVector[i] /= den	
	return meanImageVector
	

def convertTestSetToVectors(mypath):
        onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
        vectorlist = []
	#print "hi"
	#print (len(onlyfiles))
        for filename in onlyfiles:
                imageVector = read_pgm(mypath + '/' + filename)
                vectorlist.append(imageVector)
	#print len(vectorlist)
	return vectorlist


def read_pgm(filename, byteorder='>'):
    """Return image data from a raw PGM file as numpy array.

    Format specification: http://netpbm.sourceforge.net/doc/pgm.html

    """
    with open(filename, 'rb') as f:
        buffer = f.read()
    try:
        header, width, height, maxval = re.search(
            b"(^P5\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer).groups()
    except AttributeError:
        raise ValueError("Not a raw PGM file: '%s'" % filename)
    return (numpy.frombuffer(buffer,
                            dtype='u1' if int(maxval) < 256 else byteorder+'u2',
                            count=int(width)*int(height),
                            offset=len(header)
                            ).reshape((int(height), int(width)))).flatten()


if __name__ == "__main__":
    from matplotlib import pyplot
    image = read_pgm("foo.pgm", byteorder='<')
    pyplot.imshow(image, pyplot.cm.gray)
    pyplot.show()
