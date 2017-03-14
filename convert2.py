import re
import numpy

from os import listdir
from os.path import isfile, join

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
        for filename in onlyfiles:
                imageVector = read_pgm(mypath + '/' + filename)
                vectorlist.append(imageVector)
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
