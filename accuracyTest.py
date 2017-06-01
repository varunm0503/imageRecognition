import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.cm as cm
from builtins import range
from feature import *
from identifier import *
from method import PredictableModel
from tests import KFoldCrossValidation
import numpy as np
try:
    from PIL import Image
except ImportError:
    import Image
import logging

from sklearn.externals import joblib

import matplotlib.pyplot as plt
try:
    from PIL import Image
except ImportError:
    import Image
import math as math

from builtins import range

def create_font(fontname='Tahoma', fontsize=10):
    return { 'fontname': fontname, 'fontsize':fontsize }

def subplot(title, images, rows, cols, sptitle="subplot", sptitles=[], colormap=cm.gray, ticks_visible=True, filename=None):
    fig = plt.figure()
    fig.text(.5, .95, title, horizontalalignment='center')
    for i in range(len(images)):
        ax0 = fig.add_subplot(rows,cols,(i+1))
        plt.setp(ax0.get_xticklabels(), visible=False)
        plt.setp(ax0.get_yticklabels(), visible=False)
        if len(sptitles) == len(images):
            plt.title("%s #%s" % (sptitle, str(sptitles[i])), create_font('Tahoma',10))
        else:
            plt.title("%s #%d" % (sptitle, (i+1)), create_font('Tahoma',10))
        plt.imshow(np.asarray(images[i]), cmap=colormap)
    if filename is None:
        plt.show()
    else:
        fig.savefig(filename)

def minmax_normalize(X, low, high, minX=None, maxX=None, dtype=np.float):
    if minX is None:
        minX = np.min(X)
    if maxX is None:
        maxX = np.max(X)
    minX = float(minX)
    maxX = float(maxX)
    X = X - minX
    X = X / (maxX - minX)
    X = X * (high-low)
    X = X + low
    return np.asarray(X, dtype=dtype)

def save_model(filename, model):
    joblib.dump(model, filename, compress=9)

def load_model(filename):
    return joblib.load(filename)

def read_images(path, sz=None):
    c = 0
    X,y = [], []
    for dirname, dirnames, filenames in os.walk(path):
        for subdirname in dirnames:
            subject_path = os.path.join(dirname, subdirname)
            for filename in os.listdir(subject_path):
                try:
                    im = Image.open(os.path.join(subject_path, filename))
                    im = im.convert("L")
                    if (sz is not None):
                        im = im.resize(sz, Image.ANTIALIAS)
                    X.append(np.asarray(im, dtype=np.uint8))
                    y.append(c)
                except IOError as e:
                    print("I/O error: {0}".format(e))
                    raise e
                except:
                    print("Unexpected error: {0}".format(sys.exc_info()[0]))
                    raise
            c = c+1
    return [X,y]

if __name__ == "__main__":
    out_dir = None
    if len(sys.argv) < 2:
        print("USAGE: facerec_demo.py </path/to/images>")
        sys.exit()
    [X,y] = read_images(sys.argv[1])
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger = logging.getLogger("facerec")
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    feature = SpatialHistogram()
    classifier = NearestNeighbor(dist_metric=NormalizedCorrelation(), k=1)
    my_model = PredictableModel(feature=feature, classifier=classifier)
    my_model.compute(X, y)
    save_model('modelTry.pkl', my_model)
    model = load_model('modelTry.pkl')
    E = []
    #for i in range(min(model.feature.eigenvectors.shape[1], 16)):
    #    e = model.feature.eigenvectors[:,i].reshape(X[0].shape)
    #    E.append(minmax_normalize(e,0,255, dtype=np.uint8))
    # Plot them and store the plot to "python_fisherfaces_fisherfaces.pdf"
    subplot(title="Fisherfaces", images=E, rows=4, cols=4, sptitle="PCA", colormap=cm.jet, filename="fisherfaces.png")
    cv = KFoldCrossValidation(model, k=10)
    cv.validate(X, y)
    cv.print_results()
