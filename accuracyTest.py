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
    # main title
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
    """ min-max normalize a given matrix to given range [low,high].
    
    Args:
        X [rows x columns] input data
        low [numeric] lower bound
        high [numeric] upper bound
    """
    if minX is None:
        minX = np.min(X)
    if maxX is None:
        maxX = np.max(X)
    minX = float(minX)
    maxX = float(maxX)
    # Normalize to [0...1].    
    X = X - minX
    X = X / (maxX - minX)
    # Scale to [low...high].
    X = X * (high-low)
    X = X + low
    return np.asarray(X, dtype=dtype)

def save_model(filename, model):
    joblib.dump(model, filename, compress=9)

def load_model(filename):
    return joblib.load(filename)

def read_images(path, sz=None):
    """Reads the images in a given folder, resizes images on the fly if size is given.

    Args:
        path: Path to a folder with subfolders representing the subjects (persons).
        sz: A tuple with the size Resizes 

    Returns:
        A list [X,y]

            X: The images, which is a Python list of numpy arrays.
            y: The corresponding labels (the unique number of the subject, person) in a Python list.
    """
    c = 0
    X,y = [], []
    for dirname, dirnames, filenames in os.walk(path):
        for subdirname in dirnames:
            subject_path = os.path.join(dirname, subdirname)
            for filename in os.listdir(subject_path):
                try:
                    im = Image.open(os.path.join(subject_path, filename))
                    im = im.convert("L")
                    # resize to given size (if given)
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
    # This is where we write the images, if an output_dir is given
    # in command line:
    out_dir = None
    # You'll need at least a path to your image data, please see
    # the tutorial coming with this source code on how to prepare
    # your image data:
    if len(sys.argv) < 2:
        print("USAGE: facerec_demo.py </path/to/images>")
        sys.exit()
    # Now read in the image data. This must be a valid path!
    [X,y] = read_images(sys.argv[1])
    # Then set up a handler for logging:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    # Add handler to facerec modules, so we see what's going on inside:
    logger = logging.getLogger("facerec")
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    # Define the Fisherfaces as Feature Extraction method:
    feature = SpatialHistogram()
    # Define a 1-NN classifier with Euclidean Distance:
    classifier = NearestNeighbor(dist_metric=NormalizedCorrelation(), k=1)
    # Define the model as the combination
    my_model = PredictableModel(feature=feature, classifier=classifier)
    # Compute the Fisherfaces on the given data (in X) and labels (in y):
    my_model.compute(X, y)
    # We then save the model, which uses Pythons pickle module:
    save_model('modelTry.pkl', my_model)
    model = load_model('modelTry.pkl')
    # Then turn the first (at most) 16 eigenvectors into grayscale
    # images (note: eigenvectors are stored by column!)
    E = []
    #for i in range(min(model.feature.eigenvectors.shape[1], 16)):
    #    e = model.feature.eigenvectors[:,i].reshape(X[0].shape)
    #    E.append(minmax_normalize(e,0,255, dtype=np.uint8))
    # Plot them and store the plot to "python_fisherfaces_fisherfaces.pdf"
    subplot(title="Fisherfaces", images=E, rows=4, cols=4, sptitle="Fisherface", colormap=cm.jet, filename="fisherfaces.png")
    # Perform a 10-fold cross validation
    cv = KFoldCrossValidation(model, k=10)
    cv.validate(X, y)
    # And print the result:
    cv.print_results()
