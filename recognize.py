import csv, os, sys
try:
    from PIL import Image
except ImportError:
    import Image
import numpy as np
from feature import *
from identifier import *
from method import PredictableModel
from tests import KFoldCrossValidation
from os import listdir
from os.path import isfile, join

from scipy.misc import imresize

from sklearn.externals import joblib

def save_model(filename, model):
    joblib.dump(model, filename, compress=9)

def load_model(filename):
    return joblib.load(filename)

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


class NumericDataSet(object):
    def __init__(self):
        self.data = {}
        self.str_to_num_mapping = {}
        self.num_to_str_mapping = {}

    def add(self, label, image):
        try:
            self.data[label].append(image)
        except:
            self.data[label] = [image]
            numerical_identifier = len(self.str_to_num_mapping)
            # Store in mapping tables:
            self.str_to_num_mapping[label] = numerical_identifier
            self.num_to_str_mapping[numerical_identifier] = label

    def get(self):
        X = []
        y = []
        for name, num in self.str_to_num_mapping.iteritems():
            for image in self.data[name]:
                X.append(image)
                y.append(num)
        return X,y

    def resolve_by_str(self, label):
        return self.str_to_num_mapping[label]

    def resolve_by_num(self, numerical_identifier):
        return self.num_to_str_mapping[numerical_identifier]

    def length(self):
        return len(self.data)

    def __repr__(self):
        print("NumericDataSet")

# This is the face recognition module for the RESTful Webservice.
#
# The current implementation uses a fixed model defined in code. 
# A simple wrapper works around a limitation of the current framework, 
# because as time of writing this only integer labels can be passed into 
# the classifiers of the facerec framework. 

# This wrapper hides the complexity of dealing with integer labels for
# the training labels. It also supports updating a model, instead of 
# re-training it. 
class PredictableModelWrapper(object):

    def __init__(self, model):
        self.model = model
        self.numeric_dataset = NumericDataSet()
        
    def compute(self):
        X,y = self.numeric_dataset.get()
        self.model.compute(X,y)

    def set_data(self, numeric_dataset):
        self.numeric_dataset = numeric_dataset

    def predict(self, image):
        prediction_result = self.model.predict(image)
        # Only take label right now:
        num_label = prediction_result[0]
	dist = prediction_result[1]
        str_label = self.numeric_dataset.resolve_by_num(num_label)
        return str_label,dist

    def update(self, name, image):
        self.numeric_dataset.add(name, image)
        class_label = self.numeric_dataset.resolve_by_str(name)
        extracted_feature = self.feature.extract(image)
        self.classifier.update(extracted_feature, class_label)

    def __repr__(self):
        return "PredictableModelWrapper (Inner Model=%s)" % (str(self.model))


# Now define a method to get a model trained on a NumericDataSet,
# which should also store the model into a file if filename is given.
def get_model(numeric_dataset, model_filename=None):
    feature = ChainOperator(Resize((128,128)), SpatialHistogram())	
    classifier = NearestNeighbor(dist_metric=NormalizedCorrelation(), k=1) 
    inner_model = PredictableModel(feature=feature, classifier=classifier)
    model = PredictableModelWrapper(inner_model)
    model.set_data(numeric_dataset)
    model.compute()
    if not model_filename is None:
        save_model(model_filename, model)
    return model

# Now a method to read images from a folder. It's pretty simple,
# since we can pass a numeric_dataset into the read_images  method 
# and just add the files as we read them. 
def read_images(path, identifier, numeric_dataset):
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    print identifier
    print onlyfiles	
    for filename in os.listdir(path):
        try:
            img = Image.open(os.path.join(path, filename))
            img = img.convert("L")
            img = np.asarray(img, dtype=np.uint8)
            numeric_dataset.add(identifier, img)
        except IOError, (errno, strerror):
            print "I/O error({0}): {1}".format(errno, strerror)
        except:
            print "Unexpected error:", sys.exc_info()[0]
            raise

def read_from_csv(filename):
    numeric_dataset = NumericDataSet()
    with open(filename, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=';', quotechar='#')
        for row in reader:
            identifier = row[0]
            path = row[1]
            read_images(path, identifier, numeric_dataset)
    return numeric_dataset

def get_model_from_csv(filename, out_model_filename):
    numeric_dataset = read_from_csv(filename)
    model = get_model(numeric_dataset, out_model_filename)
    return model

def load_model_file(model_filename):
    model = load_model(model_filename)
    return model
