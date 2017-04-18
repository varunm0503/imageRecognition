from feature import Fisherfaces
from classifier import NearestNeighbor
from model import PredictableModel
from fisc import *

model = PredictableModel(Fisherfaces(), NearestNeighbor())
[X,y] = read_images("./resources/att_faces/")
model.compute(X,y)
model.predict(X)

