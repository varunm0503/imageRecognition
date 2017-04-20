import cStringIO
import base64

from PIL import Image

import sys

from model import PredictableModel
from lbp import ExtendedLBP
from feature import SpatialHistogram
from distance import ChiSquareDistance
from classifier import NearestNeighbor

import logging
from logging.handlers import RotatingFileHandler

import recognition

def read_image(base64_image):
    """ Decodes Base64 image data, reads it with PIL and converts it into grayscale.

    Args:
    
        base64_image [string] A Base64 encoded image (all types PIL supports).
    """
    enc_data = base64.b64decode(base64_image)
    file_like = cStringIO.StringIO(enc_data)
    im = Image.open(file_like)
    im = im.convert("L")
    print im
    return im

def preprocess_image(image_data):
    image = read_image(image_data)
    return image

def get_prediction(image_data):
    image = preprocess_image(image_data)
    print image
    prediction = model.predict(image)
    return prediction

def identify(image_path, csv_file , model_filename):
	print image_path
	print csv_file
	print model_filename
	global model	
	model = recognition.get_model_from_csv(filename=csv_file,out_model_filename=model_filename)
	print model
	image = open(image_path, 'rb') #open binary file in read mode
	image_read = image.read()
	image_data = base64.encodestring(image_read)
	#print image_data
	prediction = get_prediction(image_data)
	return prediction


if __name__ == '__main__':
	from argparse import ArgumentParser	
	parser = ArgumentParser()
	parser.add_argument("-t", "--train", action="store", dest="dataset", default=None,help="Calculates a new model from a given CSV file. CSV format: <person>;</path/to/image/folder>.", required=False)
	parser.add_argument('model_filename', nargs='?', help="Filename of the model to use or store")
	print "=== Usage ===" 
	parser.print_help() 
	args = parser.parse_args()
	print args
	global model
	#model = recognition.load_model_file(args.model_filename)
	model = recognition.get_model_from_csv(filename=args.dataset,out_model_filename=args.model_filename)
	print model
	image = open('cr.pgm', 'rb') #open binary file in read mode
	image_read = image.read()
	image_data = base64.encodestring(image_read)
	#print image_data
	prediction = get_prediction(image_data)
	print prediction
