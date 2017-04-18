import numpy as np
import cv2
from PIL import Image
from resizeimage import resizeimage

def detectFace(img_path):
	faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
	eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
	cascade = cv2.CascadeClassifier("/vagrant/detect/haarcascade_frontalface_alt.xml")
	img = cv2.imread(img_path)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	# Detect faces in the image
	faces = faceCascade.detectMultiScale( gray, 1.3,5)
	print (len(faces))
	imgList = []
	# Draw a rectangle around the faces
	for (x,y,w,h) in faces:
		#print x
		#print y
		#print w
		#print h
		image = Image.open(img_path)
		img2 = image.crop((x,y,x+w,y+h))
		name = "img" + str(x) + str(y) + ".jpg" 
		name3 = "./static/disp" + str(x) + str(y) + ".jpg" 
		new_width = 92
		new_height = 112
		img2 = img2.resize((new_width, new_height), Image.ANTIALIAS)
		img2.save(name)
		img2.save(name3)
		img3 = cv2.imread(name)
		gray2 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
		name2 = "./static/img" + str(x) + str(y) + ".pgm"
		im = Image.fromarray(gray2)
		imgList.append( "img" + str(x) + str(y) + ".pgm" )
		im.save(name2)
		#gray2.save(name2)
	print "detected"
	return imgList

		
