import os
import sqlite3
from detect import *
from convert import *
from flask import render_template, g, send_from_directory
from werkzeug.utils import secure_filename
from flask import Flask, request, redirect, url_for, flash
from new import *
import glob
import pickle
from shutil import copyfile,move
import csv

INPUT_FOLDER = './face_jpg/'
OUTPUT_FOLDER = './face_pgm/'
UPLOAD_FOLDER = './static/'
ALLOWED_EXTENSIONS = set(['jpg', 'jpeg'])

app = Flask(__name__)
app.config['INPUT_FOLDER'] = INPUT_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/training',  methods = ['GET', 'POST'])
def index():
	if request.method == 'GET':
		# flash('this is your first time right!')
		return render_template('train.html')

	if request.method == 'POST':
		files = request.files.getlist("file[]")
		name = request.form.get("name")
		print name
		src_path = INPUT_FOLDER+name+'/'
		os.makedirs(src_path)
		imageList=[]
		for file in files:
			file.save(src_path+file.filename)
			tempList = detectFace(src_path + file.filename)
			imageList.extend(tempList)
		print imageList
		no_of_folders = len(glob.glob(OUTPUT_FOLDER+'*'))
		des_path = OUTPUT_FOLDER+ 's'+ str(no_of_folders+1)+'/'
		os.makedirs(des_path) #new folder created
		for i in imageList:
			copyfile(UPLOAD_FOLDER + i, des_path + i)  #copy and paste
		# 	# move(UPLOAD_FOLDER+i, new_path)    #cut and paste
		with open(INPUT_FOLDER+'list.csv', 'a') as f:
			writer = csv.writer(f)
			writer.writerow([name,des_path])
		return "detection done"

if __name__ == '__main__':
	app.secret_key = 'super secret key'
	app.run(debug = True)

