import os
import sqlite3
from detect import *
from convert import *
from flask import render_template, g, send_from_directory
from werkzeug.utils import secure_filename
from flask import Flask, request, redirect, url_for,flash, jsonify
from new import *
import glob
import pickle
import csv
from shutil import copyfile, move
from werkzeug.routing import BaseConverter
from zipfile import *

DATABASE = '/path/to/database.db'
UPLOAD_FOLDER = './static/'
INPUT_FOLDER = './face_jpg/'
OUTPUT_FOLDER = './face_pgm/'
ALLOWED_EXTENSIONS = set(['jpg', 'jpeg'])


app = Flask(__name__)
app.config['INPUT_FOLDER'] = INPUT_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def get_db():
	db = getattr(g, '_database', None)
	if db is None:
		db = g._database = sqlite3.connect(DATABASE)
	return db

@app.teardown_appcontext
def close_connection(exception):
	db = getattr(g, '_database', None)
	if db is not None:
		db.close()

@app.route('/zip/<name>')
def download_zip(name):
	filename = name + ".zip"
	return send_from_directory("./zips/", filename, as_attachment = True)

@app.route('/gallery/<path:filename>')
def download_img(filename):
	return send_from_directory("./gallery/", filename, as_attachment = True)

@app.route('/gallery')
def gallery():
	files = glob.glob("./gallery/*.jpg")#searches all .jpg files, all support later for other extensions
	dict_image = {} #each images links to a list of names
	
	try:
		with open("./gallery/dict_image.dat", 'rb+') as sid:
			dict_image = pickle.load(sid)
	except:
		print "No such File"
		dict_image = {}

	for f in files:
		if f in dict_image:
			continue
		pgmList = detectFaceGallery(f) #detection function path changes to gallery
		#print pgmList, jpgList

		fiscList = []
		for p in pgmList:
			fiscPrediction = identify(p,"abc.csv","model.pkl") #recognition
			if (fiscPrediction[1]['distances'][0] < 0.33):
				fiscList.append(fiscPrediction[0])
		print fiscList

		dict_image[f] = fiscList
		
		for x in pgmList:
			os.remove(x) #removing pgm files for no chance of collission with next one

	print dict_image
	with open("./gallery/dict_image.dat", 'wb') as sid:
		pickle.dump(dict_image,sid)

	return "Gallery Linking Done"

@app.route('/train')
def fisctrain():
	fiscTrain("abc.csv","model.pkl")
        return "Training Done"

class RegexConverter(BaseConverter):
    def __init__(self, url_map, *items):
        super(RegexConverter, self).__init__(url_map)
        self.regex = items[0]


app.url_map.converters['regex'] = RegexConverter

@app.route('/remove/<x>/<y>')
def rmPGM(x,y):
	print x,y
	csvList = {}
	with open('abc.csv') as csvfile:
		reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
		for row in reader:
			r = row[0].split(';')
			csvList[r[0]] = r[1]
	print csvList
	path = csvList[y] + "/" + x
	os.remove(path)
	return jsonify(1);

@app.route('/',  methods = ['GET', 'POST'])
def index():
	if request.method == 'GET':
		return render_template('new.html', flag = 1)

	if request.method == 'POST':
		f = request.files['file']
		f.save(os.path.join(app.config['UPLOAD_FOLDER'],secure_filename(f.filename)))
		imageList = detectFace(os.path.join(app.config['UPLOAD_FOLDER'],secure_filename(f.filename)))
		print "hi"
		print imageList
		imgList = []
		for i in imageList:
			imgList.append( os.path.join(app.config['UPLOAD_FOLDER'],secure_filename(i))  )		
		print imgList
		displayList = []
		pList = []
		#from PCA
		for i in imgList:
			print i
			print "hi"
			displayList.append(i[:9] + "disp" + i[12:-3] + "jpg")

			#####PCA commented for now #####

			#a = train("./resources/ethnic",i)
			#pList.append(a)
		#print "PCA"
		print displayList	
		#print pList
		
		####Commenting because PCA is commented out ######
		
		#con = sqlite3.connect('database.db')
		#con.row_factory = sqlite3.Row
		#cur = con.cursor()
		#print "Opened database successfully"
		#string = "select NAME from user where Id in ("
		#for p in pList:
		#	string += str(p)
		#	string += ","
		#string = string[:-1]
		#string += ")"
		#print string
		#try:
		#	cur.execute(string)
		#except:
		#	return render_template('new.html', flag = 1, msg = 1)
		#rows = cur.fetchall()
		#print rows
		#myList = []
		#for r in rows:
		#	myList.append(r[0])	
		#print myList`

		#from fischer
		print "Fischer"
		fiscList = []
		for i in imgList:
			fiscPrediction = identify(i,"abc.csv","model.pkl")
			print fiscPrediction
			print "result"
			print fiscPrediction[1]['distances'][0]
			if (fiscPrediction[1]['distances'][0] < 0.33):
				fiscList.append(fiscPrediction[0])
		#print fiscList	
		
		detList = zip(fiscList, displayList, imgList)
		#print detList
		
		csvList = {}
		with open('abc.csv') as csvfile:
			reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
			for row in reader:
				r = row[0].split(';')
				csvList[r[0]] = r[1]
		
		#print csvList	
		
		for p in detList:
			src = p[2]
			dst = csvList[p[0]] + '/' + (src.split('/'))[2]
			print src
			print dst
			copyfile(src, dst)
		#	gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
		#	im.save(csvList[p[0]])

		#gallery part from here #####################	
		person_images_dict = {}
		
		for person in fiscList:
			#print person
			dict_image = {} #each images links to a list of names
	
			try:
				with open("./gallery/dict_image.dat", 'rb+') as sid:
					dict_image = pickle.load(sid)
			except:
				print "No such File"
				dict_image = {}

			temp = []
			for k,v in dict_image.items():
				if person in v:
					temp.append(k)
			
			final = []
			for t in temp:
				final.append(t[10:])
			print final

			person_images_dict[person] = final


		###ZIP_PART_HERE
		#remove all initially made zips (from last iteration)
		zfiles = glob.glob("./zips/*.zip")
		for zf in zfiles:
			os.remove(zf)

		for zk,zv in person_images_dict.items():
			zname = "./zips/" + zk + ".zip"
			zip_archive = ZipFile(zname, "w")
			for item in zv:
				zip_archive.write("./gallery/" + item, arcname = item)
			zip_archive.close()

		print person_images_dict
		return render_template('new.html', image1 = os.path.join(app.config['UPLOAD_FOLDER'],secure_filename(f.filename)), imgList = imageList, pList = fiscList, displayList = displayList, flag = 2, total = person_images_dict, detList = detList)

@app.route('/training',  methods = ['GET', 'POST'])
def index_():
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
