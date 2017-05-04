import os
import sqlite3
from detect import *
from convert import *
from flask import render_template, g, send_from_directory
from werkzeug.utils import secure_filename
from flask import Flask, request, redirect, url_for
from new import *
import glob
import pickle

DATABASE = '/path/to/database.db'
UPLOAD_FOLDER = './static/'
ALLOWED_EXTENSIONS = set(['jpg', 'jpeg'])

app = Flask(__name__)
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
			if (fiscPrediction[1]['distances'][0] < 3.0):
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
			a = train("./resources/ethnic",i)
			pList.append(a)
		print "PCA"
		print displayList	
		print pList
		
		con = sqlite3.connect('database.db')
		con.row_factory = sqlite3.Row
		cur = con.cursor()
		print "Opened database successfully"
		string = "select NAME from user where Id in ("
		for p in pList:
			string += str(p)
			string += ","
		string = string[:-1]
		string += ")"
		print string
		try:
			cur.execute(string)
		except:
			return render_template('new.html', flag = 1, msg = 1)
		rows = cur.fetchall()
		print rows
		myList = []
		for r in rows:
			myList.append(r[0])	
		print myList

		#from fischer
		print "Fischer"
		fiscList = []
		for i in imgList:
			fiscPrediction = identify(i,"abc.csv","model.pkl")
			print fiscPrediction
			if (fiscPrediction[1]['distances'][0] < 3.0):
				fiscList.append(fiscPrediction[0])
		#print fiscList	

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

		print person_images_dict
		return render_template('new.html', image1 = os.path.join(app.config['UPLOAD_FOLDER'],secure_filename(f.filename)), imgList = imageList, pList = fiscList, displayList = displayList, flag = 2, total = person_images_dict)

if __name__ == '__main__':
	app.run(debug = True)
