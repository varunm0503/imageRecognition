import os
import sqlite3
from detect import *
from convert import *
from flask import render_template, g
from werkzeug.utils import secure_filename
from flask import Flask, request, redirect, url_for
from new import *

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

@app.route('/hello/')
def hello_world():
	return render_template('front.html')

@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
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
		cur.execute(string)
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
			fiscList.append(fiscPrediction)
		print fiscList		
		return render_template('afterDetection.html', image1 = os.path.join(app.config['UPLOAD_FOLDER'],secure_filename(f.filename)), imgList = imageList, pList = fiscList, displayList = displayList)
		
if __name__ == '__main__':
	app.run(debug = True)
