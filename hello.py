import os
from detect import *
from flask import render_template
from werkzeug.utils import secure_filename
from flask import Flask, request, redirect, url_for

UPLOAD_FOLDER = './static/'
ALLOWED_EXTENSIONS = set(['jpg', 'jpeg'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/hello/')
def hello_world():
	return render_template('front.html')

@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
	if request.method == 'POST':
		f = request.files['file']
		f.save(os.path.join(app.config['UPLOAD_FOLDER'],secure_filename(f.filename)))
		imgList = detectFace(os.path.join(app.config['UPLOAD_FOLDER'],secure_filename(f.filename)))
		print "hi"
		print imgList
		#return "test"
		return render_template('afterDetection.html', image1 = os.path.join(app.config['UPLOAD_FOLDER'],secure_filename(f.filename)))
		
if __name__ == '__main__':
	app.run(debug = True)
