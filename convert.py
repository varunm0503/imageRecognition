import os 
import cv2 
import sys 
import shutil 
import random 
import numpy as np 

def train(mypath, myimgpath, _energy = 0.85):
	#onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
	faces_count = 5 
	faces_dir = mypath                                                             # directory path to the AT&T faces 

	train_faces_count = 12                                                      # number of faces used for training 
	test_faces_count = 4                                                      # number of faces used for testing 
 
	l = train_faces_count * faces_count                                         # training images count 
	m = 92                                                                      # number of columns of the image 
	n = 112                                                                     # number of rows of the image 
	mn = m * n                                                                  # length of the column vector 
	print '> Initializing started' 
	faces_dir = mypath 
	energy = _energy 
	training_ids = []                                                  # train image id's for every at&t face
 	r_ids = []
	L = np.empty(shape=(mn, l), dtype='float64')                  # each row of L represents one train image 
	
	cur_img = 0 
	for face_id in xrange(1, faces_count + 1): 
		training_ids = random.sample(range(1, 15), train_faces_count)  # the id's of the 6 random training images 
		r_ids.append(training_ids)                              # remembering the training id's for later 
		for training_id in training_ids: 
			path_to_img = os.path.join(faces_dir, 's' + str(face_id), str(training_id) + '.pgm')          # relative path
			#print path_to_img
			#print(cur_img) 
			img = cv2.imread(path_to_img, 0)                                # read a grayscale image 
			img_col = np.array(img, dtype='float64').flatten()              # flatten the 2d image into 1d 
			L[:, cur_img] = img_col[:]                                      # set the cur_img-th column to the current training image 
			#print L
			cur_img += 1 
 
	mean_img_col = np.sum(L, axis=1) / l                          # get the mean of all images / over the rows of L  
	for j in xrange(0, l):                                             # subtract from all training images 
		L[:, j] -= mean_img_col[:] 
 
	C = np.matrix(L.transpose()) * np.matrix(L)                             # instead of computing the covariance matrix as 
	C /= l                                                             # L*L^T, we set C = L^T*L, and end up with way 												                                                                           # smaller and computentionally inexpensive one 
                                                                                 # we also need to divide by the number of training 
                                                                                 # images   
	evalues, evectors = np.linalg.eig(C)                          # eigenvectors/values of the covariance matrix 
	sort_indices = evalues.argsort()[::-1]                             # getting their correct order - decreasing 
	evalues = evalues[sort_indices]                               # puttin the evalues in that order 
	evectors = evectors[sort_indices]                             # same for the evectors 

	evalues_sum = sum(evalues[:])                                      # include only the first k evectors/values so 
	evalues_count = 0                                                       # that they include approx. 85% of the energy 
	evalues_energy = 0.0 
	for evalue in evalues: 
		evalues_count += 1 
		evalues_energy += evalue / evalues_sum 
		if evalues_energy >= energy: 
			break 

	evalues = evalues[0:evalues_count]                            # reduce the number of eigenvectors/values to consider 
	evectors = evectors[0:evalues_count] 
 
	evectors = evectors.transpose()                               # change eigenvectors from rows to columns 
	evectors = L * evectors                                       # left multiply to get the correct evectors 
	norms = np.linalg.norm(evectors, axis=0)                           # find the norm of each eigenvector 
	evectors = evectors / norms                                   # normalize all eigenvectors 
 
	W = evectors.transpose() * L                                  # computing the weights 
	#print W
	print '> Initializing ended' 
	
	#img = cv2.imread("./resources/att_faces/s1/2.pgm", 0)                                        # read as a grayscale image 
	img = cv2.imread(myimgpath,0)
	img_col = np.array(img, dtype='float64').flatten()                      # flatten the image 
	img_col -= mean_img_col                                            # subract the mean column 
	img_col = np.reshape(img_col, (mn, 1))                             # from row vector to col vector 
	S = evectors.transpose() * img_col                                 # projecting the normalized probe onto the 
                                                                            # Eigenspace, to find out the weights 
  
	diff = W - S                                                       # finding the min ||W_j - S|| 
	norms = np.linalg.norm(diff, axis=0) 

	closest_face_id = np.argmin(norms)                                      # the id [0..240) of the minerror face to the sample 
	print (closest_face_id / train_faces_count) + 1 
	return (closest_face_id / train_faces_count) + 1                   # return the faceid (1..40) 

if __name__ == "__main__":
    train("./resources/ethnic","./yk.pgm")
