import cv2
import numpy as np
import glob
from scipy.spatial import distance
from imutils import face_utils
from keras.models import load_model
from keras.utils import CustomObjectScope
import tensorflow as tf

from fr_utils import *
from inception_blocks_v2 import *

with CustomObjectScope({'tf': tf}):
  FR_model = load_model('nn4.small2.v1.h5')
print("Total Params:", FR_model.count_params())

face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

threshold = 0.25

face_database = {}

for image in glob.glob('images/*'):
	identity = os.path.splitext(os.path.basename(image))[0][:-1]
	face_database[identity] = fr_utils.img_path_to_encoding(image, FR_Model)

video_capture = cv2.VideoCapture(0)
while True:
	ret, frame = video_capture.read()
	frame = cv2.flip(frame, 1)

	faces = face_cascade.detectMultiScale(frame, 1.3, 5)
	for(x,y,w,h) in faces:
		cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)
		roi = frame[y:y+h, x:x+w]
		encoding = img_to_encoding(roi, FR_Model)
		min_dist = 100
		identity = None

		for(name, encoded_image_name) in face_database.items():
			dist = np.linalg.norm(encoded_image_name - encoding)
			if(dist < min_dist):
				min_dist = dist
				identity = name

		if min_dist < 0.1:
			cv2.putText(img, "Face : " + name, (x, y - 50), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
			cv2.putText(img, "Dist : " + str(min_dist), (x, y - 20), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
		else:
			cv2.putText(img, 'No matching faces', (x, y - 20), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2)

	cv2.imshow('Face Recognition System', frame)
	if(cv2.waitKey(1) & 0xFF == ord('q')):
		break

video_capture.release()
cv2.destroyAllWindows()

