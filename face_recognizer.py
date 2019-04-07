'''Face Recognition Main File'''
import cv2
import numpy as np
import glob
from scipy.spatial import distance
from imutils import face_utils
from keras.models import load_model
import tensorflow as tf

from fr_utils import *
from inception_blocks_v2 import *

#with CustomObjectScope({'tf': tf}):
FR_model = load_model('nn4.small2.v1.h5')
print("Total Params:", FR_model.count_params())

face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

threshold = 0.25

face_database = {}

for name in os.listdir('images'):
	for image in os.listdir(os.path.join('images',name)):
		identity = os.path.splitext(os.path.basename(image))[0]
		face_database[identity] = fr_utils.img_path_to_encoding(os.path.join('images',name,image), FR_model)

print(face_database)

video_capture = cv2.VideoCapture(0)
while True:
	ret, frame = video_capture.read()
	frame = cv2.flip(frame, 1)

	faces = face_cascade.detectMultiScale(frame, 1.3, 5)
	for(x,y,w,h) in faces:
		cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
		roi = frame[y:y+h, x:x+w]
		encoding = img_to_encoding(roi, FR_model)
		min_dist = 100
		identity = None

		for(name, encoded_image_name) in face_database.items():
			dist = np.linalg.norm(encoding - encoded_image_name)
			if(dist < min_dist):
				min_dist = dist
				identity = name
			print('Min dist: ',min_dist)

		if min_dist < 0.1:
			cv2.putText(frame, "Face : " + identity[:-1], (x, y - 50), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
			cv2.putText(frame, "Dist : " + str(min_dist), (x, y - 20), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
		else:
			cv2.putText(frame, 'No matching faces', (x, y - 20), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2)

	cv2.imshow('Face Recognition System', frame)
	if(cv2.waitKey(1) & 0xFF == ord('q')):
		break

video_capture.release()
cv2.destroyAllWindows()

