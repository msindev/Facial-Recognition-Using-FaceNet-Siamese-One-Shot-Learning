import cv2
import numpy as np
import os
import time

face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

video_capture = cv2.VideoCapture(0)

name = input("Enter name of person:")

path = 'images'
print(path)
directory = os.path.join(path, name)
print(directory)
if not os.path.exists(directory):
	os.makedirs(directory, exist_ok = 'True')

number_of_images = 0
MAX_NUMBER_OF_IMAGES = 10
count = 0

while number_of_images < MAX_NUMBER_OF_IMAGES:
	ret, frame = video_capture.read()

	frame = cv2.flip(frame, 1)

	faces = face_cascade.detectMultiScale(frame, 1.3, 5)
	for(x,y,w,h) in faces:
		cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
		roi = frame[y:y+h, x:x+w]
		if count == 10:
			cv2.imwrite(os.path.join(directory, str(number_of_images)+'.jpg'), roi)
			number_of_images += 1
			count = 0
		count+=1
		print(count)

	cv2.imshow('Video', frame)
	cv2.waitKey(200)

	if(cv2.waitKey(1) & 0xFF == ord('q')):
		break

video_capture.release()
cv2.destroyAllWindows()
