
from tensorflow import keras
import cv2
import numpy as np
import os
import random


IMG_DIR = "Google-Image-Scraper-master/photos/sad faces"
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
DESTINATION_ROOT = "dataset/"
RANDOM_TOKEN = random.getrandbits(128)

if not os.path.exists("dataset"):
    os.makedirs("dataset/")
    os.makedirs("dataset/0")
    os.makedirs("dataset/1")
    os.makedirs("dataset/2")


# cv2.namedWindow("Display", cv2.WINDOW_NORMAL)
# # Using resizeWindow()
# cv2.resizeWindow("Display", 500, 500)

IMG_SHAPE = 128
def process():
	c = 0
	prefix = str(RANDOM_TOKEN)+"_google_"

	for img_name in sorted(os.listdir(IMG_DIR)):
		img_path = os.path.join(IMG_DIR, img_name)
		img = cv2.imread(img_path,cv2.IMREAD_COLOR)
		extension = img_name.split('.')[1].lower()
	
		if extension in ['gif', 'png', 'web']:
			continue
		
		gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

		#face detection
		faces = face_cascade.detectMultiScale(gray_image, 1.1, 4)
		if len(faces):          
			x, y, w, h = faces[0]
			face_cropped = img[y: y+h, x: x+w]
			try:
				img_data = cv2.resize(face_cropped, (IMG_SHAPE, IMG_SHAPE))
			except Exception as e:
				print("exception: ", e)
				continue
		else:
			print("no face detected")
			continue

		print("showing image")
		cv2.imshow('Display', img_data)
		key = cv2.waitKey(0) & 0xFF

		if key == ord('a'):
		    cv2.imwrite(os.path.join(os.path.join(DESTINATION_ROOT, "0"),
		                             prefix+"_"+str(c)+".jpg"),
		                             img_data)
		    c += 1
		elif key == ord('b'):
		    cv2.imwrite(os.path.join(os.path.join(DESTINATION_ROOT, "1"),
		                             prefix+"_"+str(c)+".jpg"),
		                             img_data)
		    c += 1

		elif key == ord('c'):
		    cv2.imwrite(os.path.join(os.path.join(DESTINATION_ROOT, "2"),
		                             prefix+"_"+str(c)+".jpg"),
		                             img_data)
		    c += 1
		elif  key == ord('x'):
		    break
		
		cv2.destroyWindow('Display')

# start
process()