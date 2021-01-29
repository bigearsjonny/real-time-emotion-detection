import cv2
import math
import os
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Flatten, Dropout, SimpleRNN
import pandas as pd
from sklearn.preprocessing import scale
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from keras.layers import BatchNormalization
from sklearn.metrics import mean_squared_error
from keras.losses import categorical_crossentropy
from keras.optimizers import Adadelta
import numpy as np
import my_model

def convert_img(face_img):
	face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
	#face_img = face_img.astype('float32')/255
	face_img = cv2.resize(face_img, (48, 48))
	#df = np.array(face_img)
	#face_img needs to become an array of 2304 items
	#print ("Shape before flatten:",np.shape(df))
	#df = df.flatten()
	#df = np.array(df)
	#print ("Shape after flatten:",np.shape(df))
	df = pd.DataFrame(face_img)
	df = df.values.flatten()
	df = df.reshape(1,2304)
	#print (df)

	#face_img = face_img.reshape(1, face_img.shape[0], face_img.shape[1], 1)
	return df

def write_on_frame(frame, text, text_x, text_y):
    (text_width, text_height) = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=2)[0]
    box_coords = ((text_x, text_y), (text_x+text_width+20, text_y-text_height-20))
    cv2.rectangle(frame, box_coords[0], box_coords[1], (255, 255, 255), cv2.FILLED)
    cv2.putText(frame, text, (text_x, text_y-10), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,0,0), thickness=2)
    return frame
    
def detect_faces(img, draw_box=True):
	#print ("Detecting a face in image")
	#$convert image to grayscale
	grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	
	#Detect faces
	faces = face_cascades.detectMultiScale(grayscale_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
	
	face_box, face_coords = None, []
	
	#Draw a bounding box around detected faces
	for (x, y, width, height) in faces:
		if draw_box:
			cv2.rectangle(img, (x, y), (x+width, y+height), (0, 255, 0), 5)
		face_box = img[y-5:y+height+5, x:x+width+5]
		face_coords = [x-5, y-5, width+10, height+10]
	return img, face_box, face_coords
			

emotions_list = ("Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral")
face_cascades = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
video_capture = cv2.VideoCapture(0)

##########################
## Build the model
##########################
model = my_model.CreateModel()
#Load the weights
model.load_weights('faces_nn_weights.h5')

while True:
	_, frame = video_capture.read()
	frame, face_img, face_coords = detect_faces(frame, draw_box=False)
	if face_img is not None:
		df = convert_img(face_img)
		prediction = model.predict(df) # prediction returns a 7 item array for each passed row of data, highest value is best guess
		prediction = np.array(prediction)
		x, y, w, h = face_coords
		emotion_index = np.where(prediction == np.amax(prediction)) # find index of highest value
		emotion_index = emotion_index[1].item() # convert result to a scalar
		if prediction[0][emotion_index].item() >= 0.9: # one hot encoded output, each slot refers to an emotion
			text = "{}!!".format(emotions_list[emotion_index]) # the emotion_list and the prediction output map to each other
			cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 5)
		else:
			text = "Detecting....!"
			cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 0, 255), 5)
		frame = write_on_frame(frame, text, face_coords[0], face_coords[1]-10)
	# Display the frame
	cv2.imshow('Video', frame)

	if cv2.waitKey(1) &0xFF == ord('q'):
		break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
