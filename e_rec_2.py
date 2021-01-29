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

def check_error(predicted, supplied):
	#convert to numpy array because have better functions for doing this
	X = np.array(predicted)
	Y = np.array(supplied)
	N = len(X)
	count = 0
	for i in range(N):
		Y_yes = np.where(Y[i] == 1)[0]
		X_yes = np.amax(X[i])
		if X_yes == Y_yes:
			#print ("supplied value==>", Y[Y_yes], "predicted value==>",X[X_yes])
			count += 1
	return (count/N)
	
# convert a 0->K value into a one-hot encoded array
def one_hot_encode(y, K):  # y is number of rows, K is # of choices
	N = len(y) # get the length of the input array
	ind = np.zeros((N, K)) # create N rows by K columns array of zeros 
	for i in range(N): # loop through each row of data
		ind[i, y.iloc[i]] = 1 # set y[i] will hold a value between 0 and K, set y[i]th element in
	return pd.DataFrame(ind)  # send back new array

#Data preprocessing
def preprocess(df):
	# convert the string of values in pixels column to a dict
	def convert_pixels_to_dict(df):
		df['pixels'] = df[' pixels'] # get rid of this stupid space
		df = df.drop([' pixels'], axis=1) # don't want this after we've converted the old ' pixels' to single columns
		return df
		
	df = convert_pixels_to_dict(df)
	
	return df


def feature_engineer(df):
	def convert_single_list_to_multiple_columns(df):
		df = df[' pixels'].str.split(' ',n=-1, expand=True)
		df.apply(pd.to_numeric)
		return df
	
	df = convert_single_list_to_multiple_columns(df)
	return df
	
#df = pd.read_csv('icml_face_data.csv', header=0, usecols=[0,2])
df = pd.read_csv('skinny.csv', header=0, usecols=[0,2]) # use for initial testing for quick turnaround

#split emotions from df
df_Emotions = df.iloc[1:, [0]].copy()

#need to split column 0 from df to get just the pixels
df = df.iloc[1:, [1]].copy()

#df = preprocess(df) 
df = feature_engineer(df)

# Now standardize, or normalize the data
df_prescaled = df.copy() # right now, df is 18 rows with 1 column for emotion, and 2304 columns for pixel values
df_scaled = scale(df_prescaled)

#convert back to pandas dataframe
cols = df.columns.tolist()
#cols.remove(0) # i think this is to remove emotions column, which is already done

df_scaled = pd.DataFrame(df_scaled, columns=df_prescaled.columns)
df = df_scaled.copy()  # do we need deep=True
df.apply(pd.to_numeric)


X = df
# need to convert df_emotions into 7 new one hot encoded columns
y = one_hot_encode(df_Emotions, 7) # only have emotions in df_Emotions
num_rows = len(df)

#Split data into training, testing, and validation sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) # split the data
#X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2) # split the data

#########################
# Build the model
#########################
model = my_model.CreateModel()

model.fit(X_train, y_train, epochs=200) 

#output = model.predict(X_test)
# Need to use my own comparison func
#my_score = check_error(output, y_test)
#print ("My score func returns", my_score)

#Check Accuracy
scores = model.evaluate(X_train, y_train)
print ("Training Accuracy: %.2f%%\n" % (scores[1]*100))

scores = model.evaluate(X_test, y_test)
print ("Testing Accuracy: %.2f%%\n" % (scores[1]*100))

model.save_weights('faces_nn_weights.h5')
