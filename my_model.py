from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Flatten, Dropout, SimpleRNN

def CreateModel():
	#########################
	# Build the model
	#########################
	model = Sequential()
	# add the first hidden layer
	# I've tried 512, 256, 128, 64, and 32. 64 seems to work the best
	# I've tried relu, softplu, softmax, and sigmoid. softplus seems to work the best
	model.add(Dense(64, activation='softplus', input_dim=2304)) # 2304 columns in the data, therefor, 2304 input_dim
	# I've tried .7, .5, .3, .1 .7 is ok with a lot of epochs, .1 is ok with fewer epochs
	model.add(Dropout(0.3))
	# I've tried some very deep networks, one or two hidden layers seem to work the best
	#model.add(Dense(128, activation='softplus')) 
	#model.add(Dropout(0.5))
	# I've tried 64, 32, 16 on the second hidden layer smaller than previous is better
	model.add(Dense(32, activation='softplus')) 
	model.add(Dropout(0.1))
	# on a whim I added a hidden layer same size as output array, seems to work ok
	model.add(Dense(7, activation='softplus')) 
	#model.add(Dense(16, activation='softplus')) 
	#model.add(Dropout(0.1))
	model.add(Dense(units=7, activation='softmax'))
	
	model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=['accuracy'])
	
	return model
