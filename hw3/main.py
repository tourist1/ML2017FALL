import pandas as pd
import numpy as np
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Dropout, Flatten, Activation, Reshape
from keras.layers.convolutional import Conv2D, ZeroPadding2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.optimizers import SGD, Adam, Adadelta
from keras.utils.np_utils import to_categorical
import keras.utils.vis_utils
import csv
from keras.preprocessing.image import ImageDataGenerator

if __name__ == '__main__':
	file1  = pd.read_csv(sys.argv[1], sep=',', header=0)
	file2  = pd.read_csv('test.csv', sep=',', header=0)
	file1 = np.array(file1)
	file2 = np.array(file2)
	
	train_y = []
	train_data = []
	test_data = []
	
	# read train
	for line in file1:
		train_y.append(line[0])
		train_data.append(np.fromstring(line[1] , dtype=float, sep=' ').reshape(48, 48, 1))
	
	train_y = np.array(train_y)
	train_data = np.array(train_data)
	train_data /= 255

	#read test
	for line in file2:
		test_data.append(np.fromstring(line[1] , dtype=float, sep=' ').reshape(48, 48, 1))
	
	test_data = np.array(test_data)
	test_data /= 255
	
	# make one hot
	#num_classes = 7
	train_y = to_categorical(train_y)#, num_classes)
	
	#arguments
	batch_size = 128
	epochs = 25
	n = len(train_data)
	steps_per_epoch = int(n / 128) + 1
	shape = (48, 48, 1)
	
	#build model
	model = Sequential()
	
	model.add(Conv2D(64, kernel_size = (5, 5), padding='valid', input_shape=shape, activation='relu'))
	model.add(ZeroPadding2D(padding=(2, 2), data_format='channels_last'))
	model.add(MaxPooling2D(pool_size=(5, 5), strides=(2, 2)))
	model.add(ZeroPadding2D(padding=(1, 1), data_format='channels_last'))

	model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
	model.add(ZeroPadding2D(padding=(1, 1), data_format='channels_last'))

	model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
	model.add(ZeroPadding2D(padding=(1, 1), data_format='channels_last'))
	model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))

	model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
	model.add(ZeroPadding2D(padding=(1, 1), data_format='channels_last'))

	model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
	model.add(ZeroPadding2D(padding=(1, 1), data_format='channels_last'))
	model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))
	model.add(Flatten())

	model.add(Dense(1024, activation='relu'))
	model.add(Dropout(0.5))

	model.add(Dense(1024, activation='relu'))
	model.add(Dropout(0.5))

	model.add(Dense(7, activation='softmax'))
	
	# opt = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
	opt = Adam(lr=1e-3)
	# opt = Adadelta(lr=0.1, rho=0.95, epsilon=1e-08)
	
	model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
	'''
	model.fit(x=train_data, y=train_y, 
		validation_split=0.1,
		batch_size=batch_size, 
		epochs=epochs, verbose=2)
	'''
	gen = ImageDataGenerator(rotation_range=2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
	gen.fit(train_data)
	model.fit_generator(gen.flow(train_data, train_y, batch_size=batch_size), steps_per_epoch=steps_per_epoch, epochs=epochs, verbose=1)
	pre = model.predict_classes(test_data)

	model.save('model1.h5')
	output = open('output.csv', 'w')
	write1 = csv.writer(output)
	write1.writerow(['id', 'label'])
	for i in range(len(pre)):
			write1.writerow([i, pre[i]])	
			
			
			
			
