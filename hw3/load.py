from keras.models import load_model
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Flatten, Activation, Reshape
from keras.layers.convolutional import Conv2D, ZeroPadding2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.optimizers import SGD, Adam, Adadelta
from keras.utils import to_categorical
import keras.utils.vis_utils
import sys
import pandas as pd
import numpy as np
import csv

def read_data(test_data_path):
	file2  = pd.read_csv(test_data_path, sep=',', header=0)
	file2 = np.array(file2)
	test_data = []
	
	#read test
	for line in file2:
		test_data.append(np.fromstring(line[1] , dtype=float, sep=' ').reshape(48, 48, 1))
	
	test_data = np.array(test_data)
	test_data /= 255
	
	return test_data

if __name__ == '__main_':
	test_x = read_data(sys.argv[1])
	model = load_model('model.h5')
	pre = model.predict_classes(test_x)
	
	#write result
	output = open(sys.argv[2], 'w')
	write1 = csv.writer(output)
	write1.writerow(['id', 'label'])
	for i in range(len(pre)):
			write1.writerow([i, pre[i]])
	
	
	
	
	
	
	
	
	
	
	
	
	
	