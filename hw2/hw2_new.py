import pandas as pd
import numpy as np
import csv
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout 

x_train_file = "X_train"
y_train_file = "Y_train"
x_test_file = "X_test"

### Load Data
x_train = pd.read_csv(x_train_file, sep=',', header=0)
x_train = np.array(x_train.values)
y_train = pd.read_csv(y_train_file, sep=',', header=0)
y_train = np.array(y_train.values)
x_test = pd.read_csv(x_test_file, sep=',', header=0)
x_test = np.array(x_test.values)

# input dim = 106

### Normalization
x_train_test = np.concatenate((x_train, x_test))
mu = (sum(x_train_test) / x_train_test.shape[0])
sigma = np.std(x_train_test, axis=0)
mu = np.tile(mu, (x_train_test.shape[0], 1))
sigma = np.tile(sigma, (x_train_test.shape[0], 1))
x_train_test_normed = (x_train_test - mu) / sigma
x_train = x_train_test_normed[0:x_train.shape[0]]
x_test = x_train_test_normed[x_train.shape[0]:]
y_train = np_utils.to_categorical(y_train)

### Training
model = Sequential()
model.add(Dense(units=500, input_dim=106, kernel_initializer='normal', activation='sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(units=500, kernel_initializer='normal', activation='sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(units=2, kernel_initializer='normal', activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x=x_train, y=y_train, validation_split=0.1, epochs=60, batch_size=128, verbose=1)
model.save('w')

### Prediction
prediction = model.predict_classes(x_test) 

### Output file
with open('ptt_drop02.csv', 'w') as csvfile:
    spamwriter = csv.writer(csvfile)
    spamwriter.writerow(['id', 'label'])
    for i in range(len(prediction)):
        spamwriter.writerow([i+1, prediction[i]])