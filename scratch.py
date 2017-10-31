#Importing packages
import keras
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.utils import np_utils

#Loading data
train = pd.read_csv('C:/Users/sayan/Desktop/mnist.csv')
test = pd.read_csv('C:/Users/sayan/Desktop/test_mnist.csv')

#One-Hot encoding! amd converion to Integer
trainY = np_utils.to_categorical(train.values[:,0].astype('int32'), 10)
testY = np_utils.to_categorical(train.values[:,0].astype('int32'),10)

trainX = train.values[:,1:].astype('float32')
trainX /= 255
testX = test.values[:,1:].astype('float32')
testX/= 255

#Building model
model = Sequential()
model.add(Dense(32, input_dim=(28,28)))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(32))
model.add(Activation('tanh'))
model.add(Dropout(0.15))
model.add(Activation('softmax'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(trainX, trainY, batch_size=32, nb_epoch=1000, validation_data=(testX, testY), shuffle=True, show_accuracy=True)

#Model result
result = model.predict_classes(x_test, batch_size=32, verbose=1)
print(result)
