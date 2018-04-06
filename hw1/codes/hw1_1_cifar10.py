#!/usr/bin/env python3  
from keras.datasets import cifar10  
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers import Activation
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils 
import numpy as np
import os
import matplotlib.pyplot as plt
np.random.seed(10)  

batch_size = 32 
# 32 examples in a mini-batch, smaller batch size means more updates in one epoch
num_classes = 10 #
epochs = 100 # rep
  
# Read MNIST data  
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
  
y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train  /= 255
x_test /= 255

model1 = Sequential()
model2 = Sequential()
model3= Sequential()
 


model1.add(Conv2D(16, (3, 3), padding='same', activation='relu', input_shape=x_train.shape[1:]))
model1.add(Dropout(0.2))

model1.add(Conv2D(16,(3,3),padding='same', activation='relu'))
model1.add(MaxPooling2D(pool_size=(2,2)))

model1.add(Conv2D(32,(3,3),padding='same',activation='relu'))
model1.add(Dropout(0.2))

model1.add(Conv2D(32,(3,3),padding='same',activation='relu'))
model1.add(MaxPooling2D(pool_size=(2,2)))

model1.add(Conv2D(64,(3,3),padding='same',activation='relu'))
model1.add(Dropout(0.2))

model1.add(Conv2D(64,(3,3),padding='same',activation='relu'))
model1.add(MaxPooling2D(pool_size=(2,2)))

model1.add(Conv2D(128,(3,3),padding='same',activation='relu'))
model1.add(Dropout(0.2))

model1.add(Conv2D(128,(3,3),padding='same',activation='relu'))
model1.add(MaxPooling2D(pool_size=(2,2)))

model1.add(Flatten())
model1.add(Dropout(0.2))
model1.add(Dense(1024,activation='relu',kernel_constraint=maxnorm(3)))
model1.add(Dense(512,activation='relu',kernel_constraint=maxnorm(3)))
model1.add(Dense(256,activation='relu',kernel_constraint=maxnorm(3)))
model1.add(Dense(128,activation='relu',kernel_constraint=maxnorm(3)))
model1.add(Dense(64,activation='relu',kernel_constraint=maxnorm(3)))
model1.add(Dense(32,activation='relu',kernel_constraint=maxnorm(3)))
model1.add(Dropout(0.2))
model1.add(Dense(num_classes, activation='softmax'))#829082


model2.add(Conv2D(16, (3, 3), padding='same', activation='relu', input_shape=x_train.shape[1:]))
model2.add(Dropout(0.2))

model2.add(Conv2D(16,(3,3),padding='same', activation='relu'))
model2.add(Conv2D(16,(3,3),padding='same', activation='relu'))
model2.add(Conv2D(16,(3,3),padding='same', activation='relu'))
model2.add(MaxPooling2D(pool_size=(2,2)))

model2.add(Conv2D(32,(3,3),padding='same',activation='relu'))
model2.add(Dropout(0.2))

model2.add(Conv2D(32,(3,3),padding='same',activation='relu'))
model2.add(Conv2D(32,(3,3),padding='same',activation='relu'))
model2.add(Conv2D(32,(3,3),padding='same',activation='relu'))
model2.add(MaxPooling2D(pool_size=(2,2)))

model2.add(Conv2D(64,(3,3),padding='same',activation='relu'))
model2.add(Dropout(0.2))

model2.add(Conv2D(64,(3,3),padding='same',activation='relu'))
model2.add(Conv2D(64,(3,3),padding='same',activation='relu'))
model2.add(Conv2D(64,(3,3),padding='same',activation='relu'))
model2.add(MaxPooling2D(pool_size=(2,2)))

model2.add(Conv2D(128,(3,3),padding='same',activation='relu'))
model2.add(Dropout(0.2))

model3.add(Conv2D(128,(3,3),padding='same',activation='relu'))
model3.add(Conv2D(128,(3,3),padding='same',activation='relu'))
model3.add(Conv2D(128,(3,3),padding='same',activation='relu'))
model3.add(MaxPooling2D(pool_size=(2,2)))

model3.add(Flatten())
model3.add(Dropout(0.2))
model3.add(Dense(1024,activation='relu',kernel_constraint=maxnorm(3)))
model3.add(Dense(512,activation='relu',kernel_constraint=maxnorm(3)))
model3.add(Dense(256,activation='relu',kernel_constraint=maxnorm(3)))
model3.add(Dense(128,activation='relu',kernel_constraint=maxnorm(3)))
model3.add(Dropout(0.2))
model3.add(Dense(num_classes, activation='softmax'))




model1.summary()
model2.summary()
model3.summary()

model1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model2.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model3.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

cnn1 = model.fit(x=x_train,  
                          y=y_train, validation_split=0.2,  
                          epochs=100, batch_size=batch_size, verbose=1, shuffle=True)
cnn2 = model.fit(x=x_train,  
                          y=y_train, validation_split=0.2,  
                          epochs=100, batch_size=batch_size, verbose=1, shuffle=True)
cnn3 = model.fit(x=x_train,  
                          y=y_train, validation_split=0.2,  
                          epochs=100, batch_size=batch_size, verbose=1, shuffle=True)


hist1 = [i for i in cnn1.history['loss']]
hist2 = [i for i in cnn2.history['loss']]
hist3 = [i for i in cnn3.history['loss']]



plt.plot(hist1,'r',label='model1')
plt.plot(hist2,'g',label='model2')
plt.plot(hist3,'b',label='model3')
plt.legend(loc='upper right')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.savefig("fig_loss_cifar10_1_1.png")
plt.show()

hist1 = [i for i in cnn1.history['acc']]
hist2 = [i for i in cnn2.history['acc']]
hist3 = [i for i in cnn3.history['acc']]


plt.plot(hist1,'r',label='model1')
plt.plot(hist2,'g',label='model2')
plt.plot(hist3,'b',label='model3')
plt.legend(loc='upper right')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.savefig("fig_acc_cifar10_1_1.png")
plt.show()
