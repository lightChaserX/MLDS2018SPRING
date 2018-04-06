#!/usr/bin/env python3  
from keras.datasets import mnist  
from keras.utils import np_utils
from keras.models import Sequential  
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback  
import numpy as np
import os
import matplotlib.pyplot as plt
import random
import math
import keras.backend as K

np.random.seed(10)
#random.shuffle(index)  
  
# Read MNIST data  

(X_Train, y_Train), (X_Test, y_Test) = mnist.load_data()
'''
training_data = list(zip(list(X_Train), list(y_Train)))
random.shuffle(training_data)
X_Train, y_Train = zip(*training_data)
X_Train = np.array(X_Train)
y_Train = np.array(y_Train)

testing_data = list(zip(list(X_Test), list(y_Test)))
random.shuffle(testing_data)
X_Test, y_Test = zip(*testing_data)
X_Test = np.array(X_Test)
y_Test = np.array(y_Test)

'''



#np.random.shuffle(t)

class GetGradientNormCallback(Callback):
    def __init__(self, layer_index):
        super(GetGradientNormCallback, self).__init__()
        self.layer_index = layer_index
        self.norms = []

    def get_gradient_norm_func(model):
        grads = K.gradients(model.total_loss, model.trainable_weights)
        summed_squares = [K.sum(K.square(g)) for g in grads]
        norm = K.sqrt(sum(summed_squares))
        inputs = model.model._feed_inputs + model.model._feed_targets + model.model._feed_sample_weights
        func = K.function(inputs, [norm])
        return func

    def on_epoch_end(self, epoch, logs=None):
        #grads = K.gradients(model.total_loss, model.trainable_weights)
        #summed_squares = [K.sum(K.square(g)) for g in grads]
        #norm = K.sqrt(sum(summed_squares))
        #inputs = model.model._feed_inputs + model.model._feed_targets + model.model._feed_sample_weights
        #func = K.function(inputs, [norm])
        get_gradient = get_gradient_norm_func(self.model)
        x = [i.reshape(1,-1) for i in X_Train40_norm]
        self.norms.append((get_gradient([X_Train40_norm, y_TrainOneHot, np.ones(len(y_TrainOneHot))])))
        #grads = K.gradients(self.model.total_loss, self.model.trainable_weights)
        #weights = model.trainable_weights
        #weights = [weight for weight in weights if model.get_layer(weight.name[:-2]).trainable]
        #print(grads)
        #print(weights)
        #optimizer = self.model.optimizer
        #print(optimizer)
        #print(optimizer.get_gradients(self.model.total_loss, self.model.trainable_weights))
        #summed_squares = [K.sum(K.square(g)) for g in grads]
        #norm = K.sqrt(sum(summed_squares))
        #self.norms.append(norm)
        #inputs = K.placeholder(shape=(None, 4, 5))
        #with K.get_session() as sess:
            #print(sess.run(norm, inputs))
        #print(norm(x))
        #sess = tf.Session()
        #sess.run(tf.global_variables_initializer())
        #print(sess.run(norm))

        #print(self.norms)


def get_gradient_norm_func(model):
    grads = K.gradients(model.total_loss, model.trainable_weights)
    summed_squares = [K.sum(K.square(g)) for g in grads]
    norm = K.sqrt(sum(summed_squares))
    inputs = model.model._feed_inputs + model.model._feed_targets + model.model._feed_sample_weights
    func = K.function(inputs, [norm])
    return func


  
# Translation of data  
X_Train40 = X_Train.reshape(X_Train.shape[0], 28, 28, 1).astype('float32')  
X_Test40 = X_Test.reshape(X_Test.shape[0], 28, 28, 1).astype('float32')

# Standardize feature data  
X_Train40_norm = X_Train40 / 255.0  
X_Test40_norm = X_Test40 /255.0  


  
# Label Onehot-encoding  
y_TrainOneHot = np_utils.to_categorical(y_Train)  
y_TestOneHot = np_utils.to_categorical(y_Test)

#model1 = Sequential()
model2 = Sequential()
#model3 = Sequential()  



model2.add(Conv2D(filters=16,  
                 kernel_size=(3,3),  
                 padding='same',  
                 input_shape=(28,28,1),  
                 activation='relu'))
model2.add(MaxPooling2D(pool_size=(2,2)))
model2.add(Conv2D(filters=32,  
                 kernel_size=(3,3),  
                 padding='same', 
                 activation='relu'))
model2.add(MaxPooling2D(pool_size=(2,2)))
model2.add(Conv2D(filters=64,  
                 kernel_size=(3,3),  
                 padding='same', 
                 activation='relu'))
model2.add(MaxPooling2D(pool_size=(2,2)))
model2.add(Conv2D(filters=128,  
                 kernel_size=(3,3),  
                 padding='same', 
                 activation='relu'))

  
# Create Max-Pool 1  
model2.add(MaxPooling2D(pool_size=(2,2)))
model2.add(Dropout(0.2))


model2.add(Flatten())

model2.add(Dense(1024, activation='relu'))
model2.add(Dense(505, activation='relu'))



 
model2.add(Dropout(0.5)) #751933



model1 = Sequential()
#model3 = Sequential()  



model1.add(Conv2D(filters=16,  
                 kernel_size=(3,3),  
                 padding='same',  
                 input_shape=(28,28,1),  
                 activation='relu'))
model1.add(MaxPooling2D(pool_size=(2,2)))
model1.add(Conv2D(filters=32,  
                 kernel_size=(3,3),  
                 padding='same', 
                 activation='relu'))
model1.add(MaxPooling2D(pool_size=(2,2)))
model1.add(Conv2D(filters=64,  
                 kernel_size=(3,3),  
                 padding='same', 
                 activation='relu'))
model1.add(MaxPooling2D(pool_size=(2,2)))
model1.add(Conv2D(filters=128,  
                 kernel_size=(3,3),  
                 padding='same', 
                 activation='relu'))

  
# Create Max-Pool 1  
model1.add(MaxPooling2D(pool_size=(2,2)))
model1.add(Dropout(0.2))


model1.add(Flatten())

model1.add(Dense(1024, activation='relu'))
model1.add(Dense(505, activation='relu'))



 
model1.add(Dropout(0.5)) #751933







#model1.add(Dense(10, activation='softmax'))
model2.add(Dense(10, activation='softmax'))
model1.add(Dense(10, activation='softmax'))
#model3.add(Dense(10, activation='softmax'))

#model1.summary()
model2.summary()
#model3.summary()

cbk = GetGradientNormCallback(layer_index=-1)
#model1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model2.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#model3.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#train_history1 = model1.fit(x=X_Train40_norm,  
                          #y=y_TrainOneHot, validation_split=0.2,  
                          #epochs=100, batch_size=300, verbose=1)
train_history2 = model2.fit(x=X_Train40_norm,  
                          y=y_TrainOneHot, validation_split=0.2,  shuffle=False,
                          epochs=30, batch_size=1024,callbacks=[], verbose=1)
train_history1 = model1.fit(x=X_Train40_norm,  
                          y=y_TrainOneHot, validation_split=0.2,  shuffle=False,
                          epochs=30, batch_size=64,callbacks=[], verbose=1)
#train_history3 = model3.fit(x=X_Train40_norm,  
                          #y=y_TrainOneHot, validation_split=0.2,  
                          #epochs=100, batch_size=300, verbose=1)


model3 = Sequential()
model3.add(Conv2D(filters=16,  
                 kernel_size=(3,3),  
                 padding='same',  
                 input_shape=(28,28,1),  
                 activation='relu'))
model3.add(MaxPooling2D(pool_size=(2,2)))
model3.add(Conv2D(filters=32,  
                 kernel_size=(3,3),  
                 padding='same', 
                 activation='relu'))
model3.add(MaxPooling2D(pool_size=(2,2)))
model3.add(Conv2D(filters=64,  
                 kernel_size=(3,3),  
                 padding='same', 
                 activation='relu'))
model3.add(MaxPooling2D(pool_size=(2,2)))
model3.add(Conv2D(filters=128,  
                 kernel_size=(3,3),  
                 padding='same', 
                 activation='relu'))

  
# Create Max-Pool 1  
model3.add(MaxPooling2D(pool_size=(2,2)))
model3.add(Dropout(0.2))


model3.add(Flatten())

model3.add(Dense(1024, activation='relu'))
model3.add(Dense(505, activation='relu'))



 
model3.add(Dropout(0.5)) #751933
model3.add(Dense(10, activation='softmax'))

loss_t = []
val_loss_t = []
acc_t = []
val_acc_t =[]


alpha = np.arange(-1.0, 2.5, 0.5)
for ii in range(len(list(alpha))):
   
    new_1_w = [ (1-alpha[ii])*ele for ele in model1.get_weights()]
    new_2_w = [ (alpha[ii])*ele for ele in model2.get_weights()]

    target_weights = [x+y for x,y in zip(new_1_w, new_2_w)]
    model3.set_weights(target_weights)
    model3.summary()
    #print("====================================================================================")
    #print(ii)

    model3.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    train_history3 = model3.fit(x=X_Train40_norm,  
                          y=y_TrainOneHot, validation_split=0.2, epochs=30, verbose=1)

    loss_t.append(train_history3.history['loss'][len(train_history3.history['loss'])-1])
    val_loss_t.append(train_history3.history['val_loss'][len(train_history3.history['val_loss'])-1])
    acc_t.append(train_history3.history['acc'][len(train_history3.history['acc'])-1])
    val_acc_t.append(train_history3.history['val_acc'][len(train_history3.history['val_acc'])-1])


print(loss_t)
print(val_loss_t)
print(acc_t)
print(val_acc_t)





