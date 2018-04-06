import numpy as np
import tensorflow as tf
from keras.layers import Dense, Activation
from keras.models import Sequential
import keras.backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
import matplotlib.pyplot as plt
import math
import time

class CollectWeightCallback(Callback):
    def __init__(self, layer_index):
        super(CollectWeightCallback, self).__init__()
        self.layer_index = layer_index
        self.weights = []
        self.count = 1
        self.accs = []
        self.losses = []

    def on_epoch_end(self, epoch, logs={}):
        if (self.count%3 == 0):
            total_weights = np.array([])
            for layer in self.model.layers:
                weights = layer.get_weights()
                total_weights = np.append(total_weights,weights[0].flatten())
            self.weights.append(list(total_weights))
            self.accs.append(logs.get('acc'))
            self.losses.append(logs.get('loss'))
        
        if(self.count==300):
            with open("Output_w.txt", "w") as text_file:
                text_file.write(str(self.weights))
            with open("Output_loss.txt", "w") as text_file:
                text_file.write(str(self.losses))
            with open("Output_acc.txt", "w") as text_file:
                text_file.write(str(self.accs))

        
        self.count += 1
        
        
            
            #print(weights[0][0])
        
        
        #print(self.model.trai(nable_weights)
        #print(layer.get_weights()[0][1])
        #print(layer.get_weights())

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
        self.norms.append((get_gradient([x.reshape((-1,1)), (np.array(y)).reshape((-1,1)), np.ones(len(y))])))
        print((get_gradient([x.reshape((-1,1)), (np.array(y)).reshape((-1,1)), np.ones(len(y))])))
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

'''
def get_gradient_norm_func(model):
    
    
    
    inputs = model.model._feed_inputs + model.model._feed_targets + model.model._feed_sample_weights
    func = K.function(inputs, [norm])
    return norm
'''

def sign(list):
	y = []
	for x in list:
		if x>0:
			y.append(1)
		else:
			y.append(-1)
	return y

x = np.arange(0.01, 1, 0.001)
y = sign(np.sin(5*math.pi*x))
#y = np.sin(5*math.pi*x)/(5*math.pi*x)
#model1 = Sequential()
model2= Sequential()
#model3 = Sequential()
'''
model1.add(Dense(input_shape=(1,),units=1,activation='tanh'))
model1.add(Dense(input_shape=(1,),units=5,activation='tanh'))
model1.add(Dense(input_shape=(5,),units=5,activation='tanh'))
model1.add(Dense(input_shape=(5,),units=5,activation='tanh'))
model1.add(Dense(input_shape=(5,),units=10,activation='tanh'))
model1.add(Dense(input_shape=(10,),units=10,activation='tanh'))
model1.add(Dense(input_shape=(10,),units=10,activation='tanh'))
model1.add(Dense(input_shape=(10,),units=5,activation='tanh'))
model1.add(Dense(input_shape=(5,),units=1,activation='tanh'))
model1.add(Dense(input_shape=(1,),units=1,activation='tanh'))
model1.add(Dense(input_shape=(1,),units=10,activation='tanh'))
model1.add(Dense(input_shape=(10,),units=18,activation='tanh'))
model1.add(Dense(input_shape=(18,),units=15,activation='tanh'))
model1.add(Dense(input_shape=(15,),units=4,activation='tanh'))
model1.add(Dense(input_shape=(4,),units=1,activation='tanh'))##987
'''


model2.add(Dense(input_shape=(1,),units=1,activation='tanh'))
model2.add(Dense(input_shape=(1,),units=5,activation='tanh'))
model2.add(Dense(input_shape=(5,),units=10,activation='tanh'))
model2.add(Dense(input_shape=(10,),units=10,activation='tanh'))
model2.add(Dense(input_shape=(10,),units=29,activation='tanh'))
model2.add(Dense(input_shape=(29,),units=10,activation='tanh'))
model2.add(Dense(input_shape=(10,),units=10,activation='tanh'))
model2.add(Dense(input_shape=(10,),units=6,activation='tanh'))
model2.add(Dense(input_shape=(6,),units=1,activation='tanh'))#984


'''
model3.add(Dense(input_shape=(1,),units=1,activation='tanh'))
model3.add(Dense(input_shape=(1,),units=5,activation='tanh'))
model3.add(Dense(input_shape=(5,),units=5,activation='tanh'))
model3.add(Dense(input_shape=(5,),units=5,activation='tanh'))
model3.add(Dense(input_shape=(5,),units=10,activation='tanh'))
model3.add(Dense(input_shape=(10,),units=10,activation='tanh'))
model3.add(Dense(input_shape=(10,),units=10,activation='tanh'))
model3.add(Dense(input_shape=(10,),units=5,activation='tanh'))
model3.add(Dense(input_shape=(5,),units=1,activation='tanh'))
model3.add(Dense(input_shape=(1,),units=1,activation='tanh'))
model3.add(Dense(input_shape=(1,),units=10,activation='tanh'))
model3.add(Dense(input_shape=(10,),units=10,activation='tanh'))
model3.add(Dense(input_shape=(10,),units=10,activation='tanh'))
model3.add(Dense(input_shape=(10,),units=4,activation='tanh'))
model3.add(Dense(input_shape=(4,),units=1,activation='tanh'))
model3.add(Dense(input_shape=(1,),units=1,activation='tanh'))
model3.add(Dense(input_shape=(1,),units=93,activation='tanh'))
model3.add(Dense(input_shape=(93,),units=1,activation='tanh')) #986
'''

#model1.compile(loss='mse', optimizer='adam', metrics=['mse'])
model2.compile(loss='mse', optimizer='adam', metrics=['mse','accuracy'])
#model3.compile(loss='mse', optimizer='adam', metrics=['mse'])
#model1.summary()
model2.summary()
#model3.summary()

t1 = time.clock()
earlystopping = EarlyStopping(monitor='mse', patience = 20, verbose=1, mode='min')
checkpoint = ModelCheckpoint(filepath='best.hdf5',
                                 verbose=1,
                                 save_best_only=True,
                                 save_weights_only=True,
                                 monitor='mse',
                                mode='min')
'''
for layer in model.layers:
    weights = layer.get_weights()
    print(layer, weights)
'''

cbk = CollectWeightCallback(layer_index=-1)
#get_gradient = get_gradient_norm_func(model)    
#hist1 = model1.fit(x, y, epochs=20000, batch_size=256, callbacks=[],verbose=1)
hist2 = model2.fit(x, y, epochs=300, batch_size=256, callbacks=[cbk],verbose=1)
#hist3 = model3.fit(x, y, epochs=20000, batch_size=256, callbacks=[],verbose=1)
#print(get_gradient([x.reshape((-1,1)), y, np.ones(len(y))]))
#print(model.total_loss)




