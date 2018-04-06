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
        self.grad_norms = []

    def on_epoch_end(self, epoch, logs=None):
        layer = self.model.layers[self.layer_index]
        self.weights.append(layer.get_weights())
        total_model_parms = 0
        
        for layer in self.model.layers:
            weights = layer.get_weights()
            for i in range(len(weights[0])):
            	for j in range(len(weights[0][i])):
            		total_model_parms += np.power(weights[0][i][j],2)

        grad_norm = np.power(total_model_parms, 0.5)
        self.grad_norms.append(grad_norm)
        print(grad_norm)
            
            #print(weights[0][0])
        
        
        #print(self.model.trainable_weights)
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

model1 = Sequential()
model2= Sequential()
model3 = Sequential()

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



model2.add(Dense(input_shape=(1,),units=1,activation='tanh'))
model2.add(Dense(input_shape=(1,),units=5,activation='tanh'))
model2.add(Dense(input_shape=(5,),units=10,activation='tanh'))
model2.add(Dense(input_shape=(10,),units=10,activation='tanh'))
model2.add(Dense(input_shape=(10,),units=29,activation='tanh'))
model2.add(Dense(input_shape=(29,),units=10,activation='tanh'))
model2.add(Dense(input_shape=(10,),units=10,activation='tanh'))
model2.add(Dense(input_shape=(10,),units=6,activation='tanh'))
model2.add(Dense(input_shape=(6,),units=1,activation='tanh'))#984



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


model1.compile(loss='mse', optimizer='adam', metrics=['mse'])
model2.compile(loss='mse', optimizer='adam', metrics=['mse'])
model3.compile(loss='mse', optimizer='adam', metrics=['mse'])

model1.summary()
model2.summary()
model3.summary()

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

#cbk = GetGradientNormCallback(layer_index=-1)
#get_gradient = get_gradient_norm_func(model)    
hist1 = model1.fit(x, y, epochs=20000, batch_size=256, callbacks=[],verbose=1)
hist2 = model2.fit(x, y, epochs=20000, batch_size=256, callbacks=[],verbose=1)
hist3 = model3.fit(x, y, epochs=20000, batch_size=256, callbacks=[],verbose=1)




pred1 = model1.predict(x)
pred2 = model2.predict(x)
pred3 = model3.predict(x)


hist1 = [math.log10(i) for i in hist1.history['loss']]
hist2 = [i for i in hist2.history['loss']]
hist3 = [math.log10(i) for i in hist3.history['loss']]



plt.plot(hist1,'r',label='model1')
plt.plot(hist2,'b',label='model2')
plt.plot(hist3,'g',label='model3')
plt.legend(loc='upper right')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.savefig("fig_loss_1_1_1.png")
plt.show()



plt.plot(x, y, 'b',label='practice')
plt.plot(x, pred1, 'r--',label='model1')
plt.plot(x, pred2, 'g--',label='model2')
plt.plot(x, pred3, 'c--',label='model3')
plt.legend(loc='upper right')
plt.savefig("fig_pred_1_1_1.png")
plt.show()

