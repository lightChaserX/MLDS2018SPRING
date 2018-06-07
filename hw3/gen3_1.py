import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import Conv2D, MaxPooling2D,Conv2DTranspose
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Input, Dense, Dropout, Activation, Flatten, Reshape
from keras.layers.core import Flatten
from keras.optimizers import Adam
import numpy as np
from PIL import Image
import glob
import random
import cv2

import sys

#sys.setrecursionlimit(100000)

#loss_file = open('./loss_file.txt','w')

n_colors = 3

def detect(image, cascade_file = "./lbpcascade_animeface.xml"):
    #if not os.path.isfile(cascade_file):
    #    raise RuntimeError("%s: not found" % cascade_file)

    cascade = cv2.CascadeClassifier(cascade_file)
    #image = cv2.imread(filename, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = cascade.detectMultiScale(gray,
                                     # detector options
                                     scaleFactor = 1.1,
                                     minNeighbors = 5,
                                     minSize = (24, 24))

    print("Detect {} faces".format(len(faces)))
    
    if len(faces) >= 20:
        print("Pass !")
    else:
        print("Fail !")
    

    for (x, y, w, h) in faces:
        cv2.rectangle(np.array(image), (x, y), (x + w, y + h), (0, 0, 255), 2)

    return len(faces)

def generator_model():
    model = Sequential()

    model.add(Dense(1024, input_shape=(1,1,100)))
    model.add(Activation('relu'))

    model.add(Dense(128 * 16 * 16))
    model.add(BatchNormalization(momentum=0.5))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    #model.add(LeakyReLU(0.2))

    model.add(Reshape((16, 16, 128)))

    #model.add(UpSampling2D(size=(2, 2)))
    #model.add(Conv2D(128, (4, 4), padding='same',kernel_initializer='glorot_uniform'))
    model.add(Conv2DTranspose(filters=128, kernel_size=(4, 4),
                                  strides=(2, 2), padding='same',
                                  data_format='channels_last',
                                  kernel_initializer='glorot_uniform'))
    model.add(BatchNormalization(momentum=0.5))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    #model.add(LeakyReLU(0.2))

    #model.add(UpSampling2D(size=(2, 2)))
    #model.add(Conv2D(64, (4, 4), padding='same',kernel_initializer='glorot_uniform'))
    model.add(Conv2DTranspose(filters=64, kernel_size=(4, 4),
                                  strides=(2, 2), padding='same',
                                  data_format='channels_last',
                                  kernel_initializer='glorot_uniform'))
    model.add(BatchNormalization(momentum=0.5))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    #model.add(LeakyReLU(0.2))



    model.add(Conv2D(n_colors, (4, 4), padding='same',kernel_initializer='glorot_uniform'))
    model.add(BatchNormalization(momentum=0.5))
    model.add(Activation('tanh'))

    optimizer = Adam(lr=0.00015, beta_1=0.5)
    model.compile(loss='binary_crossentropy',
                      optimizer=optimizer,
                      metrics=None)
    return model


def image_batch(files,batch_size):
    
    files = random.sample(files, batch_size)
    # print(files)
    res = []
    for path in files:
        img = Image.open(path)
        img = img.resize((64, 64))
        arr = np.array(img)
        arr = (arr - 127.5) / 127.5
        arr.resize((64, 64, n_colors))
        res.append(arr)
    return np.array(res)

def combine_images(generated_images, cols=5, rows=5):
    shape = generated_images.shape
    h = shape[1]
    w = shape[2]
    image = np.zeros((rows * h,  cols * w, n_colors))
    for index, img in enumerate(generated_images):
        if index >= cols * rows:
            break
        i = index // cols
        j = index % cols
        image[i*h:(i+1)*h, j*w:(j+1)*w, :] = img[:, :, :]
    image = image * 127.5 + 127.5
    image = Image.fromarray(image.astype(np.uint8))
    return image

def set_trainable(model, trainable):
    model.trainable = trainable
    for layer in model.layers:
        layer.trainable = trainable

def main():
    #noise_file = open('./noise.txt','w')
    batch_size = 50
    generator = generator_model()
    generator.load_weights('./MLDS_hw3_1_model/generator.h5')
    #noise = np.random.normal(0, 1,size=(batch_size,) + (1, 1, 100))
    #np.save('./noise.npy',noise)
    noise = np.load('./noise_3_1.npy')
    generated_images = generator.predict(noise)
    image = combine_images(generated_images)
    #detect(image)
    image.save("./samples/gen3_1.png")


main()