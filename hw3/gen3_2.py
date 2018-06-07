import pickle 
import os 
import tensorflow as tf
import tensorflow.contrib as tc
import numpy as np
import sys
import os
import scipy.stats as stats
from scipy import misc
import random
import argparse


hair_color = ['orange hair', 'white hair', 'aqua hair', 'gray hair',
    'green hair', 'red hair', 'purple hair', 'pink hair',
    'blue hair', 'black hair', 'brown hair', 'blonde hair']

eye_color = ['gray eyes', 'black eyes', 'orange eyes',
    'pink eyes', 'yellow eyes', 'aqua eyes', 'purple eyes',
    'green eyes', 'brown eyes', 'red eyes', 'blue eyes']

def conv2(batch_input, kernel=3, output_channel=64, stride=1, use_bias=True, scope='conv'):
    with tf.variable_scope(scope):
        if use_bias:
            return  tf.contrib.layers.convolution2d(
                batch_input, output_channel, [kernel, kernel], [stride, stride],
                padding='same',
                weights_initializer=tf.contrib.layers.xavier_initializer(),
                activation_fn=None
                )
        else:
            return  tf.contrib.layers.convolution2d(
                batch_input, output_channel, [kernel, kernel], [stride, stride],
                padding='same',
                weights_initializer=tf.contrib.layers.xavier_initializer(),
                activation_fn=None
                )

def residual_block(inputs, output_channel, stride, scope, train = True):

    with tf.variable_scope(scope):
        net = conv2(inputs, 3, output_channel, stride, use_bias=False, scope='conv_1')
        net = tf.layers.batch_normalization(net, training=train)
        net = tf.nn.relu(net)
        net = conv2(net, 3, output_channel, stride, use_bias=False, scope='conv_2')
        net = tf.layers.batch_normalization(net, training=train)
        net = net + inputs

    return net

def pixelShuffler(inputs, scale=2):
    size = tf.shape(inputs)
    batch_size = size[0]
    h = size[1]
    w = size[2]
    c = inputs.get_shape().as_list()[-1]

    # Get the target channel size
    channel_target = c // (scale * scale)
    channel_factor = c // channel_target

    shape_1 = [batch_size, h, w, channel_factor // scale, channel_factor // scale]
    shape_2 = [batch_size, h * scale, w * scale, 1]

    # Reshape and transpose for periodic shuffling for each channel
    input_split = tf.split(inputs, channel_target, axis=3)
    output = tf.concat([phaseShift(x, scale, shape_1, shape_2) for x in input_split], axis=3)

    return output


def phaseShift(inputs, scale, shape_1, shape_2):
    X = tf.reshape(inputs, shape_1)
    X = tf.transpose(X, [0, 1, 3, 2, 4])
    return tf.reshape(X, shape_2)


class Generator_srresnet(object):

    def __init__(self,  
        hidden_size, 
        img_row, 
        img_col, train = True):
        
        self.hidden_size = hidden_size
        self.img_row = img_row
        self.img_col = img_col
        
        self.batch_size = 64
        self.image_size = img_col

        self.num_resblock = 16
        self.train = train

    def __call__(self, tags_vectors, z, reuse=False, train=True, batch_size = 64):

        self.batch_size = batch_size
        self.train = train
        s = self.image_size # output image size [64]

        s2, s4, s8, s16 = int(s/2), int(s/4), int(s/8), int(s/16)
        gf_dim = 64
        c_dim = 3

        w_init = tf.random_normal_initializer(stddev=0.02)
        gamma_init = tf.random_normal_initializer(1., 0.02)

        with tf.variable_scope("g_net") as scope:

            if reuse:
                scope.reuse_variables()
            
            noise_vector = tf.concat([z, tags_vectors], axis=1)

            net_h0 = tc.layers.fully_connected(
                noise_vector, 64*s8*s8,
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=None
                )

            net_h0 = tf.layers.batch_normalization(net_h0, training=train)
            net_h0 = tf.reshape(net_h0, [-1, s8, s8, gf_dim])
            net = tf.nn.relu(net_h0)

            input_stage = net
  
            for i in range(1, self.num_resblock+1 , 1):
                name_scope = 'resblock_%d'%(i)
                net = residual_block(net, 64, 1, name_scope, train=train)


            net =  tf.layers.batch_normalization(net, training=train)
            net = tf.nn.relu(net)

            net = input_stage + net

            net = conv2(net, 3, 256, 1, use_bias=False, scope='conv1')
            net = pixelShuffler(net, scale=2)
            net =  tf.layers.batch_normalization(net, training=train)
            net = tf.nn.relu(net)

            net = conv2(net, 3, 256, 1, use_bias=False, scope='conv2')
            net = pixelShuffler(net, scale=2)
            net =  tf.layers.batch_normalization(net, training=train)
            net = tf.nn.relu(net)
            
            net = conv2(net, 3, 256, 1, use_bias=False, scope='conv3')
            net = pixelShuffler(net, scale=2)
            net =  tf.layers.batch_normalization(net, training=train)
            net = tf.nn.relu(net)
   

            net = conv2(net, 9, 3, 1, use_bias=False, scope='conv4')

            net = tf.nn.tanh(net)
            
            return net

    @property
    def vars(self):
        return [var for var in tf.global_variables() if "g_net" in var.name]



def load_test(test_path, hair_map, eye_map):

    test = []
    with open(test_path, 'r') as f:

        for line in f.readlines():
            hair = 0
            eye = 0
            if line == '\n':
                break
            line = line.strip().split(',')[1]
            p = line.split(' ')
            p1 = ' '.join(p[:2]).strip()
            p2 = ' '.join(p[-2:]).strip()




            if p1 in hair_map:
                if hair_map[p1]-1 < 0:
                    hair = 11
                else:
                    hair = hair_map[p1]-1
            elif p2 in hair_map:
                if hair_map[p2]-1 < 0:
                    hair = 11
                else:
                    hair = hair_map[p2]-1
            
            
            if p1 in eye_map:
                if eye_map[p1]-1 < 0:
                    eye = 10
                else:
                    eye = eye_map[p1]-1
            elif p2 in eye_map:
                if eye_map[p2]-1 < 0:
                    eye = 10
                else:
                    eye = eye_map[p2]-1



            test.append(make_one_hot(hair, eye))
    
    return test

def make_one_hot(hair, eye):

    eyes_hot = np.zeros([len(eye_color)])
    eyes_hot[eye] = 1
    hair_hot = np.zeros([len(hair_color)])
    hair_hot[hair] = 1
    tag_vec = np.concatenate((eyes_hot, hair_hot))

    return tag_vec


def dump_img(img_dir, img_feats, iters, img_size = 64):
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    img_feats = (img_feats + 1.)/2 * 255.
    img_feats = np.array(img_feats, dtype=np.uint8)

    for idx, img_feat in enumerate(img_feats):
        img_feat =  misc.imresize(img_feat, [img_size, img_size, 3])
        if idx ==0:
            final = img_feat
        else:
            final = np.concatenate((final,img_feat),axis=0)
    return final




tf.flags.DEFINE_integer("z_dim", 100, "noise dim")
tf.flags.DEFINE_string("img_dir", "./gen/", "generated image directory")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
test_size = 1
hair_map = {}
eye_map = {}

for idx, h in enumerate(hair_color):
    hair_map[h] = idx

for idx, e in enumerate(eye_color):
    eye_map[e] = idx

TEST_PATH = sys.argv[1]
MODEL_PATH = './MLDS_hw3_2_model/model/model-77500'

if __name__ == '__main__':

    seq = tf.placeholder(tf.float32, [None, len(hair_color)+len(eye_color)], name="seq")      
    z = tf.placeholder(tf.float32, [None, FLAGS.z_dim])

    g_net = Generator_srresnet(
                        hidden_size=100,
                        img_row=96,
                        img_col=96, train = False)

    result = g_net(seq, z)
    config = tf.ConfigProto(device_count = {'GPU': 0})
    sess = tf.Session(config=config)
    saver = tf.train.Saver(max_to_keep=None)

    saver.restore(sess, save_path=MODEL_PATH)

    z_sampler = stats.truncnorm((-1 - 0.) / 1., (1 - 0.) / 1., loc=0., scale=1)

    test = load_test(TEST_PATH, hair_map, eye_map)

    #z_noise = z_sampler.rvs([test_size, 100])
    #np.save('./z_noise.npy',z_noise)
    z_noise = np.load('./z_noise.npy')

    final_img = []
    for idx, t in enumerate(test):
        t = np.expand_dims(t, axis=0)
        cond = np.repeat(t, test_size, axis=0)
        feed_dict = {seq: cond,  z:z_noise}

        sampler = tf.identity(g_net(seq, z, reuse=True, train=False), name='sampler')

        f_imgs = sess.run(sampler, feed_dict=feed_dict)

        if idx % 5 == 0:
            final = dump_img(FLAGS.img_dir, f_imgs, idx+1)
        else:
            final = np.concatenate((final,dump_img(FLAGS.img_dir, f_imgs, idx+1)),axis=1)

        if idx % 5 == 4:
            final_img.append(final)
        
    for i, ele in enumerate(final_img):
        if i == 0:
            ff = ele
        else:
            ff = np.concatenate((ff,ele))

    misc.imsave('./samples/out1.png',ff)