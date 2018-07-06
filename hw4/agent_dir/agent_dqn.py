from agent_dir.agent import Agent
from collections import deque
import numpy as np 
import tensorflow as tf
import random


random.seed(87)


FINAL_EXPLORATION = 0.05
TARGET_UPDATE = 1000
ONLINE_UPDATE = 4

MEMORY_SIZE = 10000
EXPLORATION = 1000000

START_EXPLORATION = 1.
TRAIN_START = 10000
LEARNING_RATE = 0.0001
DISCOUNT = 0.99

def lrelu(x, alpha = 0.01):
    return tf.maximum(x, alpha * x)

def clipped_error(x): 
    return tf.where(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5) # condition, true, false


class Agent_DQN(Agent):

    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """
        super(Agent_DQN,self).__init__(env)
        tf.reset_default_graph() 
        self.num_action = 3
        self.minibatch = 32
        self.esp = 1
        self.model_path = "./MLDS_hw4_model/dqn/Breakout_ddqn.ckpt-0-0"
        self.replay_memory = deque()

        
        self.input = tf.placeholder("float", [None, 84, 84, 4])

        self.f1 = tf.get_variable("f1", shape=[8,8,4,32], initializer=tf.contrib.layers.xavier_initializer_conv2d())
        self.f2 = tf.get_variable("f2", shape=[4,4,32,64], initializer=tf.contrib.layers.xavier_initializer_conv2d())
        self.f3 = tf.get_variable("f3", shape=[3,3,64,64], initializer=tf.contrib.layers.xavier_initializer_conv2d())

        self.w1 = tf.get_variable("w1", shape=[7*7*64,512], initializer=tf.contrib.layers.xavier_initializer())
        self.w2 = tf.get_variable("w2", shape=[512, self.num_action], initializer=tf.contrib.layers.xavier_initializer())

        self.py_x = self.build_model(self.input, self.f1, self.f2, self.f3 , self.w1, self.w2)


        self.f1_r = tf.get_variable("f1_r", shape=[8,8,4,32], initializer=tf.contrib.layers.xavier_initializer_conv2d())
        self.f2_r = tf.get_variable("f2_r", shape=[4,4,32,64], initializer=tf.contrib.layers.xavier_initializer_conv2d())
        self.f3_r = tf.get_variable("f3_r", shape=[3,3,64,64], initializer=tf.contrib.layers.xavier_initializer_conv2d())

        self.w1_r = tf.get_variable("w1_r", shape=[7*7*64,512], initializer=tf.contrib.layers.xavier_initializer())
        self.w2_r = tf.get_variable("w2_r", shape=[512, self.num_action], initializer=tf.contrib.layers.xavier_initializer())

        self.py_x_r =self.build_model(self.input, self.f1_r, self.f2_r,self.f3_r, self.w1_r, self.w2_r)


        self.rlist=[0]
        self.recent_rlist=[0]

        self.episode = 0

        self.epoch_score = deque()
        self.epoch_Q = deque()
        self.epoch_on = False
        self.average_Q = deque()
        self.average_reward = deque()
        self.no_life_game = False

        self.a= tf.placeholder(tf.int64, [None])
        self.y = tf.placeholder(tf.float32, [None])
        self.q_target = tf.placeholder(tf.float32, [None], name='Q_target')
        
        a_one_hot = tf.one_hot(self.a, self.num_action, 1.0, 0.0)
        self.q_value = tf.reduce_sum(tf.multiply(self.py_x, a_one_hot), reduction_indices=1)
        
        error = tf.abs(self.q_target - self.q_value)

        diff = clipped_error(error)

        self.loss = tf.reduce_mean(tf.reduce_sum(diff))

        self.optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE,momentum=0,epsilon= 1e-8, decay=0.99)
        self.train_op = self.optimizer.minimize(self.loss)

        self.saver = tf.train.Saver(max_to_keep=None)

        cfg = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.1))
        self.sess = tf.Session(config=cfg)

        if args.test_dqn:
            #you can load your model here
            self.saver.restore(self.sess, save_path = self.model_path)
            print('loading trained model')

    def build_model(self, input1, f1, f2, f3, w1, w2):
    
        c1 = tf.nn.relu(tf.nn.conv2d(input1, f1, strides=[1, 4, 4, 1],data_format="NHWC", padding = "VALID"))
        c2 = tf.nn.relu(tf.nn.conv2d(c1, f2, strides=[1, 2, 2, 1],data_format="NHWC", padding="VALID"))
        c3 = tf.nn.relu(tf.nn.conv2d(c2, f3, strides=[1,1,1,1],data_format="NHWC", padding="VALID"))

        l1 = tf.reshape(c3, [-1, w1.get_shape().as_list()[0]])

        l2 = lrelu(tf.matmul(l1, w1))
        pyx = tf.matmul(l2, w2)

        return pyx    

    def init_game_setting(self):
        """
        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary

        """
        pass


    def train(self):
        """
        Implement your training algorithm here
        """
        pass


    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent

        Input:
            observation: np.array
                stack 4 last preprocessed frames, shape: (84, 84, 4)

        Return:
            action: int
                the predicted action from trained model
        """
        ##################
        # YOUR CODE HERE #
        ##################

        state = np.reshape(observation, (1, 84, 84, 4))
        Q = self.sess.run(self.py_x, feed_dict = {self.input : state})

        self.esp = 0.01

        if self.esp > np.random.rand(1):

            action = np.random.randint(self.num_action)
        else:
            action = np.argmax(Q)

        if action == 0:
            real_a = 1
        elif action == 1:
            real_a = 2
        else:
            real_a = 3

        return real_a
