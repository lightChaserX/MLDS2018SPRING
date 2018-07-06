from agent_dir.agent import Agent
import scipy
import numpy as np
import tensorflow as tf
import time


STOP_ACTION = 1
UP_ACTION = 2
DOWN_ACTION = 3
action_dict = {DOWN_ACTION: 0, UP_ACTION: 1, STOP_ACTION: 2}


def prepro(o,image_size=[80,80]):
    """
    Call this function to preprocess RGB image to grayscale image if necessary
    This preprocessing code is from
        https://github.com/hiwonjoon/tf-a3c-gpu/blob/master/async_agent.py
    
    Input: 
    RGB image: np.array
        RGB screen of game, shape: (210, 160, 3)
    Default return: np.array 
        Grayscale image, shape: (80, 80, 1)
    
    """
    y = o.astype(np.uint8)
    resized = scipy.misc.imresize(y, image_size)
    return np.expand_dims(resized.astype(np.float32),axis=2)


class Agent_PG(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """

        super(Agent_PG,self).__init__(env)

        self.env = env
        self.max_step = 10000
        self.learning_rate = 0.001
        self.input_shape = [None, 6400] #the observation
        self.hidden_dim = 200
        self.num_action = 6
        self.batch_size = 1
        self.batch_size_episodes = 1
        self.gamma = 0.99

        self.model_dir = "model_pg"

        self.value_scale = 1
        self.entropy_scale = 1
        self.model = './MLDS_hw4_model/pg/model_pg_cnn-25300.ckpt'

        self._sess = tf.Session()

        self.saver = None
        self.input = tf.placeholder(tf.float32, shape=self.input_shape, name='X')
        self.up_probability = self.build_model(self.input)


        self.sampled_actions = tf.placeholder(tf.int32, (None,1), name='sampled_actions')
        self.discounted_reward = tf.placeholder(tf.float32, (None,1), name='discounted_reward')

        self.loss = tf.losses.log_loss(labels=self.sampled_actions, predictions=self.up_probability, weights=self.discounted_reward)

        
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

        gvs = self.optimizer.compute_gradients(self.loss)
        capped_gvs = [(tf.clip_by_value(grad, -0.3, 0.3), var) for grad, var in gvs]
        self.train_op = self.optimizer.apply_gradients(capped_gvs)

        self.init_op = tf.global_variables_initializer()
        
        self.first_move = True
        self.last_observation = None
        self.current_observation = None

        if args.test_pg:
            #you can load your model here
            self.saver = tf.train.Saver()

            if self.model:
                self.saver.restore(self._sess, self.model)
                print('loading trained model')
            else:
                print("No trained moedel found")

        ##################
        # YOUR CODE HERE #
        ##################

    def build_model(self, X):

        observation = tf.reshape(X, [-1,6400])
        
        h = tf.layers.dense(observation, units=self.hidden_dim, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
        p = tf.layers.dense(h, units=1, activation=tf.sigmoid, kernel_initializer=tf.contrib.layers.xavier_initializer())

        return p


    def process_frame(self, frame):
        
        """ Atari specific preprocessing, consistent with DeepMind """
        frame = frame[35:195]
        frame = frame[::2, ::2, 0]
        frame[ frame == 144] = 0
        frame[frame == 109] = 0
        frame[frame != 0] = 1

        return frame.astype(np.float).ravel()    


    def init_game_setting(self):
        """

        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary

        """
        ##################
        # YOUR CODE HERE #
        ##################
        self.first_move = True


    def train(self):
        """
        Implement your training algorithm here
        """
        ##################
        # YOUR CODE HERE #]\]\zzzz
        ##################
        



    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent

        Input:
            observation: np.array
                current RGB screen of game, shape: (210, 160, 3)

        Return:
            action: int
                the predicted action from trained model
        """
        ##################
        # YOUR CODE HERE #
        ##################
        if (self.first_move):
        
            self.last_observation = self.process_frame(observation)
            action = self.env.action_space.sample() ##random sample action
            self.first_move = False
        else:

            self.observation = self.process_frame(observation)


            observation_delta = self.observation - self.last_observation
            self. last_observation = self.observation
                
                    
            up_probability = self._sess.run(self.up_probability, feed_dict = {self.input:observation_delta.reshape([1, -1])})

            if np.random.uniform() < up_probability:
                action = UP_ACTION
            else:
                action = DOWN_ACTION

        return action

