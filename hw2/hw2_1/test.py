import tensorflow as tf
import pandas as pd
import numpy as np
import os
import sys
import time
from keras.preprocessing import sequence
from tensorflow.python.layers.core import Dense
import random
import json
import argparse
import matplotlib.pyplot as plt
from bleu_eval import count_ngram, clip_count, best_length_match, brevity_penalty, geometric_mean, BLEU 
sys.setrecursionlimit(15000)
os.environ['CUDA_VISIBLE_DEVICES'] = "0"


test_feature = []
X_test = []
y_test = []



TEST_DATA_DIR = sys.argv[1]
TEST_VIDEO_DIR = os.path.join(TEST_DATA_DIR,'feat/')

TEST_ID_DIR = os.path.join(TEST_DATA_DIR,'id.txt')
print(TEST_ID_DIR)
MODEL_SAVE_DIR = './saved_models/'

OUTPUT_FILE = sys.argv[2]


test_id = pd.read_csv(TEST_ID_DIR, header=None, names=['id'])


image_dim = 4096
hidden_dim = 256
num_of_video_lstm_steps = 80
num_of_caption_lstm_steps = 20
num_of_frames = 80
num_of_epochs = 1
batch_size = 50
learning_rate = 0.001



class VideoCaptionGenerator():
	def __init__(self, image_dim, num_of_words, hidden_dim, batch_size, num_of_lstm_steps, num_of_video_lstm_steps, 
				num_of_caption_lstm_steps, bias_init_vector = None):
		self.image_dim = image_dim
		self.num_of_words = num_of_words
		self.hidden_dim = hidden_dim
		self.batch_size = batch_size
		self.num_of_lstm_steps = num_of_lstm_steps
		self.num_of_video_lstm_steps = num_of_video_lstm_steps
		self.num_of_caption_lstm_steps = num_of_caption_lstm_steps

		self.lstm_cell_1 = tf.nn.rnn_cell.BasicLSTMCell(hidden_dim, state_is_tuple=False)
		self.lstm_cell_2 = tf.nn.rnn_cell.BasicLSTMCell(hidden_dim, state_is_tuple=False)


	def build_model(self):
		video_feature = tf.placeholder(tf.float32, [self.batch_size, self.num_of_video_lstm_steps, self.image_dim])

		caption = tf.placeholder(tf.int32, [self.batch_size, self.num_of_caption_lstm_steps+2])
		

		caption_mask = tf.placeholder(tf.float32, [self.batch_size, self.num_of_caption_lstm_steps+1])

		dropout_prob = tf.placeholder(tf.float32, name="Dropout_Keep_Probability")

		
		encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_dim)
		encoder_output, encoder_state = tf.nn.dynamic_rnn(encoder_cell, video_feature, dtype=tf.float32, time_major=False)

		attention_mechanism = tf.contrib.seq2seq.LuongAttention( self.hidden_dim, encoder_output)

		with tf.variable_scope("embedding"):
			embedding_decoder = tf.Variable(tf.truncated_normal(shape=[ self.num_of_words,  self.hidden_dim], stddev=0.1), name='embedding_decoder')
			decoder_emb_inp = tf.nn.embedding_lookup(embedding_decoder, caption[:,:-1])

		decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_dim)
		decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism, attention_layer_size= self.hidden_dim)
		
		decoder_seq_length = [self.num_of_caption_lstm_steps+1] * self.batch_size

		helper = tf.contrib.seq2seq.TrainingHelper(decoder_emb_inp, decoder_seq_length, time_major=False)

		projection_layer = Dense(self.num_of_words, use_bias=False )
		decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper, 
			encoder_state, output_layer = projection_layer)

		outputs,_,_ = tf.contrib.seq2seq.dynamic_decode(decoder)
		logits = outputs.rnn_output
		result = outputs.sample_id

		cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=caption[:,1:])
		train_loss = (tf.reduce_sum(cross_entropy * caption_mask) / self.batch_size)
		
		return train_loss, video_feature, caption, caption_mask, outputs.sample_id, dropout_prob


	def build_generator(self):
		video_feature = tf.placeholder(tf.float32, [1, self.num_of_video_lstm_steps, self.image_dim])

		encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_dim)
		encoder_output, encoder_state = tf.nn.dynamic_rnn(encoder_cell, video_feature, dtype=tf.float32, time_major=False)

		attention_mechanism = tf.contrib.seq2seq.LuongAttention( self.hidden_dim, encoder_output)

		with tf.variable_scope("embedding"):
			embedding_decoder = tf.Variable(tf.truncated_normal(shape=[ self.num_of_words,  self.hidden_dim], stddev=0.1), name='embedding_decoder')

		decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_dim)
		decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism, attention_layer_size= self.hidden_dim)
		
		decoder_seq_length = [self.num_of_caption_lstm_steps+1] * self.batch_size

		helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding_decoder, tf.fill([1], 1), 2)

		projection_layer = Dense(self.num_of_words, use_bias=False )
		decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper, 
		decoder_cell.zero_state( 1, tf.float32).clone(cell_state=encoder_state), output_layer = projection_layer)

		outputs,_,_ = tf.contrib.seq2seq.dynamic_decode(decoder,  maximum_iterations=num_of_caption_lstm_steps)
    	
		result = outputs.sample_id

		generated_words = outputs.sample_id

		return video_feature, generated_words

def processCaptions(captions):
	captions = map(lambda x: x.replace('.', ''), captions)
	captions = map(lambda x: x.replace(',', ''), captions)
	captions = map(lambda x: x.replace('"', ''), captions)
	captions = map(lambda x: x.replace('\n',''), captions)
	captions = map(lambda x: x.replace('?', ''), captions)
	captions = map(lambda x: x.replace('!', ''), captions)
	captions = map(lambda x: x.replace('\\', ''), captions)
	captions = map(lambda x: x.replace('/', ''), captions)
	return list(captions)

def readWordVocab(util_folder):
	word2ix = np.load(os.path.join(util_folder, 'word2ix.npy'))
	ix2word = np.load(os.path.join(util_folder, 'ix2word.npy'))
	bias_init_vector = np.load(os.path.join(util_folder, 'bias_init_vector.npy'))

	

	return word2ix.tolist(), ix2word.tolist(), bias_init_vector






word2ix, ix2word, bias_init_vector =  readWordVocab('./util_folder/')




model = VideoCaptionGenerator(
			image_dim = image_dim,
			num_of_words = len(word2ix),
			hidden_dim = hidden_dim,
			batch_size = batch_size,
			num_of_lstm_steps = num_of_frames,
			num_of_video_lstm_steps = num_of_video_lstm_steps,
			num_of_caption_lstm_steps = num_of_caption_lstm_steps,
			bias_init_vector = bias_init_vector
		)


video_feature_tf, words_tf = model.build_generator()
sess = tf.InteractiveSession()
saver = tf.train.Saver()

# read testing features
test_video_names = []
for idx, v in enumerate(test_id.id):
	v_dir = TEST_VIDEO_DIR + v + '.npy'
	test_video_names.append(v)
	test_feature.append(np.load(v_dir))
test_feature = np.array(test_feature)

ix2word_series = pd.Series(np.load(os.path.join('./util_folder/', 'ix2word.npy')).tolist())


id_list = []
test_sents = []
model_name = 'modelhaha-204'
saver.restore(sess, os.path.join('./saved_models',model_name))
for idx, video_feature in enumerate(test_feature):
	
	video_feature = video_feature.reshape(1,num_of_video_lstm_steps, image_dim)
	if video_feature.shape[1] == num_of_frames:
		video_mask = np.ones((video_feature.shape[0], video_feature.shape[1]))
	
	probs_val = sess.run(words_tf, feed_dict={
							video_feature_tf: video_feature
							})
	
	generated_words = ix2word_series[list(probs_val[0])]

	punctuation = np.argmax(np.array(generated_words) == '<eos>') + 1
	generated_words = generated_words[:punctuation]

	generated_sent = ' '.join(generated_words)
	generated_sent = generated_sent.replace('<bos> ', '')
	generated_sent = generated_sent.replace(' <eos> ', '')
	generated_sent = generated_sent.replace('<eos>', '')
	generated_sent = generated_sent.replace('<pad>', '')
	generated_sent = generated_sent.replace('<pad> ', '')
	generated_sent = generated_sent.replace(' <pad> ', '')
	generated_sent = generated_sent.replace(' <unk>', '')

	id_list.append(test_video_names[idx])
	test_sents.append(generated_sent)
	
	#print(generated_sent)
submit = pd.DataFrame(np.array([id_list, test_sents]).T)
submit.to_csv(OUTPUT_FILE, index=False, header=False)


test_answer = json.load(open('./testing_label.json','r'))
result = {}
with open(OUTPUT_FILE,'r') as f:
	for line in f:
		line = line.rstrip()
		comma = line.index(',')
		test_id = line[:comma]
		caption = line[comma+1:]
		result[test_id] = caption

# bleu score
bleu=[]
for item in test_answer:
	score_per_video = []
	captions = [x.rstrip('.') for x in item['caption']]
	score_per_video.append(BLEU(result[item['id']],captions,True))
	bleu.append(score_per_video[0])
average = sum(bleu) / len(bleu)

print("Average bleu score is " + str(average))


'''
plt.figure()
plt_save_dir = './imgs'
plt_save_img_name = str(epoch) + '_belu' +str(average)[2:6]+ '.png'
plt.plot(range(len(bleu_to_draw)), bleu_to_draw, color='g')
plt.grid(True)
plt.savefig(os.path.join(plt_save_dir,plt_save_img_name))
'''