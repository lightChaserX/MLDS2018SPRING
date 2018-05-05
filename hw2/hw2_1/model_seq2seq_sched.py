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

train_feature = []
test_feature = []
X_test = []
y_train = []
y_test = []

DATA_DIR = './MLDS_hw2_1_data'
TRAIN_VIDEO_DIR = 'MLDS_hw2_1_data/training_data/feat/'
TEST_VIDEO_DIR = 'MLDS_hw2_1_data/testing_data/feat/'
TRAIN_LABEL_DIR = 'MLDS_hw2_1_data/training_label.json'
TEST_LABEL_DIR = 'MLDS_hw2_1_data/testing_label.json'
TRAIN_ID_DIR = 'MLDS_hw2_1_data/training_data/id.txt'
TEST_ID_DIR = 'MLDS_hw2_1_data/testing_data/id.txt'
MODEL_SAVE_DIR = './saved_models/'

train_id = pd.read_csv(TRAIN_ID_DIR, header=None, names=['id'])
test_id = pd.read_csv(TEST_ID_DIR, header=None, names=['id'])


image_dim = 4096
hidden_dim = 256
num_of_video_lstm_steps = 80
num_of_caption_lstm_steps = 20
num_of_frames = 80
num_of_epochs = 205
batch_size = 50
learning_rate = 0.001
decay_epoch = 30


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

		

		with tf.variable_scope("embedding"):
			embedding_decoder = tf.Variable(tf.truncated_normal(shape=[ self.num_of_words,  self.hidden_dim], stddev=0.1), name='embedding_decoder')
			decoder_emb_inp = tf.nn.embedding_lookup(embedding_decoder, caption[:,:-1])

		
		decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_dim)
		
		decoder_seq_length = [self.num_of_caption_lstm_steps+1] * self.batch_size

		helper = tf.contrib.seq2seq.ScheduledEmbeddingTrainingHelper(decoder_emb_inp, decoder_seq_length, embedding_decoder, 0.2, time_major=False)

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


def buildWordVocab (labels, util_folder, word_count_threshold=5):
	print('Preprocessing & creating vocab')
	words_count = {}
	num_of_sents = 0

	for label in labels:
		captions = label['caption']
		processed_captions = processCaptions(captions)
		for sentence in processed_captions:
			num_of_sents += 1
			for word in sentence.lower().split(' '):
				words_count[word] = words_count.get(word, 0) + 1
    		
	vocab = [word for word in sorted(words_count) if words_count[word] >= word_count_threshold]
	print('filtered words from %d to %d' % (len(words_count), len(vocab)))

	ix2word = {}
	ix2word[0] = '<pad>'
	ix2word[1] = '<bos>'
	ix2word[2] = '<eos>'
	ix2word[3] = '<unk>'

	word2ix = {}
	word2ix['<pad>'] = 0
	word2ix['<bos>'] = 1
	word2ix['<eos>'] = 2
	word2ix['<unk>'] = 3

	for i, voc in enumerate(vocab):
		word2ix[voc] = i+4
		ix2word[i+4] = voc

	words_count['<pad>'] = num_of_sents
	words_count['<bos>'] = num_of_sents
	words_count['<eos>'] = num_of_sents
	words_count['<unk>'] = num_of_sents

	# normalize 
	bias_init_vector = np.array([1.0*words_count[ix2word[i]] for i in ix2word])
	bias_init_vector /= np.sum(bias_init_vector)
	bias_init_vector = np.log(bias_init_vector)
	bias_init_vector -= np.max(bias_init_vector)

	if not os.path.exists(util_folder):
		os.makedirs(util_folder)

	np.save(os.path.join(util_folder, 'word2ix'), word2ix)
	np.save(os.path.join(util_folder, 'ix2word'), ix2word)
	np.save(os.path.join(util_folder, 'bias_init_vector'), bias_init_vector)

	return word2ix, bias_init_vector

# Building word vocab
with open(TRAIN_LABEL_DIR) as f:
	train_label = json.load(f)
with open(TEST_LABEL_DIR) as f:
	test_label = json.load(f)

word2ix, bias_init_vector =  buildWordVocab(train_label+test_label, './util_folder/', 2)


# read training features
for idx, v in enumerate(train_id.id):
	v_dir = TRAIN_VIDEO_DIR + v + '.npy'
	train_feature.append(np.load(v_dir))
train_feature = np.array(train_feature)


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

tf_loss, tf_video_feature, tf_caption, tf_caption_mask,tf_output_sample_id, tf_probs = model.build_model()
params = tf.trainable_variables()

gradients = tf.gradients(tf_loss, params)
clipped_gradients, _ = tf.clip_by_global_norm(gradients, 5)

# optimization
optimizer = tf.train.AdamOptimizer(learning_rate)
train_op = optimizer.apply_gradients(zip(clipped_gradients, params))

sess = tf.InteractiveSession()
saver = tf.train.Saver(max_to_keep=250)
tf.global_variables_initializer().run()


# read testing features
test_video_names = []
for idx, v in enumerate(test_id.id):
	v_dir = TEST_VIDEO_DIR + v + '.npy'
	test_video_names.append(v)
	test_feature.append(np.load(v_dir))
test_feature = np.array(test_feature)

ix2word_series = pd.Series(np.load(os.path.join('./util_folder/', 'ix2word.npy')).tolist())







loss_fd = open('loss_record.txt', 'w')
loss_to_draw = []
bleu_to_draw = []
loss_to_draw_epoch = []


for epoch in range(num_of_epochs):
	for start, end in zip(range(0,len(train_feature), batch_size), range(batch_size, len(train_feature), batch_size)):
		
		start_time = time.time()
		current_features = train_feature[start:end]
		current_video_masks = np.zeros((batch_size, num_of_video_lstm_steps))
		current_captions = []

		for i in range(len(current_features)):
			current_video_masks[i][:len(current_features[i])] = 1
			current_captions.append(random.choice(train_label[start+i]['caption']))

		current_captions = list(map(lambda x: '<bos> ' + x, current_captions))
		current_captions = processCaptions(current_captions)
		current_captions = list(current_captions)

		current_captions_src = []
		for idx, single_caption in enumerate(current_captions):
			words = single_caption.lower().split(' ')
			if len(words) < num_of_caption_lstm_steps+1:
				current_captions[idx] += ' <eos>'
			else:
				new_word = ''
				for i in range(num_of_caption_lstm_steps):
					new_word += (words[i] + ' ')
				current_captions[idx] = new_word + '<eos>'

		current_captions_index =[]
		for caption in current_captions:
			current_words_index = []
			for word in caption.lower().split(' '):
				if word in word2ix:
					current_words_index.append(word2ix[word])
				else:
					current_words_index.append(word2ix['<unk>'])
			current_captions_index.append(current_words_index)

		current_caption_matrix = sequence.pad_sequences(current_captions_index, padding='post', maxlen=num_of_caption_lstm_steps+1)
		current_caption_matrix = np.hstack([current_caption_matrix, np.zeros([len(current_caption_matrix), 1] ) ]).astype(int)
		current_caption_masks =  np.zeros( (current_caption_matrix.shape[0], current_caption_matrix.shape[1]-1) )

		nonzeros = np.array(list(map(lambda x: (x!=0).sum()+1, current_caption_matrix)))
		
		for idx, row in enumerate(current_caption_masks):
			row[:nonzeros[idx]] = 1

		_, loss_val = sess.run([train_op, tf_loss], feed_dict={
					tf_video_feature: current_features,
					tf_caption: current_caption_matrix,
					tf_caption_mask: current_caption_masks
					})

		loss_to_draw.append(loss_val)
		loss_fd.write('epoch '+ str(epoch) + ' loss '+str(loss_val)+'\n')

		print (" Epoch: ", epoch, " loss: ", loss_val, ' Elapsed time: ', str((time.time() - start_time)))

	test_sents = []
	id_list = []

	
	print("Epoch ", epoch, " is done. Saving the model ...")
	saver.save(sess, os.path.join(MODEL_SAVE_DIR, 'model'+str('haha')), global_step=epoch)

	'''
	plt.figure()
	loss_to_draw_epoch.append(np.mean(loss_to_draw))
	plt_save_dir = './imgs'
	plt_save_img_name = str(epoch) + '.png'
	plt.plot(range(len(loss_to_draw_epoch)), loss_to_draw_epoch, color='r')
	plt.grid(True)
	plt.savefig(os.path.join(plt_save_dir,plt_save_img_name))
	plt.figure()
	plt_save_dir = './imgs'
	plt_save_img_name = str(epoch) + '_belu' + '.png'
	plt.plot(range(len(bleu_to_draw)), bleu_to_draw, color='g')
	plt.grid(True)
	plt.savefig(os.path.join(plt_save_dir,plt_save_img_name))
	'''
