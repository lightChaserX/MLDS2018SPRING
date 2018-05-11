import tensorflow as tf
from tensorflow.python.util import nest
import os
import nltk
import numpy as np
import pickle
import random
from tqdm import tqdm
import math


# parameters
padToken, bosToken, eosToken, unkToken = 0, 1, 2, 3
hidden_dim = 512
num_of_layers = 2
embedding_size = 128
learning_rate = 0.5
learning_rate_decay_factor = 0.99
batch_size = 100
num_of_epochs = 50
model_dir = './MLDS_hw2_2_model/model/'
steps_per_checkpoint = 100



class Batch:
    def __init__(self):
        self.encoder_inputs = []
        self.encoder_inputs_length = []
        self.decoder_targets = []
        self.decoder_targets_length = []

class Seq2SeqModelForChatBot():
    def __init__(self, hidden_dim, num_of_layers, embedding_size, learning_rate, learning_rate_decay_factor, word2ix, mode,
                 beam_search, beam_size, max_gradient_norm=5.0):
        # when use gradient descent optimizer
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False, dtype=tf.float32)
        self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * learning_rate_decay_factor)
        # when use other optimizers
        #self.learning_rate = learning_rate
        self.embedding_size = embedding_size
        self.hidden_dim = hidden_dim
        self.num_of_layers = num_of_layers
        self.word2ix = word2ix
        self.vocab_size = len(self.word2ix)
        self.mode = mode
        self.beam_search = beam_search
        self.beam_size = beam_size
        self.max_gradient_norm = max_gradient_norm
        self.build_model()

    def create_rnn_cell(self):
        def create_single_rnn_cell():
            single_cell = tf.contrib.rnn.GRUCell(self.hidden_dim)
            cell = tf.contrib.rnn.DropoutWrapper(single_cell, output_keep_prob=self.keep_prob_placeholder)
            return cell
        cell = tf.contrib.rnn.MultiRNNCell([create_single_rnn_cell() for _ in range(self.num_of_layers)])
        return cell

    def build_model(self):
        self.encoder_inputs = tf.placeholder(tf.int32, [None, None], name='encoder_inputs')
        self.encoder_inputs_length = tf.placeholder(tf.int32, [None], name='encoder_inputs_length')

        self.batch_size = tf.placeholder(tf.int32, [], name='batch_size')
        self.keep_prob_placeholder = tf.placeholder(tf.float32, name='keep_prob_placeholder')

        self.decoder_targets = tf.placeholder(tf.int32, [None, None], name='decoder_targets')
        self.decoder_targets_length = tf.placeholder(tf.int32, [None], name='decoder_targets_length')

        self.max_target_sequence_length = tf.reduce_max(self.decoder_targets_length, name='max_target_len')
        self.mask = tf.sequence_mask(self.decoder_targets_length, self.max_target_sequence_length, dtype=tf.float32, name='sequence_masks')

        # encoder 
        with tf.variable_scope('encoder'):
            encoder_cell = self.create_rnn_cell()
            embedding = tf.get_variable('embedding', [self.vocab_size, self.embedding_size])
            encoder_emb_inputs = tf.nn.embedding_lookup(embedding, self.encoder_inputs)
            encoder_outputs, encoder_state = tf.nn.dynamic_rnn(encoder_cell, encoder_emb_inputs,
                                                               sequence_length=self.encoder_inputs_length,
                                                               dtype=tf.float32)

        # decoder
        with tf.variable_scope('decoder'):
            encoder_inputs_length = self.encoder_inputs_length
            if self.beam_search:
                encoder_outputs = tf.contrib.seq2seq.tile_batch(encoder_outputs, multiplier=self.beam_size)
                encoder_state = nest.map_structure(lambda s: tf.contrib.seq2seq.tile_batch(s, self.beam_size), encoder_state)
                encoder_inputs_length = tf.contrib.seq2seq.tile_batch(self.encoder_inputs_length, multiplier=self.beam_size)

            # attention
            #attention_mechanism = tf.contrib.seq2seq.LuongAttention(num_units=self.hidden_dim, memory=encoder_outputs, memory_sequence_length=encoder_inputs_length)
            decoder_cell = self.create_rnn_cell()
            # attention
            #decoder_cell = tf.contrib.seq2seq.AttentionWrapper(cell=decoder_cell, attention_mechanism=attention_mechanism,
                                                               #attention_layer_size=self.hidden_dim, name='Attention_Wrapper')
            
            batch_size = self.batch_size if not self.beam_search else self.batch_size * self.beam_size
            decoder_initial_state = encoder_state
            # attention
            #decoder_initial_state = decoder_cell.zero_state(batch_size=batch_size, dtype=tf.float32).clone(cell_state=encoder_state)
            
            output_layer = tf.layers.Dense(self.vocab_size, kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

            if self.mode == 'train':
                ending = tf.strided_slice(self.decoder_targets, [0, 0], [self.batch_size, -1], [1, 1])
                decoder_input = tf.concat([tf.fill([self.batch_size, 1], self.word2ix['<BOS>']), ending], 1)
                decoder_emb_inputs = tf.nn.embedding_lookup(embedding, decoder_input)
                training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=decoder_emb_inputs,
                                                                    sequence_length=self.decoder_targets_length,
                                                                    time_major=False, name='training_helper')
                training_decoder = tf.contrib.seq2seq.BasicDecoder(cell=decoder_cell, helper=training_helper,
                                                                   initial_state=encoder_state, output_layer=output_layer)

                # decoder_outputs = (rnn_outputs, sample_id)
                # rnn_outputs: save the probability of each word at each time t, which can be used to calculate loss
                # sample_id: save final decoding results, to show answers 
                decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder=training_decoder,
                                                                            impute_finished=True,
                                                                            maximum_iterations=self.max_target_sequence_length)
                
                self.decoder_logits_train = tf.identity(decoder_outputs.rnn_output)
                self.decoder_predict_train = tf.argmax(self.decoder_logits_train, axis=-1, name='decoder_pred_train')
                
                self.loss = tf.contrib.seq2seq.sequence_loss(logits=self.decoder_logits_train,
                                                             targets=self.decoder_targets, weights=self.mask)


                # optimizer
                #optimizer = tf.train.MomentumOptimizer(self.learning_rate, 0.9)
                #optimizer = tf.train.AdamOptimizer()
                #optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
                optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)

                trainable_params = tf.trainable_variables()
                gradients = tf.gradients(self.loss, trainable_params)
                clip_gradients, _ = tf.clip_by_global_norm(gradients, self.max_gradient_norm)
                self.train_op = optimizer.apply_gradients(zip(clip_gradients, trainable_params))

            elif self.mode == 'decode':
                start_tokens = tf.ones([self.batch_size, ], tf.int32) * self.word2ix['<BOS>']
                end_token = self.word2ix['<EOS>']

                if self.beam_search:
                    inference_decoder = tf.contrib.seq2seq.BeamSearchDecoder(cell=decoder_cell, embedding=embedding,
                                                                             start_tokens=start_tokens, end_token=end_token,
                                                                             initial_state=decoder_initial_state,
                                                                             beam_width=self.beam_size,
                                                                             output_layer=output_layer)
                else:
                    decoding_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding=embedding,
                                                                               start_tokens=start_tokens, end_token=end_token)
                    inference_decoder = tf.contrib.seq2seq.BasicDecoder(cell=decoder_cell, helper=decoding_helper,
                                                                        initial_state=decoder_initial_state,
                                                                        output_layer=output_layer)
                decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder=inference_decoder,maximum_iterations=10)
                
                if self.beam_search:
                    # decoder_outpus = (predicted_ids, beam_search_decoder_output)
                    self.decoder_predict_decode = decoder_outputs.predicted_ids
                else:
                    # decoder_outpus = (rnn_outputs, sample_id)
                    self.decoder_predict_decode = tf.expand_dims(decoder_outputs.sample_id, -1)
        
        self.saver = tf.train.Saver(tf.global_variables(),max_to_keep=10)

    def train(self, sess, batch):
        feed_dict = {self.encoder_inputs: batch.encoder_inputs,
                      self.encoder_inputs_length: batch.encoder_inputs_length,
                      self.decoder_targets: batch.decoder_targets,
                      self.decoder_targets_length: batch.decoder_targets_length,
                      self.keep_prob_placeholder: 0.5,
                      self.batch_size: len(batch.encoder_inputs)}
        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed_dict)
        return loss

    def eval(self, sess, batch):
        feed_dict = {self.encoder_inputs: batch.encoder_inputs,
                      self.encoder_inputs_length: batch.encoder_inputs_length,
                      self.decoder_targets: batch.decoder_targets,
                      self.decoder_targets_length: batch.decoder_targets_length,
                      self.keep_prob_placeholder: 1.0,
                      self.batch_size: len(batch.encoder_inputs)}
        loss = sess.run([self.loss], feed_dict=feed_dict)
        return loss

    def infer(self, sess, batch):
        feed_dict = {self.encoder_inputs: batch.encoder_inputs,
                      self.encoder_inputs_length: batch.encoder_inputs_length,
                      self.keep_prob_placeholder: 1.0,
                      self.batch_size: len(batch.encoder_inputs)}
        predict = sess.run([self.decoder_predict_decode], feed_dict=feed_dict)
        return predict


def buildWordVocab(convs,util_folder):
    word2ix = {}
    words_count = {}
    for conv in convs:
        for sent in conv:
            for word in sent.split(' '):
                words_count[word] = words_count.get(word, 0) + 1

    vocab = [word for word in sorted(words_count) if words_count[word] >= 20]
    print(len(vocab))
    ix2word = {}
    ix2word[0] = '<PAD>'
    ix2word[1] = '<BOS>'
    ix2word[2] = '<EOS>'
    ix2word[3] = '<UNK>'

    word2ix = {}
    word2ix['<PAD>'] = 0
    word2ix['<BOS>'] = 1
    word2ix['<EOS>'] = 2
    word2ix['<UNK>'] = 3

    for i, voc in enumerate(vocab):
        word2ix[voc] = i+4
        ix2word[i+4] = voc

    np.save(os.path.join(util_folder, 'word2ix'), word2ix)
    np.save(os.path.join(util_folder, 'ix2word'), ix2word)

def load_data(filename):
    conv_lines = [line.replace('\n','') for line in open(filename, 'r',encoding='utf-8',errors='ignore')]
    convs = []
    conv = []
    for index, _line in enumerate(conv_lines):
        if _line != '+++$+++':
            conv.append(_line)
        else:
            convs.append(conv)
            conv = []   
    return convs


def createBatch(samples):
    batch = Batch()

    batch.encoder_inputs_length = [len(sample[0]) for sample in samples]
    batch.decoder_targets_length = [len(sample[1]) for sample in samples]

    max_source_length = max(batch.encoder_inputs_length)
    max_target_length = max(batch.decoder_targets_length)

    for sample in samples:
        # padding the source and reverse as encoder inputs
        source = list(reversed(sample[0]))
        pad = [padToken] * (max_source_length - len(source))
        batch.encoder_inputs.append(pad + source)

        #padding target as decoder inputs
        target = list(sample[1])
        pad = [padToken] * (max_target_length - len(target))        
        batch.decoder_targets.append(target + pad)

    return batch


def getBatches(data, batch_size):
    random.shuffle(data)
    batches = []
    data_len = len(data)
    def genNextSamples():
        for i in range(0, data_len, batch_size):
            yield data[i:min(i + batch_size, data_len)]

    for samples in genNextSamples():
        batch = createBatch(samples)
        batches.append(batch)
    return batches

def sentence2ixs(sentence, word2ix):
    if sentence == '':
        return None
    tokens = list(sentence)
    wordixs = []
    for token in tokens:
        wordixs.append(word2ix.get(token, unkToken))
    batch = createBatch([[wordixs, []]])
    return batch

def readWordVocab(util_folder):
    word2ix = np.load(os.path.join(util_folder, 'word2ix.npy'))
    ix2word = np.load(os.path.join(util_folder, 'ix2word.npy'))

    return word2ix.tolist(), ix2word.tolist()

def get_data_set(convs):
    questions = []
    answers = []
    for conv in convs:
        for i in range(len(conv)-1):
            questions.append(conv[i])
            answers.append(conv[i+1])
    #print(len(questions))
    #print(len(answers))
    return questions, answers

def train_model():
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    word2ix, ix2word = readWordVocab('./util_folder/')
    convs = load_data('./clr_conversation.txt')
    questions, answers = get_data_set(convs)

    questions_int = []
    for question in questions:
        ints = []
        for word in question.split():
            if word not in word2ix:
                ints.append(word2ix['<UNK>'])
            else:
                ints.append(word2ix[word])
        questions_int.append(ints)

    answers_int = []
    for answer in answers:
        ints = []
        for word in answer.split():
            if word not in word2ix:
                ints.append(word2ix['<UNK>'])
            else:
                ints.append(word2ix[word])
        answers_int.append(ints)

    max_line_length = 20
    sorted_questions = []
    sorted_answers = []
    for length in range(1, max_line_length+1):
        for i in enumerate(questions_int):
            if len(i[1]) == length:
                sorted_questions.append(questions_int[i[0]])
                sorted_answers.append(answers_int[i[0]])

    trainingSamples = []
    for i in range(len(sorted_questions)):
        conv = []
        conv.append(sorted_questions[i])
        conv.append(sorted_answers[i])
        trainingSamples.append(conv)

    loss_fd = open('loss_record.txt', 'w')

    
    # training
    sess = tf.InteractiveSession()
    model = Seq2SeqModelForChatBot(hidden_dim, num_of_layers, embedding_size, learning_rate, learning_rate_decay_factor, word2ix,
                        mode='train', beam_search=False, beam_size=5, max_gradient_norm=5.0)

    sess.run(tf.global_variables_initializer())
    current_step = 0

    for epoch in range(num_of_epochs):
        print("----- Epoch {}/{} -----".format(epoch + 1, num_of_epochs))
        batches = getBatches(trainingSamples, batch_size)
        for nextBatch in tqdm(batches, desc="Training"):
            loss = model.train(sess, nextBatch)
            current_step += 1
            if current_step % steps_per_checkpoint == 0:
                perplexity = math.exp(float(loss)) if loss < 300 else float('inf')
                #tqdm.write("----- Step %d -- Loss %.2f -- Perplexity %.2f" % (current_step, loss, perplexity))
                print("----- Step %d -- Loss %.2f -- Perplexity %.2f" % (current_step, loss, perplexity))
                loss_fd.write("----- Step %d -- Loss %.2f -- Perplexity %.2f" % (current_step, loss, perplexity)+'\n')
                
                model_name = 'model'+str(epoch)
                checkpoint_path = os.path.join(model_dir, model_name)
                model.saver.save(sess, checkpoint_path, global_step=current_step)

def predict_ids_to_seq(predict_ids, ix2word, beam_szie):
    single_predict = predict_ids[0]
    predict_list = np.ndarray.tolist(single_predict[:, :, 0])
    predict_seq = [ix2word[idx] for idx in predict_list[0]]
    return predict_seq

def readTestingData(filename):
    testing_data = [line.replace('\n','') for line in open(filename, 'r',encoding='utf-8',errors='ignore')]
    return testing_data




def inference():
    word2ix, ix2word = readWordVocab('./util_folder/')
    sentence_list = readTestingData(sys.argv[1])


    # inference
    with tf.Session() as sess:
        model = Seq2SeqModelForChatBot(hidden_dim, num_of_layers, embedding_size, learning_rate, learning_rate_decay_factor, word2ix, mode='decode', beam_search=True, beam_size=5, max_gradient_norm=5.0)
        model_name = 'model0-100'
        model.saver.restore(sess, os.path.join(model_dir,model_name))


        answers_list = []
        for idx, sentence in enumerate(sentence_list):
            batch = sentence2ixs(sentence, word2ix)
            predicted_ids = model.infer(sess, batch)
            answers_list.append(predict_ids_to_seq(predicted_ids, ix2word, 5))
            

        f = open(sys.argv[2],'w',encoding='utf-8')
        for single_answer in answers_list:
            f.write(" ".join(single_answer)+'\n')
        


inference()