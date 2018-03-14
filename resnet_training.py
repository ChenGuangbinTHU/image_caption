# coding: utf-8



import math
import os
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
# import cv2
# import skimage
import pickle as pkl

import tensorflow.python.platform
from keras.preprocessing import sequence
from collections import Counter
import datetime



os.environ["CUDA_VISIBLE_DEVICES"] = "3, 6"


model_path = './models/resnet50_save'
model_path_transfer = './models/resnet50_final'
feature_path = './data/feats_resnet50.npy'
annotation_path = './data/caption_2.token'




def get_data(annotation_path, feature_path):
    annotations = pd.read_table(annotation_path, sep='\t', header=None, names=['caption'])
    return np.load(feature_path, 'r'), annotations['caption'].values





# feats, captions = get_data(annotation_path, feature_path)




# print(feats.shape)
# print(captions.shape)
# exit(233)




# print(captions[0])





def preProBuildWordVocab(sentence_iterator, word_count_threshold=30):  
    print('preprocessing %d word vocab' % (word_count_threshold,))
    word_counts = {}
    nsents = 0
    for sent in sentence_iterator:
        nsents += 1
        for w in sent.lower().split(' '):
            word_counts[w] = word_counts.get(w, 0) + 1
    vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]
    print('preprocessed words %d -> %d' % (len(word_counts), len(vocab)))

    # ix = index
    ixtoword = {}
    ixtoword[0] = '.'
    wordtoix = {}
    wordtoix['#START#'] = 0
    ix = 1
    for w in vocab:
        wordtoix[w] = ix
        ixtoword[ix] = w
        ix += 1

    word_counts['.'] = nsents
    bias_init_vector = np.array([1.0 * word_counts[ixtoword[i]] for i in ixtoword])
    bias_init_vector /= np.sum(bias_init_vector)
    bias_init_vector = np.log(bias_init_vector)
    bias_init_vector -= np.max(bias_init_vector)
    return wordtoix, ixtoword, bias_init_vector.astype(np.float32)





class Caption_Generator():
    def __init__(self, dim_in, dim_embed, dim_hidden, batch_size, n_lstm_steps, n_words, init_b):

        self.dim_in = dim_in  # 4096
        self.dim_embed = dim_embed  # 256
        self.dim_hidden = dim_hidden  # 256
        self.batch_size = batch_size
        self.n_lstm_steps = n_lstm_steps  # maxlen+2
        self.n_words = n_words

        with tf.device("/cpu:0"):
            self.word_embedding = tf.Variable(tf.random_uniform([self.n_words, self.dim_embed], -0.1, 0.1),
                                              name='word_embedding')

        self.embedding_bias = tf.Variable(tf.zeros([dim_embed]), name='embedding_bias')

        self.lstm = tf.contrib.rnn.BasicLSTMCell(dim_hidden)

        self.img_embedding = tf.Variable(tf.random_uniform([dim_in, dim_hidden], -0.1, 0.1), name='img_embedding')
        self.img_embedding_bias = tf.Variable(tf.zeros([dim_hidden]), name='img_embedding_bias')

        self.word_encoding = tf.Variable(tf.random_uniform([dim_hidden, n_words], -0.1, 0.1), name='word_encoding')
        self.word_encoding_bias = tf.Variable(init_b, name='word_encoding_bias')

    def build_model(self):
        img = tf.placeholder(tf.float32, [self.batch_size, self.dim_in])
        caption_placeholder = tf.placeholder(tf.int32, [self.batch_size, self.n_lstm_steps])
        mask = tf.placeholder(tf.float32, [self.batch_size, self.n_lstm_steps])

        image_embedding = tf.matmul(img, self.img_embedding) + self.img_embedding_bias

        state = self.lstm.zero_state(self.batch_size, dtype=tf.float32)

        total_loss = 0.0
        with tf.variable_scope("RNN"):
            for i in range(self.n_lstm_steps):
                if i > 0:
                    with tf.device("/cpu:0"):
                        current_embedding = tf.nn.embedding_lookup(self.word_embedding,
                                                                   caption_placeholder[:, i - 1]) + self.embedding_bias
                else:
                    current_embedding = image_embedding
                if i > 0:
                    tf.get_variable_scope().reuse_variables()

                out, state = self.lstm(current_embedding, state)

                if i > 0:
                    labels = tf.expand_dims(caption_placeholder[:, i], 1)
                    ix_range = tf.range(0, self.batch_size, 1)
                    ixs = tf.expand_dims(ix_range, 1)
                    concat = tf.concat([ixs, labels], 1)
                    onehot = tf.sparse_to_dense(
                        concat, tf.stack([self.batch_size, self.n_words]), 1.0, 0.0)

                    logit = tf.matmul(out, self.word_encoding) + self.word_encoding_bias
                    xentropy = tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=onehot)
                    xentropy = xentropy * mask[:, i]

                    loss = tf.reduce_sum(xentropy)
                    total_loss += loss

            total_loss = total_loss / tf.reduce_sum(mask[:, 1:])
            return total_loss, img, caption_placeholder, mask

dim_embed = 256
dim_hidden = 256
dim_in = 2048
batch_size = 128
momentum = 0.0
n_epochs = 100
training_size = 80000


def train(learning_rate=0.001, continue_training=False, transfer=True):
    tf.reset_default_graph()

    feats, captions = get_data(annotation_path, feature_path)
    tmp = feats.shape
    feats = feats.reshape((tmp[0], tmp[2]))
    wordtoix, ixtoword, init_b = preProBuildWordVocab(captions)
    print(feats.shape)
    print(captions.shape)
#    exit(233)
    feats = feats[:training_size]
    captions = captions[:training_size]
    
    # np.save('data/ixtoword', ixtoword)

    index = (np.arange(len(feats)).astype(int))
    np.random.shuffle(index)

    sess = tf.InteractiveSession()
    n_words = len(wordtoix)
    maxlen = np.max([x for x in map(lambda x: len(x.split(' ')), captions)])
    caption_generator = Caption_Generator(dim_in, dim_hidden, dim_embed, batch_size, maxlen + 2, n_words, init_b)

    loss, image, sentence, mask = caption_generator.build_model()

    saver = tf.train.Saver(max_to_keep=100)
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(learning_rate, global_step,
                                               int(len(index) / batch_size), 0.95)
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    tf.global_variables_initializer().run()

    if continue_training:
        if not transfer:
            saver.restore(sess, tf.train.latest_checkpoint(model_path))
        else:
            saver.restore(sess, tf.train.latest_checkpoint(model_path_transfer))
    losses = []
    for epoch in range(n_epochs):
        for start, end in zip(range(0, len(index), batch_size), range(batch_size, len(index), batch_size)):

            current_feats = feats[index[start:end]]
            current_captions = captions[index[start:end]]
            current_caption_ind = [x for x in map(
                lambda cap: [wordtoix[word] for word in cap.lower().split(' ')[:-1] if word in wordtoix],
                current_captions)]

            current_caption_matrix = sequence.pad_sequences(current_caption_ind, padding='post', maxlen=maxlen + 1)
            current_caption_matrix = np.hstack([np.full((len(current_caption_matrix), 1), 0), current_caption_matrix])

            current_mask_matrix = np.zeros((current_caption_matrix.shape[0], current_caption_matrix.shape[1]))
            nonzeros = np.array([x for x in map(lambda x: (x != 0).sum() + 2, current_caption_matrix)])

            for ind, row in enumerate(current_mask_matrix):
                row[:nonzeros[ind]] = 1

            _, loss_value = sess.run([train_op, loss], feed_dict={
                image: current_feats.astype(np.float32),
                sentence: current_caption_matrix.astype(np.int32),
                mask: current_mask_matrix.astype(np.float32)
            })

            print(str(datetime.datetime.now())[:-7], "Loss: ", loss_value, "\t Epoch {}/{}".format(epoch, n_epochs),
                  "\t Iter {}/{}".format(start, len(feats)))
        print("Saving the model from epoch: ", epoch)
        saver.save(sess, os.path.join(model_path, 'model'), global_step=epoch)


# In[ ]:


try:
    # train(.001, False, False)  # train from scratch
    # train(.001, True, True)  # continue training from pretrained weights @epoch500
    train(.001, True, False)  #train from previously saved weights
except KeyboardInterrupt:
    print('Exiting Training')

