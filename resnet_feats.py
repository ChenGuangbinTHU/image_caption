
# coding: utf-8



import math
import os
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
import cv2
import skimage
from tensorflow.python.platform import gfile

import tensorflow.python.platform
from keras.preprocessing import sequence
from collections import Counter

# os.environ["CUDA_VISIBLE_DEVICES"] = "9"


model_path = './models/tensorflow'
model_final_path = './models/tf_final'
vgg_path = './data/vgg16.tfmodel'
resnet_path = "./data/resnet/"
meta_name = "ResNet-L152.meta"
ckpt_name = "ResNet-L152.ckpt"





image_path = './data/val2014/'





dim_embed = 256
dim_hidden = 256
dim_in = 2048
batch_size = 1
learning_rate = 0.001
momentum = 0.9
n_epochs = 25





class Caption_Generator():
    def __init__(self, dim_in, dim_embed, dim_hidden, batch_size, n_lstm_steps, n_words, init_b=None):

        self.dim_in = dim_in
        self.dim_embed = dim_embed
        self.dim_hidden = dim_hidden
        self.batch_size = batch_size
        self.n_lstm_steps = n_lstm_steps
        self.n_words = n_words
        
        with tf.device("/cpu:0"):
            self.word_embedding = tf.Variable(tf.random_uniform([self.n_words, self.dim_embed], -0.1, 0.1), name='word_embedding')

        self.embedding_bias = tf.Variable(tf.zeros([dim_embed]), name='embedding_bias')
        
        self.lstm = tf.contrib.rnn.BasicLSTMCell(dim_hidden)
        
        self.img_embedding = tf.Variable(tf.random_uniform([dim_in, dim_hidden], -0.1, 0.1), name='img_embedding')
        self.img_embedding_bias = tf.Variable(tf.zeros([dim_hidden]), name='img_embedding_bias')

        self.word_encoding = tf.Variable(tf.random_uniform([dim_hidden, n_words], -0.1, 0.1), name='word_encoding')
        
        if init_b is not None:
            self.word_encoding_bias = tf.Variable(init_b, name='word_encoding_bias')
        else:
            self.word_encoding_bias = tf.Variable(tf.zeros([n_words]), name='word_encoding_bias')

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
                        current_embedding = tf.nn.embedding_lookup(self.word_embedding, caption_placeholder[:,i-1]) + self.embedding_bias
                else:
                    current_embedding = image_embedding
                if i > 0: 
                    tf.get_variable_scope().reuse_variables()

                out, state = self.lstm(current_embedding, state)

                
                if i > 0:
                    labels = tf.expand_dims(caption_placeholder[:, i], 1)
                    ix_range=tf.range(0, self.batch_size, 1)
                    ixs = tf.expand_dims(ix_range, 1)
                    concat = tf.concat([ixs, labels],1)
                    onehot = tf.sparse_to_dense(
                            concat, tf.stack([self.batch_size, self.n_words]), 1.0, 0.0)


                    logit = tf.matmul(out, self.word_encoding) + self.word_encoding_bias
                    xentropy = tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=onehot)
                    xentropy = xentropy * mask[:,i]

                    loss = tf.reduce_sum(xentropy)
                    total_loss += loss

            total_loss = total_loss / tf.reduce_sum(mask[:,1:])
            return total_loss, img,  caption_placeholder, mask


    def build_generator(self, maxlen, batchsize=1):
        img = tf.placeholder(tf.float32, [self.batch_size, self.dim_in])
        image_embedding = tf.matmul(img, self.img_embedding) + self.img_embedding_bias
        state = self.lstm.zero_state(batchsize,dtype=tf.float32)

        all_words = []
        with tf.variable_scope("RNN"):
            output, state = self.lstm(image_embedding, state)
            previous_word = tf.nn.embedding_lookup(self.word_embedding, [0]) + self.embedding_bias

            for i in range(maxlen):
                tf.get_variable_scope().reuse_variables()

                out, state = self.lstm(previous_word, state)


                logit = tf.matmul(out, self.word_encoding) + self.word_encoding_bias
                best_word = tf.argmax(logit, 1)

                with tf.device("/cpu:0"):
                    previous_word = tf.nn.embedding_lookup(self.word_embedding, best_word)

                previous_word += self.embedding_bias

                all_words.append(best_word)

        return img, all_words


if not os.path.exists('data/r_ixtoword.npy'):
    print('You must run training first.')
else:
    ixtoword = np.load('data/r_ixtoword.npy').tolist()
    n_words = len(ixtoword)
    maxlen = 15
    graph = tf.get_default_graph()
    sess = tf.InteractiveSession(graph=graph)

    print("loading meta/ckpt...")
    saver = tf.train.import_meta_graph(resnet_path + meta_name)
    saver.restore(sess, resnet_path + ckpt_name)
    '''os.system("rm -rf /tmp/load")
    tf.train.write_graph(sess.graph_def, "/tmp/load", "test.pb", False)

    # tf.reset_default_graph()
    print("loading graph...")
    with gfile.FastGFile("/tmp/load/test.pb", 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
    '''
    
    print("Init generator...")
    caption_generator = Caption_Generator(dim_in, dim_hidden, dim_embed, batch_size, maxlen+2, n_words)
    graph = tf.get_default_graph()
    '''
    mylist = graph.get_operations()
    import pprint
    pp = pprint.PrettyPrinter()
    pp.pprint(mylist)
    '''
    image, generated_words = caption_generator.build_generator(maxlen=maxlen)


def crop_image(x, target_height=227, target_width=227, as_float=True):
    image = cv2.imread(x)
    if as_float:
        image = image.astype(np.float32)

    if len(image.shape) == 2:
        image = np.tile(image[:,:,None], 3)
    elif len(image.shape) == 4:
        image = image[:,:,:,0]

    height, width, rgb = image.shape
    if width == height:
        resized_image = cv2.resize(image, (target_height,target_width))

    elif height < width:
        resized_image = cv2.resize(image, (int(width * float(target_height)/height), target_width))
        cropping_length = int((resized_image.shape[1] - target_height) / 2)
        resized_image = resized_image[:,cropping_length:resized_image.shape[1] - cropping_length]

    else:
        resized_image = cv2.resize(image, (target_height, int(height * float(target_width) / width)))
        cropping_length = int((resized_image.shape[0] - target_width) / 2)
        resized_image = resized_image[cropping_length:resized_image.shape[0] - cropping_length,:]

    return cv2.resize(resized_image, (target_height, target_width))



def read_image(path):

     img = crop_image(path, target_height=224, target_width=224)
     if img.shape[2] == 4:
         img = img[:,:,:3]

     img = img[None, ...]
     return img



def test(sess,image,generated_words,ixtoword): # Naive greedy search
    
    feats = []  # n_samples * 2048
    files = os.listdir(image_path)
    files = sorted(files)
    # for filename in files:
    #     print(filename)
    # exit(2333)
    cnt = 0
    for filename in files:
        cnt += 1
        if cnt % 100 == 0:
            print float(cnt) / 400.0, " finished"
        if cnt == 4042:
            break
        child = os.path.join(image_path, filename)
        if os.path.isfile(child):
            feat = read_image(child)
            fc7 = sess.run(graph.get_tensor_by_name("avg_pool:0"), feed_dict={'images:0': feat})
            # print(fc7)
            # print(fc7.shape)
            # print("Success!")
            # exit(2333)
            feats.append(fc7)
        if cnt == 2:
            break
    if (feats[0]==feats[1]).all():
        print('Fail..')
    else:
        print('Success!')
    # exit(233)
    np.save("./data/feats_resnet152_val.npy", feats)
    print("feats.shape =", np.array(feats).shape)
    
    '''
    child = './giraffe.jpg'
    feats = read_image(child)
    fc7 = sess.run(graph.get_tensor_by_name("avg_pool:0"), feed_dict={'images:0': feats})
    np.save("./giraffe.npy", fc7)
    print("fc7.shape =", np.array(fc7).shape)
    '''
    
    # feat = read_image(test_image_path)
    # # feat = np.zeros((1, 224, 224, 3), dtype="float32")
    # fc7 = sess.run(graph.get_tensor_by_name("scale5/block3/c/beta:0"), feed_dict={images:feat})
    # print(fc7)
    # print(fc7.shape)
    # print("yeaaaah!")

    # saver = tf.train.Saver()
    # sanity_check=False
    # # sanity_check=True
    # if not sanity_check:
    #     saved_path=tf.train.latest_checkpoint(model_path)
    #     saver.restore(sess, saved_path)
    # else:
    #     tf.global_variables_initializer().run()
    #
    # generated_word_index= sess.run(generated_words, feed_dict={image: fc7})
    # generated_word_index = np.hstack(generated_word_index)
    # generated_words = [ixtoword[x] for x in generated_word_index]
    # punctuation = np.argmax(np.array(generated_words) == '.')+1
    #
    # generated_words = generated_words[:punctuation]
    # generated_sentence = ' '.join(generated_words)
    # print(generated_sentence)


# In[ ]:


test(sess,image,generated_words,ixtoword)

