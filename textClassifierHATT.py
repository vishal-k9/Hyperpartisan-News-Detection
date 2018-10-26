# author - Richard Liao
# Dec 26 2016
import numpy as np
import pandas as pd
import cPickle
from collections import defaultdict
import re

from bs4 import BeautifulSoup

import sys
import os


from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Merge, Dropout, LSTM, GRU, Bidirectional, TimeDistributed
from keras.models import Model

from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json

from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras import initializers

MAX_SENT_LENGTH = 100
MAX_SENTS = 50
MAX_NB_WORDS = 40000
EMBEDDING_DIM = 300
# VALIDATION_SPLIT = 0.2


def clean_str(string):
    """
    Tokenization/string cleaning for dataset
    Every dataset is lower cased except
    """
    string = re.sub(r"\\", "", string)
    string = re.sub(r"\'", "", string)
    string = re.sub(r"\"", "", string)
    return string.strip().lower()


# data_train = pd.read_csv('labeledTrainData.tsv', sep='\t')
# print data_train

from nltk import tokenize

reviews = []
labels = []
texts = []

# for idx in range(data_train.review.shape[0]):
#     text = BeautifulSoup(data_train.review[idx])
#     text = clean_str(text.get_text().encode('ascii', 'ignore'))
#     texts.append(text)
#     sentences = tokenize.sent_tokenize(text)
#     reviews.append(sentences)

#     labels.append(data_train.sentiment[idx])



# train_X=[]
# train_y=[]
i=0
with open("../train_data.txt","rb") as f1:
    for row in f1:
        col= row.strip().split()
        filename=col[0]
        with open("../articles-training/"+filename+".xml","rb") as myfile:
            data=myfile.read()
        text = BeautifulSoup(data)
        text = clean_str(text.get_text().encode('ascii', 'ignore'))
        texts.append(text)
        sentences = tokenize.sent_tokenize(text)
        reviews.append(sentences)

          
        # train_X.append(sent2vec(data))
        if col[2].strip()=="true":
            labels.append(1)
        else:
            labels.append(0) 

        i+=1
        if(i%100==0):
            # break
            print("Train: ",i)  
            break

# z=list(zip(train_X,train_y))
# random.shuffle(z)
# train_X,train_y=zip(*z)
# train_y = to_categorical(np.asarray(train_y))

tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)

data = np.zeros((len(texts), MAX_SENTS, MAX_SENT_LENGTH), dtype='int32')






for i, sentences in enumerate(reviews):
    for j, sent in enumerate(sentences):
        if j < MAX_SENTS:
            wordTokens = text_to_word_sequence(sent)
            k = 0
            for _, word in enumerate(wordTokens):
                if k < MAX_SENT_LENGTH and tokenizer.word_index[word] < MAX_NB_WORDS:
                    data[i, j, k] = tokenizer.word_index[word]
                    k = k + 1

word_index = tokenizer.word_index
print('Total %s unique tokens.' % len(word_index))
# print labels[:10]
# print "************************"
labels = to_categorical(np.asarray(labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)
# print labels[:10]
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
# nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])
# print len(data[0][0])              
# exit()
x_train = data
y_train = labels

# dev_X=[]
# dev_y=[]

reviews = []
labels = []
texts = []

i=0
with open("../dev_data.txt","rb") as f1:
    for row in f1:
        col= row.strip().split()
        filename=col[0]
        with open("../articles-validation/"+filename+".xml","rb") as myfile:
            data=myfile.read()
        text = BeautifulSoup(data)
        text = clean_str(text.get_text().encode('ascii', 'ignore'))
        texts.append(text)
        sentences = tokenize.sent_tokenize(text)
        reviews.append(sentences)

          
        # train_X.append(sent2vec(data))
        if col[2].strip()=="true":
            labels.append(1)
        else:
            labels.append(0)

        # dev_X.append(sent2vec(data))
        # if col[2].strip()=="true":
        #     dev_y.append([1,0])
        # else:
        #     dev_y.append([0,1]) 

        i+=1
        if(i%100==0):
            # break
            print("Dev: ",i)
            break

# z=list(zip(dev_X,dev_y))
# random.shuffle(z)
# dev_X,dev_y=zip(*z)
# dev_y = to_categorical(np.asarray(dev_y))
tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)

data = np.zeros((len(texts), MAX_SENTS, MAX_SENT_LENGTH), dtype='int32')






for i, sentences in enumerate(reviews):
    for j, sent in enumerate(sentences):
        if j < MAX_SENTS:
            wordTokens = text_to_word_sequence(sent)
            k = 0
            for _, word in enumerate(wordTokens):
                if k < MAX_SENT_LENGTH and tokenizer.word_index[word] < MAX_NB_WORDS:
                    data[i, j, k] = tokenizer.word_index[word]
                    k = k + 1

word_index = tokenizer.word_index
print('Total %s unique tokens.' % len(word_index))
# print labels[:10]
# print "************************"
labels = to_categorical(np.asarray(labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)
# print labels[:10]
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

x_val = data
y_val = labels

print('Number of positive and negative reviews in traing and validation set')
print y_train.sum(axis=0)
print y_val.sum(axis=0)

GLOVE_DIR = "/home/vishal/Files/Study/Stream-Project/SEM V/word embedding"
embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.840B.300d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Total %s word vectors.' % len(embeddings_index))

embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector


# building Hierachical Attention network
embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SENT_LENGTH,
                            trainable=True,
                            mask_zero=True)


class AttLayer(Layer):
    def __init__(self, attention_dim):
        self.init = initializers.get('normal')
        self.supports_masking = True
        self.attention_dim = attention_dim
        super(AttLayer, self).__init__()

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = K.variable(self.init((input_shape[-1], self.attention_dim)))
        self.b = K.variable(self.init((self.attention_dim, )))
        self.u = K.variable(self.init((self.attention_dim, 1)))
        self.trainable_weights = [self.W, self.b, self.u]
        super(AttLayer, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, x, mask=None):
        # size of x :[batch_size, sel_len, attention_dim]
        # size of u :[batch_size, attention_dim]
        # uit = tanh(xW+b)
        uit = K.tanh(K.bias_add(K.dot(x, self.W), self.b))
        ait = K.dot(uit, self.u)
        ait = K.squeeze(ait, -1)

        ait = K.exp(ait)

        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            ait *= K.cast(mask, K.floatx())
        ait /= K.cast(K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        ait = K.expand_dims(ait)
        weighted_input = x * ait
        output = K.sum(weighted_input, axis=1)

        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])


sentence_input = Input(shape=(MAX_SENT_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sentence_input)
l_lstm = Bidirectional(GRU(100, return_sequences=True))(embedded_sequences)
l_att = AttLayer(100)(l_lstm)
sentEncoder = Model(sentence_input, l_att)

review_input = Input(shape=(MAX_SENTS, MAX_SENT_LENGTH), dtype='int32')
review_encoder = TimeDistributed(sentEncoder)(review_input)
l_lstm_sent = Bidirectional(GRU(100, return_sequences=True))(review_encoder)
l_att_sent = AttLayer(100)(l_lstm_sent)
preds = Dense(2, activation='softmax')(l_att_sent)
model = Model(review_input, preds)

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

print("model fitting - Hierachical attention network")
model.fit(x_train, y_train,
          epochs=1, batch_size=50)
model.evaluate(x_val, y_val, verbose=0)
# print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100.0))
pred=model.predict(x_val, verbose=1)
print (pred)
import tensorflow as tf

# Create a session
sess = tf.InteractiveSession()

indexes = tf.argmax(pred, axis=1)
print(indexes.eval())
# model.predict_classes(x_val, verbose=1)
# # serialize model to JSON
# model_json = model.to_json()
# with open("hatt-model.json", "wb") as json_file:
#     json_file.write(model_json)
# # serialize weights to HDF5
# model.save_weights("model.h5")
# print("Saved model to disk")

# # load json and create model
# json_file = open('hatt-model.json', 'rb')
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(loaded_model_json, custom_objects={'AttLayer': AttLayer()})
# # load weights into new model
# loaded_model.load_weights("model.h5")
# print("Loaded model from disk")
 
# # evaluate loaded model on test data
# loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
# score = loaded_model.evaluate(X, Y, verbose=0)
# print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))