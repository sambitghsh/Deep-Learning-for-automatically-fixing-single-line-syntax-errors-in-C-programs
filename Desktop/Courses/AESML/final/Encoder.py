import tensorflow as tf
import numpy as np
import time
from keras.preprocessing.text import Tokenizer
from keras.models import Model, Sequential
from keras.layers import GRU, Input, Dense, TimeDistributed, Activation, RepeatVector, Bidirectional, Dropout, LSTM
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.losses import sparse_categorical_crossentropy
import re
from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate.bleu_score import sentence_bleu


#Creating the encoding class 
class Encoder(tf.keras.Model):

    def __init__(self, vocab_size, embedding_dim, encoder_units, batch_size):
    
        super(Encoder, self).__init__()
        self.batch_size = batch_size
        self.encoder_units = encoder_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.encoder_units, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')

    def __call__(self, x, hidden):
    
        x = self.embedding(x)
        output, state = self.gru(x, initial_state=hidden)
        return output, state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_size, self.encoder_units))