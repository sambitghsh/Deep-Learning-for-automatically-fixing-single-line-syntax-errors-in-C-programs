import pandas as pd
import numpy as np
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, Sequential
from keras.layers import GRU, Input, Dense, TimeDistributed, Activation, RepeatVector, Bidirectional, Dropout, LSTM
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.losses import sparse_categorical_crossentropy
import re
from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate.bleu_score import sentence_bleu
import pickle
from Encoder import Encoder
from Decoder import Decoder
from Bahdanauattention import BahdanauAttention
import sys
from tqdm import tqdm
import progressbar


max_length_inp = 25
max_length_targ = 25
embedding_dim = 64
units = 300
BATCH_SIZE = 64

def evaluate(sentence):
    result = []
    sentence = eval(sentence)
    inputs = [0]*len(sentence)

    for i in range(len(sentence)):
        try:
            inputs[i] = token2index[sentence[i]]
        except KeyError:
            inputs[i] = 3

    if(len(inputs) > max_length_inp - 2 ):
        inputs = inputs[:(max_length_inp - 2)]
        
    inputs = [1] + inputs + [2]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs], maxlen=max_length_inp, padding='post')
    inputs = tf.convert_to_tensor(inputs)
    hidden = [tf.zeros((1, units))]
    enc_out, enc_hidden = encoder(inputs, hidden)

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([token2index['SOS']], 0)

    for t in range(max_length_targ):
    
        predictions, dec_hidden, attention_weights = decoder(dec_input, dec_hidden, enc_out)
        predicted_id = tf.argmax(predictions[0]).numpy()
        result.append(index2token[predicted_id])
        
        if predicted_id == token2index['EOS']:
            return result
            
        dec_input = tf.expand_dims([predicted_id], 0)

    return result


def validate():

    result = []
    start = time.time()
    #bar = progressbar.ProgressBar(maxval = val_data.shape[0]).start()
    for i in tqdm(range(val_data.shape[0])): 
    
        valid_source_text = val_data['sourceLineTokens'][i]
        pred_text = evaluate(valid_source_text)
        pred_text_list = pred_text[:-1]
        #bar.update(i)
        if len(pred_text_list)>0 and pred_text_list[-1] == 'OOV':
              pred_text_list = pred_text_list[:-1]
        
        result.append(str(pred_text_list))
    print('\n')
    print("Time Required: ",time.time()-start)
    return result
    
 

    
if __name__ == '__main__':
        
        #load the dictionaries
        with open('index2token.pickle', 'rb') as handle:
                index2token = pickle.load(handle)    
                
        with open('token2index.pickle', 'rb') as handle:
                token2index = pickle.load(handle) 
                
        # load the data and create the required class's objects      
        val_data = pd.read_csv(sys.argv[1])
        vocab_inp_size = vocab_tar_size = len(index2token)
        encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)
        decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE)
        
        #loas the checkpoint
        optimizer = tf.keras.optimizers.Adam()
        checkpoint_dir = './training_checkpoints'
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        checkpoint = tf.train.Checkpoint(step = tf.Variable(1), optimizer = optimizer, encoder = encoder, decoder = decoder)
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
        
        
        #predict and load the result into the csv
        result = validate()
        val_data['fixedTokens'] = result
        val_data.to_csv(r'C:/Users/Lenovo/Desktop/Courses/AESML/final/'+sys.argv[2], index = False, header=True)

    
        