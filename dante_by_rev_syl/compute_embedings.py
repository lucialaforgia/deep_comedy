import os
import sys
sys.path.append(os.path.abspath(".."))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

import csv
import string
from utils import save_vocab, load_vocab

working_dir = os.path.abspath('.')

# Path where the vocab is saved
logs_dir = os.path.join(working_dir, 'logs')
os.makedirs(logs_dir, exist_ok = True) 
vocab_file = os.path.join(working_dir, 'logs', 'vocab.json')

vocab, idx2syl, syl2idx = load_vocab(vocab_file)

# Path where the model is saved
models_dir = os.path.join(working_dir, 'models')
os.makedirs(models_dir, exist_ok = True) 
model_file = os.path.join(models_dir, "dante_by_rev_syl_model.h5")

model = tf.keras.models.load_model(model_file)

#SEQ_LENGTH = 250
SEQ_LENGTH = model.get_layer('embedding').output.shape[1]
EMBEDDING_DIM = model.get_layer('embedding').output.shape[2]
for l in model.layers:
    if l.name == 'first_lstm':
        RNN_TYPE = '2lstm'
        break
    if l.name == 'last_lstm':
        RNN_TYPE = 'lstm' 
        break
    if l.name == 'first_gru':
        RNN_TYPE = '2gru' 
        break
    if l.name == 'last_gru':
        RNN_TYPE = 'gru' 
        break
if 'lstm' in RNN_TYPE:
    RNN_UNITS = model.get_layer('last_lstm').output.shape[-1]
    SINGLE_OUTPUT = False if len(model.get_layer('last_lstm').output.shape) == 3 else True
if 'gru' in RNN_TYPE:
    RNN_UNITS = model.get_layer('last_gru').output.shape[-1]
    SINGLE_OUTPUT = False if len(model.get_layer('last_gru').output.shape) == 3 else True

model.summary()

model_filename = 'model_by_syl_seq{}_emb{}_{}{}_singleoutput{}'.format(SEQ_LENGTH, EMBEDDING_DIM, RNN_TYPE, RNN_UNITS, SINGLE_OUTPUT)


print("\nMODEL: {}\n".format(model_filename))


embedding_layer = model.get_layer('embedding')


embedding_file = os.path.join(logs_dir, model_filename, "embedding.tsv")
metadata_file = os.path.join(logs_dir, model_filename, "metadata.tsv")


f_e = open(embedding_file, "w")
f_m = open(metadata_file, "w")

#f_m.write("{}\n".format('syl'))
writer = csv.writer(f_e, delimiter='\t', lineterminator='\n')             
for s in vocab:
    f_m.write('"{}"\n'.format(s))
    emb = embedding_layer(np.array(syl2idx[s])).numpy()
    writer.writerow(emb)

f_e.close()
f_m.close()