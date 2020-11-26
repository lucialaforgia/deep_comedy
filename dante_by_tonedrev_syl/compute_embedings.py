import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

import csv
import string
from utils import save_vocab, load_vocab

working_dir = os.path.dirname(os.path.abspath(__file__))

# Path where the vocab is saved
logs_dir = os.path.join(working_dir, 'logs')
os.makedirs(logs_dir, exist_ok = True) 
vocab_file_rhyme = os.path.join(logs_dir, 'vocab_rhyme.json')

vocab_rhyme, idx2syl_rhyme, syl2idx_rhyme = load_vocab(vocab_file_rhyme)

# Length of the vocabulary
vocab_size = len(vocab_rhyme)


# Path where the model is saved
models_dir = os.path.join(working_dir, 'models')
os.makedirs(models_dir, exist_ok = True) 
model_file_rhyme = os.path.join(models_dir, "dante_by_rev_syl_rhyme_model.h5")

model_rhyme = tf.keras.models.load_model(model_file_rhyme)
SEQ_LENGTH_RHYME = model_rhyme.get_layer('embedding').output.shape[1]

EMBEDDING_DIM_RHYME  = model_rhyme.get_layer('embedding').output.shape[2]
for l in model_rhyme.layers:
    if l.name == 'first_lstm':
        RNN_TYPE_RHYME  = '2lstm'
        break
    if l.name == 'last_lstm':
        RNN_TYPE_RHYME  = 'lstm' 
        break
    if l.name == 'first_gru':
        RNN_TYPE_RHYME  = '2gru' 
        break
    if l.name == 'last_gru':
        RNN_TYPE_RHYME  = 'gru' 
        break
if 'lstm' in RNN_TYPE_RHYME :
    RNN_UNITS_RHYME  = model_rhyme.get_layer('last_lstm').output.shape[-1]
if 'gru' in RNN_TYPE_RHYME:
    RNN_UNITS_RHYME  = model_rhyme.get_layer('last_gru').output.shape[-1]

model_rhyme.summary()

model_filename_rhyme = 'model_by_rev_syl_rhyme_seq{}_emb{}_{}{}'.format(SEQ_LENGTH_RHYME , EMBEDDING_DIM_RHYME , RNN_TYPE_RHYME , RNN_UNITS_RHYME )


print("\nMODEL RHYME: {}\n".format(model_filename_rhyme))

embedding_layer_rhyme = model_rhyme.get_layer('embedding')


embedding_file_rhyme = os.path.join(logs_dir, model_filename_rhyme, "embedding.tsv")
metadata_file_rhyme = os.path.join(logs_dir, model_filename_rhyme, "metadata.tsv")


f_e = open(embedding_file_rhyme, "w")
f_m = open(metadata_file_rhyme, "w")

#f_m.write("{}\n".format('syl'))
writer = csv.writer(f_e, delimiter='\t', lineterminator='\n')             
for s in vocab_rhyme:
    f_m.write('"{}"\n'.format(s))
    emb = embedding_layer_rhyme(np.array(syl2idx_rhyme[s])).numpy()
    writer.writerow(emb)

f_e.close()
f_m.close()