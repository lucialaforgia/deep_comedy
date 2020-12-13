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
vocab_file_tone = os.path.join(logs_dir, 'vocab_tone.json')

vocab_tone, idx2cha_tone, char2idx_tone = load_vocab(vocab_file_tone)

# Length of the vocabulary
vocab_size = len(vocab_tone)

# Path where the model is saved
models_dir = os.path.join(working_dir, 'models')
os.makedirs(models_dir, exist_ok = True) 
model_file_tone = os.path.join(models_dir, "dante_by_tonedrev_syl_tone_model.h5")


model_tone = tf.keras.models.load_model(model_file_tone)

MAX_WORD_LENGTH = model_tone.get_layer('output').output.shape[1]
EMBEDDING_DIM_TONE  = model_tone.get_layer('embedding').output.shape[2]

RNN_TYPE_TONE = 'lstm'
RNN_UNITS_TONE = model_tone.get_layer('bidirectional').output.shape[-1]//2

model_tone.summary()

model_filename_tone = 'model_by_tonedrev_syl_tone_wlen_{}_emb{}_{}{}'.format(MAX_WORD_LENGTH, EMBEDDING_DIM_TONE, RNN_TYPE_TONE, RNN_UNITS_TONE)


print("\nMODEL TONE: {}\n".format(model_filename_tone))

embedding_layer_tone = model_tone.get_layer('embedding')


embedding_file_rhyme = os.path.join(logs_dir, model_filename_tone, "embedding.tsv")
metadata_file_rhyme = os.path.join(logs_dir, model_filename_tone, "metadata.tsv")


f_e = open(embedding_file_rhyme, "w")
f_m = open(metadata_file_rhyme, "w")

#f_m.write("{}\n".format('syl'))
writer = csv.writer(f_e, delimiter='\t', lineterminator='\n')             
for s in vocab_tone:
    f_m.write('"{}"\n'.format(s))
    emb = embedding_layer_tone(np.array(char2idx_tone[s])).numpy()
    writer.writerow(emb)

f_e.close()
f_m.close()