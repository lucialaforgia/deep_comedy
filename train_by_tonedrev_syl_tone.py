import os
import sys
import pandas as pd
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from dante_by_tonedrev_syl.data_preparation import build_vocab_tone, build_dataset_tone, split_dataset
from dante_by_tonedrev_syl.dante_model import build_tonenet_model
from dante_by_tonedrev_syl.training_dante import train_model
from utils import save_vocab, load_vocab

working_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dante_by_tonedrev_syl')

tone_dataset_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'toned_dataset_file.csv') 

tone_dataframe = pd.read_csv(tone_dataset_file, sep='\t', encoding='utf-8')

##############################
# Training's hyper-parameters

## VERSION 1

BATCH_SIZE = 8
EPOCHS = 100
MAX_WORD_LENGTH = 30
EMBEDDING_DIM = 64
RNN_UNITS = 512
RNN_TYPE = 'lstm'

## VERSION 2

# BATCH_SIZE = 32
# EPOCHS = 200
# MAX_WORD_LENGTH = 25
# EMBEDDING_DIM = 64
# RNN_UNITS = 512
# RNN_TYPE = 'lstm'

## VERSION 3

# BATCH_SIZE = 32
# EPOCHS = 200
# MAX_WORD_LENGTH = 100
# EMBEDDING_DIM = 64
# RNN_UNITS = 512
# RNN_TYPE = 'lstm'

##############################

vocab_tone, idx2char_tone, char2idx_tone = build_vocab_tone(tone_dataframe)

dataset_tone = build_dataset_tone(tone_dataframe, vocab_tone, idx2char_tone, char2idx_tone, MAX_WORD_LENGTH)

# Path where the vocab will be saved
logs_dir = os.path.join(working_dir, 'logs')
os.makedirs(logs_dir, exist_ok = True) 
vocab_file_tone = os.path.join(logs_dir, 'vocab_tone.json')

save_vocab(vocab_tone, idx2char_tone, char2idx_tone, vocab_file_tone)

dataset_train_tone, dataset_val_tone = split_dataset(dataset_tone)


dataset_train_tone = dataset_train_tone.batch(BATCH_SIZE, drop_remainder=True)
dataset_val_tone = dataset_val_tone.batch(BATCH_SIZE, drop_remainder=True)


model_tone = build_tonenet_model(
    name='ToneNetwork',
    vocab_size = len(vocab_tone),
    output_size = MAX_WORD_LENGTH,
    embedding_dim=EMBEDDING_DIM,
    rnn_type = RNN_TYPE,
    rnn_units=RNN_UNITS,
    learning_rate=0.01,
    )



model_filename_tone = 'model_by_tonedrev_syl_tone_wlen_{}_emb{}_{}{}'.format(MAX_WORD_LENGTH, EMBEDDING_DIM, RNN_TYPE, RNN_UNITS)

train_model(working_dir, 
        model_tone,
        model_filename_tone,
        dataset_train_tone, 
        dataset_val_tone, 
        epochs=EPOCHS, 
        )



word = 'abate'

word_as_int = [ char2idx_tone[c] for c in word ]
word_as_int = tf.keras.preprocessing.sequence.pad_sequences([word_as_int], padding='post', maxlen=MAX_WORD_LENGTH)
#word_as_int = np.expand_dims(word_as_int, axis=0)
print(word_as_int.shape)
output = model_tone.predict(word_as_int)
print(output.shape)
print(output)
print(np.argmax(output, axis=1))
