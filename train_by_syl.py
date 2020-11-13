import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from dante_by_syl.syllabification import syllabify_verse
from dante_by_syl.data_preparation import text_in_syls, build_vocab, build_dataset, split_dataset
from dante_by_syl.text_processing import clean_comedy, prettify_text, special_tokens
from dante_by_syl.dante_model import build_model
from dante_by_syl.training_dante import train_model
from utils import save_vocab, load_vocab

working_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dante_by_syl')

divine_comedy_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "divina_commedia", "divina_commedia_accent_UTF-8.txt") 

with open(divine_comedy_file,"r") as f:
    divine_comedy = f.read()

divine_comedy = clean_comedy(divine_comedy, special_tokens)

##############################
# Training's hyper-parameters

## VERSION 1
#
BATCH_SIZE = 32
EPOCHS = 200
SEQ_LENGTH = 125
EMBEDDING_DIM = 256
RNN_UNITS = 1024
RNN_TYPE = 'lstm'

## VERSION 2

# BATCH_SIZE = 32
# EPOCHS = 200
# SEQ_LENGTH = 125
# EMBEDDING_DIM = 256
# RNN_UNITS = 1024
# RNN_TYPE = '2lstm'

## VERSION 3

# BATCH_SIZE = 32
# EPOCHS = 200
# SEQ_LENGTH = 125
# EMBEDDING_DIM = 256
# RNN_UNITS = 1024
# RNN_TYPE = 'gru'


##############################

vocab, idx2syl, syl2idx = build_vocab(divine_comedy)


dataset = build_dataset(divine_comedy, vocab, idx2syl, syl2idx, seq_length=SEQ_LENGTH)

print("Corpus length: {} syllables".format(len(text_in_syls(divine_comedy))))
print("Vocab size:", len(vocab))

# Path where the vocab will be saved
logs_dir = os.path.join(working_dir, 'logs')
os.makedirs(logs_dir, exist_ok = True) 
vocab_file = os.path.join(working_dir, 'logs', 'vocab.json')

save_vocab(vocab, idx2syl, syl2idx, vocab_file)


dataset_train, dataset_val = split_dataset(dataset)


dataset_train = dataset_train.batch(BATCH_SIZE, drop_remainder=True)
dataset_val = dataset_val.batch(BATCH_SIZE, drop_remainder=True)


model = build_model(
    vocab_size = len(vocab),
    seq_length = SEQ_LENGTH,
    embedding_dim=EMBEDDING_DIM,
    rnn_type = RNN_TYPE,
    rnn_units=RNN_UNITS,
    learning_rate=0.01,
    )


model_filename = 'model_by_syl_seq{}_emb{}_{}{}'.format(SEQ_LENGTH, EMBEDDING_DIM, RNN_TYPE, RNN_UNITS)

train_model(working_dir, 
        model,
        model_filename,
        dataset_train, 
        dataset_val, 
        epochs=EPOCHS, 
        )



