import os
import sys
sys.path.append(os.path.abspath("."))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from dante_by_rev_syl.syllabification import syllabify_verse
from dante_by_rev_syl.data_preparation import build_vocab_verse, build_dataset_verse, split_dataset
from dante_by_rev_syl.text_processing import clean_comedy, prettify_text, special_tokens
from dante_by_rev_syl.dante_model import build_model
from dante_by_rev_syl.training_dante import train_model
from utils import save_vocab, load_vocab

working_dir = os.path.abspath('dante_by_rev_syl')

divine_comedy_file = os.path.join(".", "divina_commedia", "divina_commedia_accent_UTF-8.txt") 

with open(divine_comedy_file,"r") as f:
    divine_comedy = f.read()

divine_comedy = clean_comedy(divine_comedy, special_tokens)

#divine_comedy = divine_comedy[:10000]


##############################
# Training's hyper-parameters

## VERSION 1
#
BATCH_SIZE = 32
EPOCHS = 200
SEQ_LENGTH = 50
EMBEDDING_DIM = 256
RNN_UNITS = 1024
RNN_TYPE = 'lstm'

## VERSION 2

#BATCH_SIZE = 32
#EPOCHS = 200
#SEQ_LENGTH = 50
#EMBEDDING_DIM = 256
#RNN_UNITS = 1024
#RNN_TYPE = '2lstm'

## VERSION 3

#BATCH_SIZE = 32
#EPOCHS = 200
#SEQ_LENGTH = 50
#EMBEDDING_DIM = 256
#RNN_UNITS = 1024
#RNN_TYPE = 'gru'

##############################

vocab_verse, idx2syl_verse, syl2idx_verse = build_vocab_verse(divine_comedy)

dataset_verse = build_dataset_verse(divine_comedy, vocab_verse, idx2syl_verse, syl2idx_verse, seq_length=SEQ_LENGTH)


# Path where the vocab will be saved
logs_dir = os.path.join(working_dir, 'logs')
os.makedirs(logs_dir, exist_ok = True) 
vocab_file_verse = os.path.join(working_dir, 'logs', 'vocab_verse.json')

save_vocab(vocab_verse, idx2syl_verse, syl2idx_verse, vocab_file_verse)

dataset_train_verse, dataset_val_verse = split_dataset(dataset_verse)


dataset_train_verse = dataset_train_verse.batch(BATCH_SIZE, drop_remainder=True)
dataset_val_verse = dataset_val_verse.batch(BATCH_SIZE, drop_remainder=True)


model_verse = build_model(
    name='VerseNetwork',
    vocab_size = len(vocab_verse),
    seq_length = SEQ_LENGTH,
    embedding_dim=EMBEDDING_DIM,
    rnn_type = RNN_TYPE,
    rnn_units=RNN_UNITS,
    learning_rate=0.01,
    )



model_filename_verse = 'model_by_rev_syl_verse_seq{}_emb{}_{}{}'.format(SEQ_LENGTH, EMBEDDING_DIM, RNN_TYPE, RNN_UNITS)

train_model(working_dir, 
        model_verse,
        model_filename_verse,
        dataset_train_verse, 
        dataset_val_verse, 
        epochs=EPOCHS, 
        )



