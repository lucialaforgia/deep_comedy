import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from dante_by_tonedrev_syl.syllabification import syllabify_verse
from dante_by_tonedrev_syl.data_preparation import split_dataset, text_in_syls_rhyme, build_vocab_rhyme, build_dataset_rhyme
from dante_by_tonedrev_syl.text_processing import clean_comedy, prettify_text, special_tokens
from dante_by_tonedrev_syl.dante_model import build_model
from dante_by_tonedrev_syl.training_dante import train_model
from utils import save_vocab, load_vocab

working_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dante_by_tonedrev_syl')

divine_comedy_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "divina_commedia", "divina_commedia_accent_UTF-8.txt") 

with open(divine_comedy_file,"r") as f:
    divine_comedy = f.read()

divine_comedy = clean_comedy(divine_comedy, special_tokens)


##############################
# Training's hyper-parameters

## VERSION 1

# BATCH_SIZE = 32
# EPOCHS = 200
# SEQ_LENGTH = 24
# EMBEDDING_DIM = 256
# RNN_UNITS = 512
# RNN_TYPE = 'lstm'

## VERSION 2

BATCH_SIZE = 4
EPOCHS = 200
SEQ_LENGTH = 580
EMBEDDING_DIM = 256
RNN_UNITS = 512
RNN_TYPE = 'lstm'

## VERSION 3

# BATCH_SIZE = 32
# EPOCHS = 200
# SEQ_LENGTH = 580
# EMBEDDING_DIM = 256
# RNN_UNITS = 512
# RNN_TYPE = 'lstm'

##############################

vocab_rhyme, idx2syl_rhyme, syl2idx_rhyme = build_vocab_rhyme(divine_comedy)

dataset_rhyme = build_dataset_rhyme(divine_comedy, vocab_rhyme, idx2syl_rhyme, syl2idx_rhyme, seq_length=SEQ_LENGTH)


# Path where the vocab will be saved
logs_dir = os.path.join(working_dir, 'logs')
os.makedirs(logs_dir, exist_ok = True) 
vocab_file_rhyme = os.path.join(logs_dir, 'vocab_rhyme.json')

save_vocab(vocab_rhyme, idx2syl_rhyme, syl2idx_rhyme, vocab_file_rhyme)

dataset_train_rhyme, dataset_val_rhyme = split_dataset(dataset_rhyme)

dataset_train_rhyme = dataset_train_rhyme.batch(BATCH_SIZE, drop_remainder=True)
dataset_val_rhyme = dataset_val_rhyme.batch(BATCH_SIZE, drop_remainder=True)


model_rhyme = build_model(
    name='RhymeNetwork',
    vocab_size = len(vocab_rhyme),
    seq_length = SEQ_LENGTH,
    embedding_dim=EMBEDDING_DIM,
    rnn_type = RNN_TYPE,
    rnn_units=RNN_UNITS,
    learning_rate=0.01,
    )



model_filename_rhyme = 'model_by_rev_syl_rhyme_seq{}_emb{}_{}{}'.format(SEQ_LENGTH, EMBEDDING_DIM, RNN_TYPE, RNN_UNITS)



train_model(working_dir, 
        model_rhyme,
        model_filename_rhyme,
        dataset_train_rhyme, 
        dataset_val_rhyme, 
        epochs=EPOCHS, 
        )



