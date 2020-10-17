import os
import sys
sys.path.append(os.path.abspath("."))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from dante_by_char.data_preparation import build_vocab, build_dataset, split_dataset
from dante_by_char.text_processing import clean_comedy, prettify_text, special_tokens
from dante_by_char.dante_model import build_model
from dante_by_char.training_dante import train_model

working_dir = os.path.abspath('dante_by_char')

divine_comedy_file = os.path.join(".", "divina_commedia", "divina_commedia_accent_UTF-8.txt") 

with open(divine_comedy_file,"r") as f:
    divine_comedy = f.read()

divine_comedy = clean_comedy(divine_comedy, special_tokens)

#divine_comedy = divine_comedy[:100000]

##############################
# Training's hyper-parameters

## VERSION 1
#
BATCH_SIZE = 32
EPOCHS = 50
SEQ_LENGTH = 200
EMBEDDING_DIM = 32
RNN_UNITS = 512
RNN_TYPE = 'lstm'
SINGLE_OUTPUT = False

## VERSION 2

#BATCH_SIZE = 32
#EPOCHS = 50
#SEQ_LENGTH = 200
#EMBEDDING_DIM = 32
#RNN_UNITS = 256
#RNN_TYPE = '2lstm'
#SINGLE_OUTPUT = False

## VERSION 3

#BATCH_SIZE = 32
#EPOCHS = 50
#SEQ_LENGTH = 200
#EMBEDDING_DIM = 32
#RNN_UNITS = 512
#RNN_TYPE = 'lstm'
#SINGLE_OUTPUT = True

## VERSION 4

#BATCH_SIZE = 32
#EPOCHS = 50
#SEQ_LENGTH = 200
#EMBEDDING_DIM = 32
#RNN_UNITS = 256
#RNN_TYPE = '2lstm'
#SINGLE_OUTPUT = True

##############################

vocab, idx2char, char2idx = build_vocab(divine_comedy)

dataset = build_dataset(divine_comedy, vocab, idx2char, char2idx, seq_length=SEQ_LENGTH, single_output=SINGLE_OUTPUT)

print("Corpus length: {} characters".format(len(divine_comedy)))
print("Vocab size:", len(vocab))

dataset_train, dataset_val = split_dataset(dataset)

#for s in dataset_train.take(1).as_numpy_iterator():
#    print(s)

dataset_train = dataset_train.batch(BATCH_SIZE, drop_remainder=True)
dataset_val = dataset_val.batch(BATCH_SIZE, drop_remainder=True)


model = build_model(
    vocab_size = len(vocab),
    seq_length = SEQ_LENGTH,
    embedding_dim=EMBEDDING_DIM,
    rnn_type = RNN_TYPE,
    rnn_units=RNN_UNITS,
    learning_rate=0.001,
    single_output=SINGLE_OUTPUT,
    )

model_filename = 'model_by_char_seq{}_emb{}_{}{}_singleoutput{}'.format(SEQ_LENGTH, EMBEDDING_DIM, RNN_TYPE, RNN_UNITS, SINGLE_OUTPUT)

train_model(working_dir, 
        model, 
        model_filename,
        dataset_train, 
        dataset_val, 
        epochs=EPOCHS, 
        )


