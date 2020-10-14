import os
import sys
sys.path.append(os.path.abspath("."))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from dante_by_syl.syllabification import remove_puctuation, syllabify_verse
from dante_by_syl.data_preparation import text_in_syls, build_vocab, build_dataset, split_dataset
from dante_by_syl.text_processing import clean_comedy, prettify_text, special_tokens
from dante_by_syl.dante_model import build_model
from dante_by_syl.training_dante import train_model

working_dir = os.path.abspath('dante_by_syl')

divine_comedy_file = os.path.join(".", "divina_commedia", "divina_commedia_accent_UTF-8.txt") 

with open(divine_comedy_file,"r") as f:
    divine_comedy = f.read()

divine_comedy = clean_comedy(divine_comedy, special_tokens)

#divine_comedy = divine_comedy[:100000]

##############################
# Training's hyper-parameters
BATCH_SIZE = 32
EPOCHS = 50
SEQ_LENGTH = 150
##############################

vocab, idx2syl, syl2idx = build_vocab(divine_comedy)

#x_train, y_train = build_dataset(divine_comedy, vocab, idx2char, char2idx, seq_length)
#x_train, y_train, x_val, y_val = split_dataset(x_train, y_train)

dataset = build_dataset(divine_comedy, vocab, idx2syl, syl2idx, seq_length=SEQ_LENGTH)

print("Corpus length: {} syllables".format(len(text_in_syls(divine_comedy))))
print("Vocab size:", len(vocab))

dataset_train, dataset_val = split_dataset(dataset)

dataset_train = dataset_train.batch(BATCH_SIZE, drop_remainder=True)
dataset_val = dataset_val.batch(BATCH_SIZE, drop_remainder=True)

print(dataset_train)
for s in dataset_train.take(1).as_numpy_iterator():
    print(s)

model = build_model(
    vocab_size = len(vocab),
    seq_length = SEQ_LENGTH,
    embedding_dim=128,
    rnn_units=1024,
    learning_rate=0.001,
#    batch_size=BATCH_SIZE
    )


train_model(working_dir, 
        model, 
        dataset_train, 
        dataset_val, 
        epochs=EPOCHS, 
#        batch_size=BATCH_SIZE
        )


