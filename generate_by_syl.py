import os
import sys
sys.path.append(os.path.abspath("."))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from dante_by_syl.data_preparation import build_vocab, text_in_syls
from dante_by_syl.text_processing import clean_comedy, prettify_text, special_tokens
from dante_by_syl.generate_dante import generate_text
from utils import save_vocab, load_vocab

working_dir = os.path.abspath('dante_by_syl')

divine_comedy_file = os.path.join(".", "divina_commedia", "divina_commedia_accent_UTF-8.txt") 

with open(divine_comedy_file,"r") as f:
    divine_comedy = f.read()

divine_comedy = clean_comedy(divine_comedy, special_tokens)

#divine_comedy = divine_comedy[:100000]


#vocab, idx2syl, syl2idx = build_vocab(divine_comedy)


# Path where the vocab is saved
logs_dir = os.path.join(working_dir, 'logs')
os.makedirs(logs_dir, exist_ok = True) 
vocab_file = os.path.join(working_dir, 'logs', 'vocab.json')

vocab, idx2syl, syl2idx = load_vocab(vocab_file)


# Path where the model is saved
models_dir = os.path.join(working_dir, 'models')
os.makedirs(models_dir, exist_ok = True) 
model_file = os.path.join(models_dir, "dante_by_syl_model.h5")

model = tf.keras.models.load_model(model_file)

#SEQ_LENGTH = 250
#SINGLE_OUTPUT = False

SEQ_LENGTH = model.get_layer('embedding').output.shape[1]
SINGLE_OUTPUT = False if len(model.get_layer('last_lstm').output.shape) == 3 else True

# Length of the vocabulary
vocab_size = len(vocab)

model.summary()


start_string = divine_comedy[:374]
#start_string = special_tokens['START_OF_CANTO']

#print(start_string)

generated_text = generate_text(model, special_tokens, vocab_size, syl2idx, idx2syl, SEQ_LENGTH, SINGLE_OUTPUT, start_string, temperature=1.0)

#print(prettify_text(generated_text, special_tokens))


