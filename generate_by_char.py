import os
import sys
sys.path.append(os.path.abspath("."))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from dante_by_char.data_preparation import build_vocab
from dante_by_char.text_processing import clean_comedy, prettify_text, special_tokens
from dante_by_char.generate_dante import generate_text

SEQ_LENGTH = 250

working_dir = os.path.abspath('dante_by_char')

divine_comedy_file = os.path.join(".", "divina_commedia", "divina_commedia_accent_UTF-8.txt") 

with open(divine_comedy_file,"r") as f:
    divine_comedy = f.read()

divine_comedy = clean_comedy(divine_comedy, special_tokens)

#divine_comedy = divine_comedy[:10000]

vocab, idx2char, char2idx = build_vocab(divine_comedy)

models_dir = os.path.join(working_dir, 'models')
os.makedirs(models_dir, exist_ok = True) 
model_file = os.path.join(models_dir, "dante_by_char_best_model.h5")

model = tf.keras.models.load_model(model_file)

# Length of the vocabulary
vocab_size = len(vocab)

model.summary()

start_string = divine_comedy[:105]
#start_string = special_tokens['START_OF_CANTO']

#print(start_string)

generated_text = generate_text(model, special_tokens, vocab_size, char2idx, idx2char, SEQ_LENGTH, start_string, temperature=1.0)

print(prettify_text(generated_text, special_tokens))
