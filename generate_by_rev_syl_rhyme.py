import os
import sys
sys.path.append(os.path.abspath("."))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from dante_by_rev_syl.data_preparation import text_in_syls_rhyme
from dante_by_rev_syl.text_processing import clean_comedy, prettify_text, special_tokens
from dante_by_rev_syl.generate_dante import generate_text
from utils import save_vocab, load_vocab

working_dir = os.path.abspath('dante_by_rev_syl')

divine_comedy_file = os.path.join(".", "divina_commedia", "divina_commedia_accent_UTF-8.txt") 

with open(divine_comedy_file,"r") as f:
    divine_comedy = f.read()

divine_comedy = clean_comedy(divine_comedy, special_tokens)


#vocab, idx2syl, syl2idx = build_vocab(divine_comedy)


# Path where the vocab is saved
logs_dir = os.path.join(working_dir, 'logs')
os.makedirs(logs_dir, exist_ok = True) 
vocab_file = os.path.join(working_dir, 'logs', 'vocab_rhyme.json')

vocab, idx2syl, syl2idx = load_vocab(vocab_file)

# Length of the vocabulary
vocab_size = len(vocab)


# Path where the model is saved
models_dir = os.path.join(working_dir, 'models')
os.makedirs(models_dir, exist_ok = True) 
model_file_rhyme = os.path.join(models_dir, "dante_by_rev_syl_rhyme_model.h5")

model_rhyme = tf.keras.models.load_model(model_file_rhyme)

#SEQ_LENGTH = 250
#SINGLE_OUTPUT = False

SEQ_LENGTH = model_rhyme.get_layer('embedding').output.shape[1]
EMBEDDING_DIM = model_rhyme.get_layer('embedding').output.shape[2]
for l in model_rhyme.layers:
    if l.name == 'first_lstm':
        RNN_TYPE = '2lstm'
        break
    if l.name == 'last_lstm':
        RNN_TYPE = 'lstm' 
        break
    if l.name == 'first_gru':
        RNN_TYPE = '2gru' 
        break
    if l.name == 'last_gru':
        RNN_TYPE = 'gru' 
        break
if 'lstm' in RNN_TYPE:
    RNN_UNITS = model_rhyme.get_layer('last_lstm').output.shape[-1]
if 'gru' in RNN_TYPE:
    RNN_UNITS = model_rhyme.get_layer('last_gru').output.shape[-1]

model_rhyme.summary()

model_filename = 'model_by_rev_syl_rhyme_seq{}_emb{}_{}{}'.format(SEQ_LENGTH, EMBEDDING_DIM, RNN_TYPE, RNN_UNITS)

print("\nMODEL: {}\n".format(model_filename))

output_file = os.path.join(logs_dir, model_filename, "output.txt")
raw_output_file = os.path.join(logs_dir, model_filename, "raw_output.txt")


divine_comedy = text_in_syls_rhyme(divine_comedy)


index_eoc = divine_comedy.index(special_tokens['END_OF_CANTO']) + 1
start_seq = divine_comedy[index_eoc - SEQ_LENGTH:index_eoc]
#start_seq = divine_comedy[:374]
#start_seq = special_tokens['START_OF_CANTO']

#print(start_seq)

generated_text = generate_text(model_rhyme, special_tokens, vocab_size, syl2idx, idx2syl, SEQ_LENGTH, start_seq, temperature=1.0)

#print(prettify_text(generated_text, special_tokens))


with open(output_file,"w") as f:
    f.write(prettify_text(generated_text, special_tokens))

with open(raw_output_file,"w") as f:
    f.write(generated_text)
