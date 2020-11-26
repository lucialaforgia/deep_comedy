import os
import sys
sys.path.append(os.path.abspath("."))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from dante_by_tonedrev_syl.data_preparation import annotate_accent
from dante_by_tonedrev_syl.text_processing import clean_comedy, prettify_text, special_tokens, remove_all_punctuation
from dante_by_syl.generate_dante import generate_text
from utils import save_vocab, load_vocab

working_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dante_by_tonedrev_syl')

divine_comedy_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "divina_commedia", "divina_commedia_accent_UTF-8.txt") 

with open(divine_comedy_file,"r") as f:
    divine_comedy = f.read()

divine_comedy = clean_comedy(divine_comedy, special_tokens)
divine_comedy = prettify_text(divine_comedy, special_tokens)
divine_comedy = remove_all_punctuation(divine_comedy)

#vocab, idx2char, char2idx = build_vocab(divine_comedy)


# Path where the vocab is saved
logs_dir = os.path.join(working_dir, 'logs')
os.makedirs(logs_dir, exist_ok = True) 
vocab_file = os.path.join(logs_dir, 'vocab_tone.json')

vocab, idx2char, char2idx = load_vocab(vocab_file)

# Length of the vocabulary
vocab_size = len(vocab)


# Path where the model is saved
models_dir = os.path.join(working_dir, 'models')
os.makedirs(models_dir, exist_ok = True) 
model_file_tone = os.path.join(models_dir, "dante_by_tonedrev_syl_tone_model.h5")

model_tone = tf.keras.models.load_model(model_file_tone)

MAX_WORD_LENGTH = model_tone.get_layer('output').output.shape[1]
EMBEDDING_DIM = model_tone.get_layer('embedding').output.shape[2]
# for l in model_tone.layers:
#     if l.name == 'bidirectional':
#         RNN_TYPE = 'lstm' 
#         break
#     if l.name == 'last_gru':
#         RNN_TYPE = 'gru' 
#         break

RNN_TYPE = 'lstm'
RNN_UNITS = model_tone.get_layer('bidirectional').output.shape[-1]//2

# if 'lstm' in RNN_TYPE:
#     RNN_UNITS = model_tone.get_layer('bidirectional').output.shape[-1]
# if 'gru' in RNN_TYPE:
#     RNN_UNITS = model_tone.get_layer('last_gru').output.shape[-1]

model_tone.summary()


model_filename = 'model_by_tonedrev_syl_tone_wlen_{}_emb{}_{}{}'.format(MAX_WORD_LENGTH, EMBEDDING_DIM, RNN_TYPE, RNN_UNITS)

print("\nMODEL: {}\n".format(model_filename))

os.makedirs(os.path.join(logs_dir, model_filename), exist_ok = True) 


# output_file = os.path.join(logs_dir, model_filename, "output.txt")
# raw_output_file = os.path.join(logs_dir, model_filename, "raw_output.txt")


divine_comedy_words = divine_comedy.split()[:100]


for w in divine_comedy_words:
    print(annotate_accent(model_tone, w, char2idx, MAX_WORD_LENGTH), flush=True, end=' ')




# indexes = [i for i, x in enumerate(divine_comedy_verse) if x == special_tokens['END_OF_VERSO'] and i > SEQ_LENGTH]
# index_eov = np.random.choice(indexes)
# start_seq = divine_comedy_verse[index_eov - SEQ_LENGTH:index_eov]

# #print(start_seq)

# generated_text = generate_text(model_tone, special_tokens, vocab_size, char2idx, idx2char, SEQ_LENGTH, start_seq, temperature=1.0)

# #print(prettify_text(generated_text, special_tokens))


# with open(output_file,"w") as f:
#     f.write(prettify_text(generated_text, special_tokens))

# with open(raw_output_file,"w") as f:
#     f.write(generated_text)
