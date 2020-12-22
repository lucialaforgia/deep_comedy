import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import numpy as np
import re
from utils import load_vocab



def remove_tone(syllables, special_tokens):
    toned_vowels = {
        'à': 'a',
        'è': 'e',
        'é': 'e',
        'ì': 'i',
        'ò': 'o',
        'ó': 'o',
        'ù': 'u',
    }
    cleaned_syllables = []
    for i, syl in enumerate(syllables[:-1]):
        syl = syllables[i]
        next_s = syllables[i+1]
        if syl.strip() in special_tokens.values():
            cleaned_syllables.append(syl)
        elif next_s in special_tokens.values() and syl[-1] in toned_vowels.keys():
            cleaned_syllables.append(syl)
        elif special_tokens['WORD_SEP'] in syl:
            new_sub_syls = []
            sub_syls = syl.split(special_tokens['WORD_SEP'])
            for s in sub_syls[:-1]:
                if s[-1] in toned_vowels.keys():
                    new_sub_syls.append(s)
                else:
                    new_sub_syls.append(''.join([toned_vowels[c] if c in toned_vowels.keys() else c for c in s]))

            if next_s in special_tokens.values() and sub_syls[-1] in toned_vowels.keys():
                new_sub_syls.append(sub_syls[-1])
            else:
                new_sub_syls.append(''.join([toned_vowels[c] if c in toned_vowels.keys() else c for c in sub_syls[-1]]))
            new_syl = special_tokens['WORD_SEP'].join(new_sub_syls)
            cleaned_syllables.append(new_syl)

        else: 
            cleaned_s = ''.join([toned_vowels[c] if c in toned_vowels.keys() else c for c in syl])
            cleaned_syllables.append(cleaned_s)

    if syllables[-1] in toned_vowels.keys():
        cleaned_syllables.append(syllables[-1])
    else:
        cleaned_syllables.append(''.join([toned_vowels[c] if c in toned_vowels.keys() else c for c in syllables[-1] ]))
    return cleaned_syllables










class ToneTagger():
    def __init__(self, path=None):

        self.vocab = None
        self.idx2char = None
        self.char2idx = None
        self.vocab_size = None
        if path != None:
            self.working_dir = path
        else:
            self.working_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_tone = None
        self.model_filename = None
        self.max_word_len = None

        self.toned_vowels = {
            'a': 'à',
            'e': 'è',
            'i': 'ì',
            'o': 'ò',
            'u': 'ù',
        }

        self.load_vocab_tone()
        self.load_tone_model()
    
    def load_vocab_tone(self):

        # Path where the vocab is saved
        logs_dir = os.path.join(self.working_dir, 'logs')
        os.makedirs(logs_dir, exist_ok = True) 
        vocab_file = os.path.join(logs_dir, 'vocab_tone.json')

        self.vocab, self.idx2char, self.char2idx = load_vocab(vocab_file)

        # Length of the vocabulary
        self.vocab_size = len(self.vocab)



    def load_tone_model(self):

        # Path where the model is saved
        models_dir = os.path.join(self.working_dir, 'models')
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
        
#        model_tone.summary()
        
        model_filename = 'model_by_tonedrev_syl_tone_wlen_{}_emb{}_{}{}'.format(MAX_WORD_LENGTH, EMBEDDING_DIM, RNN_TYPE, RNN_UNITS)
        self.max_word_len = MAX_WORD_LENGTH
        self.model_tone = model_tone
        self.model_filename = model_filename


    def tone(self, word):
        word = word.lower()

#        v = r"""(?i)([ÄäÁÀàáAaËëÉÈèéEeÏïÍÌíìIiÖöÓÒóòOoÜüÚÙúùUu]{1})"""
        v = r"""(?i)([ÁÀàáAaÉÈèéEeÍÌíìIiÓÒóòOoÚÙúùUu]{1})"""
        if len(re.findall(v, word)) == 1:
            return word

#        toned_v = r"""(?i)([ÁÀàáÉÈèéÍÌíìÓÒóòÚÙúù]{1})"""
#        if len(re.findall(toned_v, word)) == 1:
#            return word

        # do not tone some words
        not_tone = ['che', 'la', 'lo', 'le', 'qui', 'qua', 'quo', 
                    ] 
#        not_tone = ['che', 'la', 'lo', 'le', 'qui', 'qua', 'quo', 'io',
#                    'tuo', 'tua', 'tue', 'suo', 'sua', 'sue', 'mio', 'mia', 'mie'
#                    ] 
        if word in not_tone:
            return word

#        # do not tone word of 2 letters
#        if len(word) < 3:
#            return word

        dv = r"""(?i)([äëïöü]{1})"""
        if re.search(dv, word):
            dieresis_vowels = {
                'ä': 'a',
                'ë': 'e',
                'ï': 'i',
                'ö': 'o',
                'ü': 'u',
            }
            word_input = ''.join([ dieresis_vowels[c] if c in dieresis_vowels.keys() else c for c in word ])
        else:
            word_input = word 
        word_as_int = [ self.char2idx[c] if c in self.char2idx.keys() else self.char2idx['#'] for c in word_input ]
        word_as_int = tf.keras.preprocessing.sequence.pad_sequences([word_as_int], padding='post', maxlen=self.max_word_len)
        #word_as_int = np.expand_dims(word_as_int, axis=0)
        output = self.model_tone.predict(word_as_int)

    #    print(output.shape)
    #    print(output)

        tone_index = np.argmax(output, axis=1)[0] - 1
#        print('\ntone', word, tone_index)
        if tone_index >= len(word):
            return word
        if word[tone_index] in self.toned_vowels.keys():
            toned_word = list(word)
            toned_word[tone_index] = self.toned_vowels[word[tone_index]]
            return ''.join(toned_word)
        else:
            return word
