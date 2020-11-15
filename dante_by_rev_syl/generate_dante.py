import os
import numpy as np
import tensorflow as tf
from dante_by_syl.data_preparation import text_in_syls
from dante_by_syl.text_processing import prettify_text, special_tokens



def generate_text(model_rhyme, model_verse, special_tokens, vocab_size_rhyme, vocab_size_verse, syl2idx_rhyme, idx2syl_rhyme, syl2idx_verse, idx2syl_verse, seq_length_rhyme, seq_length_verse, start_seq_rhyme, start_seq_verse, temperature=1.0):
    seq_text_rhyme = start_seq_rhyme
    seq_text_verse = start_seq_verse

    generated_text_list = []

    model_rhyme.reset_states()
    model_verse.reset_states()
    end_of_canto = False
    while not end_of_canto:
        #      and generated_text_list.count(special_tokens['END_OF_TERZINA']) < 45 \
        #      and generated_text_list.count(special_tokens['END_OF_VERSO']) < 136:

        next_syl_rhyme = ''
        end_verse_list = []
        structure_list = []
        while not end_of_canto and next_syl_rhyme != special_tokens['END_OF_VERSO']:

            seq_text_rhyme = seq_text_rhyme[-seq_length_rhyme:]

            sequence_rhyme = [ syl2idx_rhyme[syl] for syl in seq_text_rhyme[-seq_length_rhyme:] ]
            sequence_rhyme = tf.keras.preprocessing.sequence.pad_sequences([sequence_rhyme], maxlen=seq_length_rhyme)
            x_rhyme = np.array(sequence_rhyme, dtype='int64')


            prediction_rhyme = model_rhyme.predict(x_rhyme, verbose=0)

            prediction_rhyme = tf.squeeze(prediction_rhyme, 0)[-1]
            prediction_rhyme = prediction_rhyme / temperature
        #    prediction = tf.nn.softmax(prediction_rhyme).numpy()
        #    prediction /= np.sum(prediction_rhyme)
            prediction_rhyme = prediction_rhyme.numpy()

        #    index_rhyme = np.random.choice(len(prediction_rhyme), size=1, p=prediction_rhyme)[0]
            index_rhyme = np.argmax(prediction_rhyme)

            next_syl_rhyme = idx2syl_rhyme[index_rhyme]
            seq_text_rhyme.append(next_syl_rhyme)

            if next_syl_rhyme in special_tokens.values() and next_syl_rhyme != special_tokens['END_OF_VERSO']:
                    structure_list.append(next_syl_rhyme)
            else:
                    end_verse_list.append(next_syl_rhyme)

            if next_syl_rhyme == special_tokens['END_OF_CANTO']:
                    end_of_canto = True

        generated_text_list += structure_list

        reverse_rhyme_list = end_verse_list[::-1]

##        seq_text_verse += structure_list
        seq_text_verse += reverse_rhyme_list

        next_syl_verse = ''

        rest_revese_verse_list = []

        while not end_of_canto and next_syl_verse != special_tokens['END_OF_VERSO']:
##        while not end_of_canto and (next_syl_verse == special_tokens['WORD_SEP'] or next_syl_verse not in special_tokens.values()):

            seq_text_verse = seq_text_verse[-seq_length_verse:]

            sequence_verse = [ syl2idx_verse[syl] for syl in seq_text_verse[-seq_length_verse:] ]
            sequence_verse = tf.keras.preprocessing.sequence.pad_sequences([sequence_verse], maxlen=seq_length_verse)
            x_verse = np.array(sequence_verse, dtype='int64')


            prediction_verse = model_verse.predict(x_verse, verbose=0)
            prediction_verse = tf.squeeze(prediction_verse, 0)[-1]
            prediction_verse = prediction_verse / temperature
            prediction_verse = prediction_verse.numpy()

            index_verse = np.random.choice(len(prediction_verse), size=1, p=prediction_verse)[0]

            next_syl_verse = idx2syl_verse[index_verse]
##            if next_syl_verse == special_tokens['WORD_SEP'] or next_syl_verse not in special_tokens.values():
            if next_syl_verse != special_tokens['END_OF_VERSO']:
                    seq_text_verse.append(next_syl_verse)
                    rest_revese_verse_list.append(next_syl_verse)


        whole_verse_list = rest_revese_verse_list[::-1] + end_verse_list

        generated_text_list += whole_verse_list
    
        print(prettify_text(''.join(structure_list), special_tokens),  end='', flush=True)
        print(prettify_text(''.join(whole_verse_list), special_tokens),  end='', flush=True)
#        print(''.join(structure_list),  end='\n', flush=True)
#        print(''.join(whole_verse_list),  end='\n', flush=True)

    return ''.join(generated_text_list)


