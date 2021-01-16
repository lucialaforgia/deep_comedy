import os
import json
import numpy as np
import tensorflow as tf
from dante_by_tonedrev_syl.syllabification import syllabify_verse, is_toned_syl
from dante_by_tonedrev_syl.text_processing import special_tokens
from dante_by_tonedrev_syl.tone import ToneTagger
from utils import save_syls_list, load_syls_list

def text_in_syls_rhyme(text):
    #this LIST's elements will be the verses of the DC
    verses = text.splitlines()
    verses_syl = []

    for i in range(len(verses)):
        verse = syllabify_verse(verses[i], special_tokens)
        if len(verse) > 1:
            syllables_rev = verse[::-1]
            for i, syl in enumerate(syllables_rev):
                if is_toned_syl(syl):
                    index = len(syllables_rev)-i-1
                    verses_syl += verse[index:]
                    break

        else:
            verses_syl += verse

    return verses_syl
    
def text_in_rev_syls(text):
    #this LIST's elements will be the verses of the DC
    verses = text.splitlines()
    verses_syl = []

    for i in range(len(verses)):
        verse = syllabify_verse(verses[i], special_tokens)
        if len(verse) > 1:
            verses_syl += verse[::-1]

    return verses_syl

def build_vocab_verse(text):

    text_in_syls_verse_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'text_in_syls_verse.json')

    # if os.path.isfile(text_in_syls_verse_file):
    #     syls_verse_list = load_syls_list(text_in_syls_verse_file)
    # else:
    #     syls_verse_list = text_in_rev_syls(text)
    #     save_syls_list(syls_verse_list, text_in_syls_verse_file)

    # vocab = sorted(list(set(syls_verse_list)))
    
    syls_verse_list = text_in_rev_syls(text)
    save_syls_list(syls_verse_list, text_in_syls_verse_file)
    
    vocab = sorted(list(set(syls_verse_list)))

    idx2syl = { i : s for (i, s) in enumerate(vocab) }
    syl2idx = { s : i for (i, s) in enumerate(vocab) }
    
    return vocab, idx2syl, syl2idx

def build_vocab_rhyme(text):

    text_in_syls_rhyme_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'text_in_syls_rhyme.json')

    # if os.path.isfile(text_in_syls_rhyme_file):
    #     syls_rhyme_list = load_syls_list(text_in_syls_rhyme_file)
    # else:
    #     syls_rhyme_list = text_in_syls_rhyme(text)
    #     save_syls_list(syls_rhyme_list, text_in_syls_rhyme_file)

    # vocab = sorted(list(set(syls_rhyme_list)))
    
    
    syls_rhyme_list = text_in_syls_rhyme(text)
    save_syls_list(syls_rhyme_list, text_in_syls_rhyme_file)

    vocab = sorted(list(set(syls_rhyme_list)))

    idx2syl = { i : s for (i, s) in enumerate(vocab) }
    syl2idx = { s : i for (i, s) in enumerate(vocab) }
    
    return vocab, idx2syl, syl2idx

def build_vocab_tone(tone_dataframe):

    vocab = sorted(list(set(''.join(list(tone_dataframe['word'])))))
    
    vocab.insert(0, '#')

    idx2char = { i : c for (i, c) in enumerate(vocab) }
    char2idx = { c : i for (i, c) in enumerate(vocab) }
    
    return vocab, idx2char, char2idx

def build_dataset_tone(tone_dataframe, vocab_tone, idx2char, char2idx, max_length):
    
    words_as_int = [[ char2idx[c] for c in w ] for w in list(tone_dataframe['word']) ]
    words_as_int = tf.keras.preprocessing.sequence.pad_sequences(words_as_int, padding='post', maxlen=max_length)
    
    # print(words_as_int)
    # print(words_as_int.shape)

    toned_index = [i for i in list(tone_dataframe['index'])]

    toned_index = tf.keras.utils.to_categorical(toned_index)

    toned_index  = tf.keras.preprocessing.sequence.pad_sequences(toned_index, padding='post', maxlen=max_length)

    # print(toned_index.shape)
    # print(toned_index[0])
    
    dataset = tf.data.Dataset.from_tensor_slices((words_as_int, toned_index))
#    print(list(dataset.take(1).as_numpy_iterator()))

    dataset = dataset.shuffle(1000)

    return dataset

def build_dataset_rhyme(text, vocab, idx2syl, syl2idx, seq_length):
    
#    step_length = 32
    step_length = seq_length + 1


    text_in_syls_rhyme_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'text_in_syls_rhyme.json')

    if os.path.isfile(text_in_syls_rhyme_file):
        syls_rhyme_list = load_syls_list(text_in_syls_rhyme_file)
    else:
        syls_rhyme_list = text_in_syls_rhyme(text)
        save_syls_list(syls_rhyme_list, text_in_syls_rhyme_file)


    # syls_rhyme_list = text_in_syls_rhyme(text)

    text_as_int = np.array([syl2idx[s] for s in syls_rhyme_list])

    dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
    dataset = dataset.window(seq_length + 1, shift=step_length, stride=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(seq_length + 1))

    def split_input_target(chunk):
        input_text = chunk[:-1]
        target_text = chunk[1:]
        return input_text, target_text

    dataset = dataset.map(split_input_target)

    dataset = dataset.shuffle(1000)

    return dataset

def build_dataset_verse(text, vocab, idx2syl, syl2idx, seq_length):
    
#    step_length = 8
    step_length = seq_length + 1
    
    text_in_syls_verse_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'text_in_syls_verse.json')

    if os.path.isfile(text_in_syls_verse_file):
        syls_verse_list = load_syls_list(text_in_syls_verse_file)
    else:
        syls_verse_list = text_in_rev_syls(text)
        save_syls_list(syls_verse_list, text_in_syls_verse_file)


    # syls_verse_list = text_in_rev_syls(text)

    text_as_int = np.array([syl2idx[s] for s in syls_verse_list])

    dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
    dataset = dataset.window(seq_length + 1, shift=step_length, stride=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(seq_length + 1))

    def split_input_target(chunk):
        input_text = chunk[:-1]
        target_text = chunk[1:]
        return input_text, target_text

    dataset = dataset.map(split_input_target)

    dataset = dataset.shuffle(1000)

    return dataset


def split_dataset(dataset):
    
    tot_samples=0
    for _ in dataset:
        tot_samples+=1
    print("Total samples in the dataset: {}".format(tot_samples))
    train_val_split = 0.8
    train_samples = round(tot_samples * train_val_split)

    dataset_train = dataset.take(train_samples)
    dataset_val = dataset.skip(train_samples).take(tot_samples-train_samples)
    
    tot_samples=0
    for _ in dataset_train:
        tot_samples+=1
    print("Total samples in the train dataset: {}".format(tot_samples))
    
    tot_samples=0
    for _ in dataset_val:
        tot_samples+=1
    print("Total samples in the validation dataset: {}".format(tot_samples))

    return dataset_train, dataset_val
