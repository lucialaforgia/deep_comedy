import numpy as np
import tensorflow as tf
from dante_by_rev_syl.syllabification import syllabify_verse
from dante_by_rev_syl.text_processing import special_tokens

def text_in_syls_rhyme(text):
    #this LIST's elements will be the verses of the DC
    verses = text.splitlines()
    verses_syl = []

    for i in range(len(verses)):
        verse = syllabify_verse(verses[i], special_tokens)
        verses_syl += verse[-3:]
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
    
    vocab = sorted(list(set(text_in_rev_syls(text))))
    
    idx2syl = { i : s for (i, s) in enumerate(vocab) }
    syl2idx = { s : i for (i, s) in enumerate(vocab) }
    
    return vocab, idx2syl, syl2idx

def build_vocab_rhyme(text):
    
    vocab = sorted(list(set(text_in_syls_rhyme(text))))
    
    idx2syl = { i : s for (i, s) in enumerate(vocab) }
    syl2idx = { s : i for (i, s) in enumerate(vocab) }
    
    return vocab, idx2syl, syl2idx

def build_dataset_rhyme(text, vocab, idx2syl, syl2idx, seq_length):
    
#    step_length = 32
    step_length = seq_length + 1
    
    text_as_int = np.array([syl2idx[s] for s in text_in_syls_rhyme(text)])

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
    
    text_as_int = np.array([syl2idx[s] for s in text_in_rev_syls(text)])

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
