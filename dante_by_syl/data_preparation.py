import numpy as np
import tensorflow as tf
from dante_by_syl.syllabification import syllabify_verse
from dante_by_syl.text_processing import special_tokens

def text_in_syls(text):
    #this LIST's elements will be the verses of the DC
    verses = text.splitlines()
    verses_syl = []

    for i in range(len(verses)):
        verse = syllabify_verse(verses[i], special_tokens)
        verses_syl += verse

    return verses_syl
    

def build_vocab(text):
    
    vocab = sorted(list(set(text_in_syls(text))))
    
    idx2syl = { i : s for (i, s) in enumerate(vocab) }
    syl2idx = { s : i for (i, s) in enumerate(vocab) }
    
    return vocab, idx2syl, syl2idx


def build_dataset(text, vocab, idx2syl, syl2idx, seq_length, single_output=False):
    
    step_length = 16 
    
    text_as_int = np.array([syl2idx[s] for s in text_in_syls(text)])

    dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
    dataset = dataset.window(seq_length + 1, shift=step_length, stride=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(seq_length + 1))

    def split_input_target(chunk):
        input_text = chunk[:-1]
        if single_output:
            target_text = chunk[-1]
        else:
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
