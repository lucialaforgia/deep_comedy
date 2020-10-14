import numpy as np
import tensorflow as tf
from dante_by_syl.syllabification import remove_puctuation, syllabify_verse


def text_in_syls(text):
    #this LIST's elements will be the verses of the DC
    verses = text.splitlines()
    verses_syl = []

    for i in range(len(verses)):
        verse = syllabify_verse(verses[i])
        verses_syl += verse

    return verses_syl
    

def build_vocab(text):
    
    vocab = sorted(list(set(text_in_syls(text))))
    
    idx2syl = { i : s for (i, s) in enumerate(vocab) }
    syl2idx = { s : i for (i, s) in enumerate(vocab) }
    
    return vocab, idx2syl, syl2idx


def build_dataset(text, vocab, idx2syl, syl2idx, seq_length):
    
    step_length = 32
    
    text_as_int = np.array([syl2idx[s] for s in text_in_syls(text)])

    dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
    dataset = dataset.window(seq_length + 1, shift=step_length, stride=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(seq_length + 1))
#    dataset = dataset.batch(seq_length+1, drop_remainder=True)

    def split_input_target(chunk):
        input_text = chunk[:-1]
#        target_text = chunk[-1]
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
    return dataset_train, dataset_val

#def build_dataset(text, vocab, idx2char, char2idx, seq_length):
#    # generate sequences
#    step_length = 32 
#
#    x = []   # extracted sequences
#    y = []   # the target: follow up character for each sequence in x
#    for i in range(0, len(text) - seq_length - 1, step_length):
#        seq_in = text[i:i + seq_length]
#        seq_out = text[i+1:i + seq_length+1]
#        y.append([char2idx[ch] for ch in seq_out])
#        
##        seq_out = text[i+seq_length]
##        y.append(char2idx[seq_out])
#        
#        x.append([char2idx[ch] for ch in seq_in])
#
#    print('Number of sequences:', len(x))
#    
#    # reshape x to be [samples, time steps, features]
#    x = np.reshape(x, (len(x), seq_length, 1))
#    # normalize
#    x = x / float(len(vocab))
#
#
#    # one hot encode the output variable
#    y = tf.keras.utils.to_categorical(y, num_classes=len(vocab))
#
##    sequences = []
##    next_chars = []
##    for i in range(0, len(text) - seq_length, step_length):
##        sequences.append(text[i : i + seq_length])
##        next_chars.append(text[i + seq_length])
##    print("Number of sequences:", len(sequences))
##
##    x = np.zeros((len(sequences), seq_length, len(vocab)), dtype=np.bool)
##    y = np.zeros((len(sequences), len(vocab)))
##    for i, sentence in enumerate(sequences):
##        for t, char in enumerate(sentence):
##            x[i, t, char2idx[char]] = 1
##        y[i, char2idx[next_chars[i]]] = 1
#
#
#    print("Input shape: {}".format(x.shape))
#    print("Target shape: {}".format(y.shape))
#
#    return x, y
#
#def split_dataset(x, y):
#
#    train_val_split = 0.7
#    train_samples = round(len(x) * train_val_split)
#
#    x_train, x_val = x[:train_samples,:,:], x[train_samples:,:,:]
#    y_train, y_val = y[:train_samples,:,:], y[train_samples:,:,:]
#    #y_train, y_val = y[:train_samples,:], y[train_samples:,:]
#    return x_train, y_train, x_val, y_val
