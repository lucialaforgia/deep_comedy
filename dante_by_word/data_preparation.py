import numpy as np
import tensorflow as tf

def build_vocab(text):

    vocab = sorted(list(set(text.split())))
    
    idx2word = { i : w for (i, w) in enumerate(vocab) }
    word2idx = { w : i for (i, w) in enumerate(vocab) }
    
    return vocab, idx2word, word2idx


def build_dataset(text, vocab, idx2word, word2idx, seq_length, single_output=False):
    
#    step_length = 4
    step_length = seq_length + 1
    
    text_as_int = np.array([word2idx[w] for w in text.split()])

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
