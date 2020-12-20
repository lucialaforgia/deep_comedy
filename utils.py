import json


def save_vocab(vocab, idx2text, text2idx, vocab_file):
    to_save = {'vocab': vocab, 'idx2text': idx2text, 'text2idx': text2idx}
    with open(vocab_file, 'w') as f:
        json.dump(to_save, f) 


def load_vocab(vocab_file):
    with open(vocab_file, 'r') as f:
        loaded = json.load(f)
    
    vocab = loaded['vocab']
    idx2text = { int(k) : v for k, v in loaded['idx2text'].items() }
    text2idx = { k : int(v) for k, v in loaded['text2idx'].items() }
    return vocab, idx2text, text2idx


def save_syls_list(syls_list, filename):
    d = {'text_in_syls': syls_list}
    with open(filename, 'w') as f:
        json.dump(d, f) 


def load_syls_list(filename):
    with open(filename, 'r') as f:
        syls_list = json.load(f)['text_in_syls']
    return syls_list

