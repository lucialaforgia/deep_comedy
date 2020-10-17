import json


def save_vocab(vocab, idx2text, text2idx, vocab_file):
    to_save = {'vocab': vocab, 'idx2text': idx2text, 'text2idx': text2idx}
    with open(vocab_file, 'w') as fp:
        json.dump(to_save, fp) 


def load_vocab(vocab_file):
    with open(vocab_file, 'r') as fp:
        loaded = json.load(fp)
    
    vocab = loaded['vocab']
    idx2text = { int(k) : v for k, v in loaded['idx2text'].items() }
    text2idx = { k : int(v) for k, v in loaded['text2idx'].items() }
    return vocab, idx2text, text2idx
