import tensorflow_datasets as tfds

def ngrams_plagiarism(generated_text, original_text, n=4):
    # the tokenizer is used to remove non-alphanumeric symbols
    tokenizer = tfds.deprecated.text.Tokenizer()
    original_text = tokenizer.join(tokenizer.tokenize(original_text.lower()))
    generated_text_tokens = tokenizer.tokenize(generated_text.lower())
    total_ngrams = len(generated_text_tokens) - n + 1
    plagiarism_counter = 0

    for i in range(total_ngrams):
        ngram = tokenizer.join(generated_text_tokens[i:i+n])
        plagiarism_counter += 1 if ngram in original_text else 0

    return 1 - (plagiarism_counter / total_ngrams)
