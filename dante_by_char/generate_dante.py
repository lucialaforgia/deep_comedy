import os
import numpy as np
import tensorflow as tf

def generate_text(model, special_tokens, vocab_size, char2idx, idx2char, seq_length, temperature=1.0, start_string=""):
    generated_text = start_string
    sequence = start_string
    prediction = ''
    model.reset_states()
    i = 0
    while prediction != special_tokens['END_OF_CANTO'] \
            and generated_text.count(special_tokens['START_OF_TERZINA']) < 45 \
            and generated_text.count(special_tokens['END_OF_VERSO']) < 136 \
            and i < 500:
#        x = np.zeros((1, seq_length, vocab_size))
#        for t, ch in enumerate(text):
#            x[0, t, char2idx[ch]] = 1.0
        # normalize
#        x = x / float(vocab_size)
        sequence = [ char2idx[ch] for ch in generated_text[-seq_length:] ]
        sequence = tf.keras.preprocessing.sequence.pad_sequences([sequence], maxlen=seq_length)
        x = np.array(sequence, dtype='int64')
#        print(x)

#        x = np.reshape(sequence, (1, seq_length))
#        x = np.reshape(sequence, (1, seq_length, 1))

        prediction = model.predict(x, verbose=0)
        prediction = tf.squeeze(prediction, 0)[-1]
        prediction = prediction / temperature
        index = np.argmax(prediction)
        
#        prediction = model.predict(x, verbose=0)
#        prediction = tf.squeeze(prediction, 0)
#        prediction = prediction / temperature
#        index = tf.random.categorical(prediction, num_samples=1)[-1,0].numpy()

        prediction = idx2char[index]
        generated_text += prediction
#        sequence = sequence[1:] + prediction

        print(prediction, end='', flush=True)
        i+=1        
    print('\n')        
    return generated_text


