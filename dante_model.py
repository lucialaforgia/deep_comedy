import tensorflow as tf

def build_model(vocab_size, seq_length, embedding_dim=128, rnn_type='lstm', rnn_units=512, learning_rate=0.001, single_output=False):

    model = tf.keras.Sequential(name='DeepComedy')

    model.add(tf.keras.layers.Input((seq_length,)))
    model.add(tf.keras.layers.Embedding(vocab_size, embedding_dim))
    if single_output:
        if rnn_type == 'lstm':
            model.add(tf.keras.layers.LSTM(rnn_units,
                              dropout=0.3,
                              recurrent_initializer='glorot_uniform')
            )
        elif rnn_type == 'gru':
            model.add(tf.keras.layers.GRU(rnn_units,
                              dropout=0.3,
                              recurrent_initializer='glorot_uniform')
            )
    else:
        if rnn_type == 'lstm':
            model.add(tf.keras.layers.LSTM(rnn_units,
                              return_sequences=True,
                              dropout=0.3,
                              recurrent_initializer='glorot_uniform')
            )
        elif rnn_type == 'gru':
            model.add(tf.keras.layers.GRU(rnn_units,
                              return_sequences=True,
                              dropout=0.3,
                              recurrent_initializer='glorot_uniform')
            )

#    model.add(tf.keras.layers.Dense(256, activation='relu'))

    model.add(tf.keras.layers.Dense(vocab_size, activation='softmax'))

#    model.add(tf.keras.layers.Dense(vocab_size))
    
    
#    def loss(labels, logits):
#        return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
#    model.compile(loss=loss, metrics="accuracy", optimizer=optimizer)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss="sparse_categorical_crossentropy", metrics="accuracy", optimizer=optimizer)

    
    model.summary()

    return model
