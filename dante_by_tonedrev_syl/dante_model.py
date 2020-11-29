import tensorflow as tf

def build_model(name, vocab_size, seq_length, embedding_dim=64, rnn_type='lstm', rnn_units=512, learning_rate=0.01):

    model = tf.keras.Sequential(name=name)

    model.add(tf.keras.layers.Input((seq_length,), name='input'))
    model.add(tf.keras.layers.Embedding(vocab_size, embedding_dim, name='embedding'))
    if rnn_type == 'lstm':
        model.add(tf.keras.layers.LSTM(rnn_units,
                          return_sequences=True,
                          dropout=0.3,
                          recurrent_initializer='glorot_uniform',
                          name='last_lstm')
        )
    elif rnn_type == 'gru':
        model.add(tf.keras.layers.GRU(rnn_units,
                          return_sequences=True,
                          dropout=0.3,
                          recurrent_initializer='glorot_uniform',
                          name='last_gru')
        )
    elif rnn_type == '2lstm':
        model.add(tf.keras.layers.LSTM(rnn_units,
                          return_sequences=True,
                          dropout=0.3,
                          recurrent_initializer='glorot_uniform',
                          name='first_lstm')
        )
        model.add(tf.keras.layers.LSTM(rnn_units,
                          return_sequences=True,
                          dropout=0.3,
                          recurrent_initializer='glorot_uniform',
                          name='last_lstm')
        )

    elif rnn_type == '2gru':
        model.add(tf.keras.layers.GRU(rnn_units,
                          return_sequences=True,
                          dropout=0.3,
                          recurrent_initializer='glorot_uniform',
                          name='first_gru')
        )
        model.add(tf.keras.layers.GRU(rnn_units,
                          return_sequences=True,
                          dropout=0.3,
                          recurrent_initializer='glorot_uniform',
                          name='last_gru')
        )
#    model.add(tf.keras.layers.Dense(128, activation='relu', name='dense'))

    model.add(tf.keras.layers.Dense(vocab_size, activation='softmax', name='output'))

#    model.add(tf.keras.layers.Dense(vocab_size, name='output'))
    
    
#    def loss(labels, logits):
#        return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

#    model.compile(loss=loss, metrics="accuracy", optimizer=optimizer)
    
    model.compile(loss="sparse_categorical_crossentropy", metrics="accuracy", optimizer=optimizer)
    model.summary()

    return model


def build_tonenet_model(name, vocab_size, max_word_len, embedding_dim=32, rnn_type='lstm', rnn_units=512, learning_rate=0.01):

    model = tf.keras.Sequential(name=name)

    model.add(tf.keras.layers.Input((max_word_len,), name='input'))
    model.add(tf.keras.layers.Embedding(vocab_size, embedding_dim, mask_zero=True, name='embedding'))
    if rnn_type == 'lstm':
#         model.add(tf.keras.layers.LSTM(rnn_units,
# #                          return_sequences=True,
#                           dropout=0.3,
# #                          recurrent_initializer='glorot_uniform',
#                           name='last_lstm')
#         )
        model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(rnn_units,
#                          return_sequences=True,
                          dropout=0.3,
                          name='last_lstm'
                        ), name='bidirectional'
                    )
        )
#     elif rnn_type == 'gru':
#         model.add(tf.keras.layers.GRU(rnn_units,
# #                          return_sequences=True,
#                           dropout=0.3,
# #                          recurrent_initializer='glorot_uniform',
#                           name='last_gru')
#         )
   
#    model.add(tf.keras.layers.Dense(128, activation='relu', name='dense'))

    model.add(tf.keras.layers.Dense(max_word_len, activation='softmax', name='output'))    
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(loss="categorical_crossentropy", metrics="accuracy", optimizer=optimizer) 
    # anche qui non so se e' corretta la categorical crossentropy...

#    model.compile(loss="sparse_categorical_crossentropy", metrics="accuracy", optimizer=optimizer)
    model.summary()

    return model
