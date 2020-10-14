import tensorflow as tf

def build_model(vocab_size, seq_length, embedding_dim=256, rnn_units=1024, learning_rate=0.001, batch_size=32):
#    input_layer = tf.keras.layers.Input((seq_length,))
#    embedding_layer = tf.keras.layers.Embedding(vocab_size, embedding_dim)(input_layer)
#
#    rnn_layer = tf.keras.layers.LSTM(rnn_units,
#                      return_sequences=True,
#                      dropout=0.3,
#                      recurrent_initializer='glorot_uniform')(embedding_layer)
#    output_layer = tf.keras.layers.Dense(vocab_size, activation='softmax')(rnn_layer)
#    model = tf.keras.Model(input_layer, output_layer, name='DeepComedy')


    model = tf.keras.Sequential([
      tf.keras.layers.Input((seq_length,)),
      tf.keras.layers.Embedding(vocab_size, embedding_dim),
      tf.keras.layers.LSTM(rnn_units,
                          return_sequences=True,
                          dropout=0.3,
                          recurrent_initializer='glorot_uniform'),

#      tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[batch_size, None]),
#      tf.keras.layers.LSTM(rnn_units,
#                          return_sequences=True,
#                          stateful=True,
#                          dropout=0.3,
#                          recurrent_initializer='glorot_uniform'),

#      tf.keras.layers.Dense(128, activation='relu'),
#      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Dense(vocab_size, activation='softmax'),
#      tf.keras.layers.Dense(vocab_size),
    ], name='DeepComedy')
    
    
    def loss(labels, logits):
        return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(loss="sparse_categorical_crossentropy", metrics="accuracy", optimizer=optimizer)
#    model.compile(loss=loss, metrics="accuracy", optimizer=optimizer)
    
    model.summary()

    return model
