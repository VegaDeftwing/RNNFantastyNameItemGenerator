# Model for Final
import tensorflow as tf


class GenerativeGRU(tf.keras.Model):
    '''
    A GRU based model meant to generate text. It has an embedding layer
    on the inputs and a dense layer on the output.
    '''
    def __init__(self, vocab_size, embedding_dim, rnn_units):
        super().__init__(self)
        self.old_weights = None
        self.embedding = tf.keras.layers.Embedding(
            vocab_size, embedding_dim, mask_zero=True)
        self.gru = tf.keras.layers.GRU(
            rnn_units, return_sequences=True, return_state=True)
        self.dense = tf.keras.layers.Dense(vocab_size)

    def call(self, inputs, states=None, return_state=False, training=False):
        x = inputs
        x = self.embedding(x, training=training)
        if states is None:
            states = self.gru.get_initial_state(x)
        x, states = self.gru(x, initial_state=states, training=training)
        x = self.dense(x, training=training)

        if return_state:
            return x, states
        else:
            return x

    def apply_random(self, noise_weight):
        # if not self.old_weights:
        #     self.old_weights = self.dense.weights
        #     print(self.old_weights)
        #     return
        # print(self.old_weights)
        # self.dense.set_weights(self.old_weights)
        for layer in self.dense.trainable_weights:
            noise = tf.random.normal(layer.shape) * noise_weight
            layer.assign_add(noise)


class OneStep(tf.keras.Model):
    '''
    A one step model wrapper that assumes an RNN based input to generate
    one step of output data. This can be done continuously in a loop to
    generate output based on previous state.
    '''
    def __init__(self,
                 model,
                 chars_from_ids,
                 ids_from_chars,
                 temperature=1.0,
                 noise_weight=0.0):
        super().__init__()
        self.temperature = temperature
        self.model = model
        self.chars_from_ids = chars_from_ids
        self.ids_from_chars = ids_from_chars
        self.noise_weight = noise_weight

        # create a mask to prevent "" or "[UNK]" from being generated
        skip_ids = self.ids_from_chars(['', '[UNK]'])[:, None]
        sparse_mask = tf.SparseTensor(
            # put in -inf at each bad index.
            values=[-float('inf')]*len(skip_ids),
            indices=skip_ids,
            # Match the shape to the vocabulary
            dense_shape=[len(ids_from_chars.get_vocabulary())]
        )
        self.prediction_mask = tf.sparse.to_dense(sparse_mask)

    @tf.function
    def generate_one_step(self, inputs, states=None):
        # Convert strings to token IDs
        input_chars = tf.strings.unicode_split(inputs, 'UTF-8')
        input_ids = self.ids_from_chars(input_chars).to_tensor()

        # Generate noise and apply to model weights
        self.model.apply_random(self.noise_weight)

        # Run model
        predicted_logits, states = self.model(
            inputs=input_ids, states=states, return_state=True)

        # Only use last prediction
        predicted_logits = predicted_logits[:, -1, :]
        predicted_logits = predicted_logits/self.temperature
        # Apply the prediction mask
        predicted_logits = predicted_logits + self.prediction_mask

        # Sample the output logits to generate token IDs
        predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)
        predicted_ids = tf.squeeze(predicted_ids, axis=-1)

        # Convert from token ids to chars
        predicted_chars = self.chars_from_ids(predicted_ids)

        return predicted_chars, states


class SensibleTextOutputMetric():
    def __init__(self,
                 name='sensible_text_output_metric',
                 dtype=None,
                 from_logits=False,
                 axis=-1):
        self.name = name
        self.dtype = dtype
        self.from_logits = from_logits
        self.axis = axis
        return

    def __call__(self):
        return
