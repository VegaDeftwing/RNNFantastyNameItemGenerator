# Model for Final

# Math and ML libs
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# util libs
from rich import pretty, print
import os


class GenerativeGRU(tf.keras.Model):
    '''
    A GRU based model meant to generate text.
    It has an embedding layer on the inputs and a dense layer on the output
    '''
    def __call__(self, vocab_size, embedding_dim, rnn_units):
        super().__init__(self)
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim) #NOTE: Support more params in future
        self.gru = tf.keras.layers.GRU(rnn_units, return_sequences=True, return_state=True)
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
