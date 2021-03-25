import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from model import GenerativeGRU
import util

from rich import pretty, print

def main():
    vocab = util.load_all_the_data()
    # print(vocab[:100])
    vocab = tf.strings.unicode_split(vocab, input_encoding='UTF-8')
    data = vocab.to_tensor()
    ids_from_chars = tf.keras.layers.experimental.preprocessing.StringLookup()
    ids_from_chars.adapt(vocab)
    ids = ids_from_chars(data)
    chars_from_ids = tf.keras.layers.experimental.preprocessing.StringLookup(vocabulary=ids_from_chars.get_vocabulary(), invert=True)
    chars = chars_from_ids(ids)

    def text_from_ids(ids):
        return tf.strings.reduce_join(chars_from_ids(ids), axis=-1)

    all_ids = ids_from_chars(data)

    ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)


    for ids in ids_dataset.take(10):
        print(chars_from_ids(ids).numpy().astype('U13'))
    
    # seq_length = 50

    # examples_per_epoch = len(data.numpy().flatten())//(seq_length+1)
    # print(examples_per_epoch)

    # sequences = ids_dataset.batch(seq_length+1, drop_remainder=True)

    # print(sequences)

    for seq in ids_dataset.take(1):
        print(chars_from_ids(seq))

    for seq in ids_dataset.take(5):
        print(text_from_ids(seq).numpy())
    
    def split_input_target(sequence):
        input_text = sequence[:-1]
        target_text = sequence[1:]
        return input_text, target_text
    
    dataset = ids_dataset.map(split_input_target)

    for input_example, target_example in dataset.take(1):
        print("Input :", text_from_ids(input_example).numpy())
        print("Target:", text_from_ids(target_example).numpy())
    
    # Batch size
    BATCH_SIZE = 128

    # Buffer size to shuffle the dataset
    BUFFER_SIZE = 10000

    dataset = (
        dataset
        .shuffle(BUFFER_SIZE)
        .batch(BATCH_SIZE, drop_remainder=True)
        .prefetch(tf.data.experimental.AUTOTUNE))

    print(dataset) 
    # Length of the vocabulary in chars
    vocab_size = len(ids_from_chars.get_vocabulary())

    # The embedding dimension
    embedding_dim = 256

    # Number of RNN units
    rnn_units = 2048

    model = GenerativeGRU(vocab_size, embedding_dim, rnn_units)

    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=['accuracy'])
    
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=2)
    model.fit(dataset.take(10000//128), epochs=20, callbacks=[early_stopping_callback])

    for input_example_batch, target_example_batch in dataset.take(1):
        example_batch_predictions = model(input_example_batch)
        print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")

    for i in range(len(example_batch_predictions)):
        sampled_indices = tf.random.categorical(example_batch_predictions[i], num_samples=1)
        sampled_indices = tf.squeeze(sampled_indices, axis=-1).numpy()
        print(f"Input: {text_from_ids(input_example_batch[i]).numpy()}")
        print(f"Next Char Predictions: {text_from_ids(sampled_indices).numpy()}")
        print()
    
    util.fuck(1)
    return


if __name__ == "__main__":
    pretty.install()
    gpus = tf.config.list_physical_devices('GPU')
    if len(gpus) > 0:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    main()
