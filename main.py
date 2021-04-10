import tensorflow as tf
import numpy as np
from datetime import datetime
from model import GenerativeGRU, OneStep
import util
from rich import pretty, print
import os


def main():
    input_path = 'final/items_now_adj_big.csv'
    vocab = util.load_all_the_data(input_path)
    vocab = tf.strings.unicode_split(vocab, input_encoding='UTF-8')
    data = vocab.to_tensor()
    ids_from_chars = tf.keras.layers.experimental.preprocessing.StringLookup()
    ids_from_chars.adapt(vocab)
    # ids = ids_from_chars(data)
    chars_from_ids = tf.keras.layers.experimental.preprocessing.StringLookup(
        vocabulary=ids_from_chars.get_vocabulary(), invert=True)
    # chars = chars_from_ids(ids)

    def text_from_ids(ids):
        return tf.strings.reduce_join(chars_from_ids(ids), axis=-1)

    all_ids = ids_from_chars(data)

    ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)

    # for ids in ids_dataset.take(10):
    #     print(chars_from_ids(ids).numpy().astype('U13'))

    # seq_length = 50

    # examples_per_epoch = len(data.numpy().flatten())//(seq_length+1)
    # print(examples_per_epoch)

    # sequences = ids_dataset.batch(seq_length+1, drop_remainder=True)

    # print(sequences)

    # for seq in ids_dataset.take(1):
    #     print(chars_from_ids(seq))

    # for seq in ids_dataset.take(5):
    #     print(text_from_ids(seq).numpy())

    def split_input_target(sequence):
        input_text = sequence[:-1]
        target_text = sequence[1:]
        return input_text, target_text

    dataset = ids_dataset.map(split_input_target)

    # for input_example, target_example in dataset.take(1):
    #     print("Input :", text_from_ids(input_example).numpy())
    #     print("Target:", text_from_ids(target_example).numpy())

    # Batch size
    BATCH_SIZE = 512

    # Buffer size to shuffle the dataset
    BUFFER_SIZE = 1000000

    dataset = (
        dataset
        .shuffle(BUFFER_SIZE)
        .batch(BATCH_SIZE, drop_remainder=True)
        .prefetch(tf.data.experimental.AUTOTUNE))

    # print(dataset)
    # Length of the vocabulary in chars
    vocab_size = len(ids_from_chars.get_vocabulary())

    # The embedding dimension
    embedding_dim = vocab_size

    # Number of RNN units
    rnn_units = 512

    model = GenerativeGRU(vocab_size, embedding_dim, rnn_units)

    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=['accuracy']
    )

    checkpoint_path = './training_checkpoints_512_big'
    if os.path.isdir(checkpoint_path) and len(os.listdir(checkpoint_path)) > 0:
        latest = tf.train.latest_checkpoint(checkpoint_path)
        model.load_weights(latest).expect_partial()
        chkpt = tf.train.Checkpoint(model)
        chkpt.restore(latest).expect_partial()
    else:
        checkpoint_prefix = os.path.join(checkpoint_path, 'ckpt_{epoch}')
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_prefix, save_weights_only=True)
        # early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        #     monitor='loss', patience=2)
        logdir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
        model.fit(
            dataset, epochs=20,
            callbacks=[checkpoint_callback, tensorboard_callback])

    temperature = 0.75
    noise_weight = 0.0

    one_step_model = OneStep(
        model, chars_from_ids, ids_from_chars, temperature, noise_weight)

    states = None
    seeds = np.random.randint(2, len(ids_from_chars.get_vocabulary()), 100)
    next_char = tf.constant(chars_from_ids(seeds))
    result = [next_char]

    for _ in range(50):
        next_char, states = one_step_model.generate_one_step(
            next_char, states=states)
        result.append(next_char)

    result = tf.strings.join(result)

    for r in result:
        print(r.numpy().decode('UTF-8'))

    return


if __name__ == "__main__":
    pretty.install()
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only allocate 6.2GB of memory
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(
                    memory_limit=6400
                )]
            )
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)
    main()

