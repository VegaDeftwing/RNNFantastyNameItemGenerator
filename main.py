import tensorflow as tf
from datetime import datetime
from model import GenerativeGRU, OneStep
from util import load_all_the_data
from rich import pretty, print
import os
import nltk
import random


def main():
    input_path = 'final/items_now_adj_big.csv'
    vocab = load_all_the_data(input_path)
    # similarity_metric = WordSimilarityMetric(vocab.tolist())
    vocab = tf.strings.unicode_split(vocab, input_encoding='UTF-8')
    data = vocab.to_tensor()
    ids_from_chars = tf.keras.layers.experimental.preprocessing.StringLookup()
    ids_from_chars.adapt(vocab)
    chars_from_ids = tf.keras.layers.experimental.preprocessing.StringLookup(
        vocabulary=ids_from_chars.get_vocabulary(), invert=True)

    def text_from_ids(ids):
        return tf.strings.reduce_join(chars_from_ids(ids), axis=-1)

    all_ids = ids_from_chars(data)

    ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)

    def split_input_target(sequence):
        input_text = sequence[:-1]
        target_text = sequence[1:]
        return input_text, target_text

    dataset = ids_dataset.map(split_input_target)

    # Batch size
    BATCH_SIZE = 1024

    # Buffer size to shuffle the dataset
    BUFFER_SIZE = 1000000

    dataset = (
        dataset
        .shuffle(BUFFER_SIZE)
        .batch(BATCH_SIZE, drop_remainder=True)
        .prefetch(tf.data.experimental.AUTOTUNE))

    # Length of the vocabulary in chars
    vocab_size = len(ids_from_chars.get_vocabulary())

    # The embedding dimension
    embedding_dim = 64

    # Number of RNN units
    rnn_units = 512

    model = GenerativeGRU(vocab_size, embedding_dim, rnn_units)

    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=['accuracy']
    )

    checkpoint_path = './training_checkpoints_512_big_embed_64'
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
            dataset, epochs=40,
            callbacks=[checkpoint_callback, tensorboard_callback])

    temperature = 1.0
    noise_weight = 0.0

    one_step_model = OneStep(
        model, chars_from_ids, ids_from_chars, temperature, noise_weight)

    states = None

    NUM_OF_EXAMPLES = 100
    MAX_NUM_CHARS = 12
    MAX_NUM_WORDS = 7
    MIN_NUM_WORDS = 3
    i = 0
    while i < NUM_OF_EXAMPLES:
        seed = random.randint(0, 25)
        next_char = tf.constant([string.ascii_letters[seed]])
        # next_char = tf.constant(['sword'])
        result = [next_char]
        num_words = 0
        num_chars = 0
        idx = 0
        words = []
        states = None
        should_print = True
        while(num_words < MAX_NUM_WORDS):
            next_char, states = one_step_model.generate_one_step(
                next_char, states=states)
            result.append(next_char)
            idx += 1
            num_chars += 1
            if (next_char == ' '):
                num_chars = 0
                words.append(
                    tf.strings.join(result)[0]
                    .numpy()
                    .decode('UTF-8')
                    .strip()
                )
                num_words += 1
                result = []
                if num_words > MIN_NUM_WORDS:
                    tag = nltk.pos_tag(words)[-1][1]
                    if 'NN' in tag or 'VB' in tag:
                        break
            if (next_char == '-'):
                num_chars = 0
            if num_chars > MAX_NUM_CHARS:
                should_print = False
                i -= 1
                break
        i += 1
        if should_print:
            # print(similarity_metric(' '.join(words)))
            print(' '.join(words))

    return


if __name__ == "__main__":
    pretty.install()
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only allocate 6.2GB of memory (3070)
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


def three_char_groups():
    x = ['abc', 'defg', 'hijklm']
    valid_list = []

    for word in x:
        while len(word) > 2:
            valid_list.append(word[:3])
            word = word[1:]
    print(valid_list)
