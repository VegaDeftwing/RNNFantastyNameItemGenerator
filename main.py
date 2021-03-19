import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from model import GenerativeGRU
import util

from rich import pretty, print

def main():
    # model = GenerativeGRU()
    # util.fuck(1)
    data = util.load_all_the_data()


if __name__ == "__main__":
    pretty.install()
    gpus = tf.config.list_physical_devices('GPU')
    if len(gpus) > 0:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    main()
