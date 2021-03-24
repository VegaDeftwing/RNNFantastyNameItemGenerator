
# Math and ML libs
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# util libs
from rich import pretty, print
import os

#hahahha

def fuck(how_many):
    import webbrowser
    print("Ha, i give no fucks, get stick bugged")
    webbrowser.open("https://www.youtube.com/embed/Tt7bzxurJ1I")
    exit(-1)

def load_all_the_data():
    # Need to load in the data... as a csv
    df = pd.read_csv('final/data.csv', header=None, names=['names'])
    df = df.drop_duplicates()
    # df.info(memory_usage='deep')
    df = df.drop(df[df['names'].str.contains('/|_|\\\\', regex=True)].index)
    # indeces = df[ (df)]
    # df.info(memory_usage='deep')

    # dropped.info(memory_usage='deep')

    # print(df.shape[0] - dropped.shape[0])

    # df.to_csv('dropped_data.csv', index=False, header=False)

    # print(df.to_numpy())

    return df.to_numpy().flatten().tolist()
