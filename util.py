# Math and ML libs
# import tensorflow as tf
import pandas as pd


def load_all_the_data(path: str):
    # Need to load in the data... as a csv
    df = pd.read_csv(path, header=None, names=['names'])
    df = df.astype('str')
    df = df['names'].str.lower()
    # df.info(memory_usage='deep')
    # df = df.drop_duplicates()
    # df = df.drop(df[df['names'].str.contains('/|_|\\\\', regex=True)].index)
    # indeces = df[ (df)]
    # df.info(memory_usage='deep')
    # dropped.info(memory_usage='deep')
    # print(df.shape[0] - dropped.shape[0])
    # df = df.sort_values('names')
    # df.to_csv('final/dropped_data.csv', index=False, header=False)
    # print(df.to_numpy())

    return df.to_numpy().flatten()


if __name__ == "__main__":
    load_all_the_data()
