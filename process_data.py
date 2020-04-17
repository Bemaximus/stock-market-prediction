import os
import pickle

import glob
from tqdm import tqdm
import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

def build_sequences(df, seq_len, var_mean, var_std):
    df_len = len(df.index)
    inputs = []
    non_seq_inputs = []
    labels = []
    binary_labels = []
    # Skip the first seq_len values because they do not have historic data
    for i in tqdm(range(seq_len, df_len)):
        start = i - seq_len
        # norm_factor = df.iloc[start]["Open"]
        sequence = []
        for index,row in df.iloc[start:i].iterrows():
            # Normalize each datapoint by the first opening value
            # data = [v / norm_factor if norm_factor > 0 else 0 for v in row]
            data = [(row[c] - var_mean[c]) / var_std[c] for c in row.keys()]
            sequence.append(data)
        cur = df.iloc[i]
        change = cur["Close"] / cur["Open"]
        inputs.append(sequence)
        non_seq_inputs.append(cur["Open"])
        labels.append(change)
        binary_labels.append(1 if change > 1 else 0)
    return inputs, non_seq_inputs, labels, binary_labels

def build_sequences_async(df, seq_len, var_mean, var_std):
    df_len = len(df.index)
    inputs = []
    non_seq_inputs = []
    labels = []
    binary_labels = []
    # Skip the first seq_len values because they do not have historic data
    for i in tqdm(range(seq_len, df_len)):
        start = i - seq_len
        # norm_factor = df.iloc[start]["Open"]
        sequence = []
        for index,row in df.iloc[start:i].iterrows():
            # Normalize each datapoint by the first opening value
            # data = [v / norm_factor if norm_factor > 0 else 0 for v in row]
            data = [(row[c] - var_mean[c]) / var_std[c] for c in row.keys()]
            sequence.append(data)
        cur = df.iloc[i]
        change = cur["Close"] / cur["Open"]
        inputs.append(sequence)
        non_seq_inputs.append(cur["Open"])
        labels.append(change)
        binary_labels.append(1 if change > 1 else 0)
    return inputs, non_seq_inputs, labels, binary_labels

def calc_mean(all_means, all_counts):
    """
    Combines the means of several groups
    """
    total_values = sum([m * c for m,c in zip(all_means, all_counts)])
    total_counts = sum(all_counts)
    return total_values / total_counts

def calc_std(all_means, all_counts, all_variances, true_mean):
    """
    Combines the variance of multiple groups.
    See here:
    https://www.researchgate.net/post/How_do_I_combine_mean_and_standard_deviation_of_two_groups
    """
    mean_diff = [m - true_mean for m in all_means]
    top = sum([c * (v + d**2) for c,v,d in
               zip(all_counts, all_variances, mean_diff)])
    total_counts = sum(all_counts)
    return np.sqrt(top / total_counts)

def main(filenames, seq_len=10, name="processed_dataset"):
    inputs, non_seq_inputs, labels, binary_labels = [], [], [], []
    variables = ["Open", "High", "Low", "Close", "Volume"]
    var_means = {k:[] for k in variables}
    var_variance = {k:[] for k in variables}
    var_count = {k:[] for k in variables}
    dfs = []
    for filename in filenames:
        print(f"Parsing {filename}...")
        df = pd.read_csv(filename)
        null_count = df.isnull().sum().sum()
        if null_count > 0:
            print(f"{null_count} null values found! Skipping file.")
            break
        df = df[variables]
        for v in variables:
            var_means[v].append(df[v].mean())
            var_variance[v].append(df[v].var())
            var_count[v].append(df[v].size)
        dfs.append(df)

    total_var_means, total_var_std = {}, {}
    for v in variables:
        total_var_means[v] = calc_mean(var_means[v], var_count[v])
        total_var_std[v] = calc_std(var_means[v], var_count[v],
                                    var_variance[v], total_var_means[v])

    for df in dfs:
        _inputs, _non_seq_inputs, _labels, _binary_labels = \
                build_sequences(df, seq_len, total_var_means, total_var_std)
        inputs.extend(_inputs)
        non_seq_inputs.extend(_non_seq_inputs)
        labels.extend(_labels)
        binary_labels.extend(_binary_labels)

    # Convert to numpy arrays and save processed data
    save_dict = {"inputs": np.asarray(inputs),
                 "non_seq_inputs": np.asarray(non_seq_inputs),
                 "labels": np.asarray(labels),
                 "binary_labels": np.asarray(binary_labels)}
    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{name}_{seq_len}.pickle"
    with open(f"{output_dir}/{output_file}", 'wb') as sf:
        pickle.dump(save_dict, sf)
    print(f"File saved to {output_dir}/{output_file}!")
    print("Final shapes:")
    for k,v in save_dict.items():
        print(f"{k}: {v.shape}")

if __name__ == "__main__":
    stock = "AAPL"
    filenames = [f"../../data/{stock}.csv"]
    main(filenames, name=f"{stock}_dataset")

    # filenames = glob.glob("../../data/*.csv")
    # main(filenames, name="large_dataset")
