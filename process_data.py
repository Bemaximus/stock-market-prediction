import os
import argparse
import pickle

import glob
from tqdm import tqdm
import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

def build_sequences(df, seq_len, var_mean, var_std):
    """
    Builds sequences of prices from a dataframe.

    Parameters:
        df: Dataframe of stock prices to build from
        seq_len: Length of the sequence to build
        var_mean: Dict containing the mean of each variable in the dataframe,
        calculated over all the data being used
        var_std: Dict containing the std deviation  of each variable in the
        dataframe, calculated over all the data being used
    """
    df_len = len(df.index)
    inputs = []
    non_seq_inputs = []
    labels = []
    binary_labels = []
    # Skip the first seq_len values because they do not have historic data
    for i in tqdm(range(seq_len, df_len)):
        start = i - seq_len
        norm_factors = {k: ((v - var_mean[k]) / var_std[k])
                        for k,v in df.iloc[start].iteritems()}
        sequence = []
        for index,row in df.iloc[start:i].iterrows():
            # Normalize each datapoint by overall mean, overall std deviation
            # and the first opening value
            data = [((v - var_mean[k]) / var_std[k]) / norm_factors[k]
                    if norm_factors[k] != 0 else 0
                    for k,v in row.iteritems()]
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

def main(filenames, seq_len=10, name="processed_dataset", train_size=0.8):
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
        train, val = train_test_split(df, train_size=train_size)
        for v in variables:
            var_means[v].append(train[v].mean())
            var_variance[v].append(train[v].var())
            var_count[v].append(train[v].size)
        dfs.append((train, val))
    print()

    total_var_means, total_var_std = {}, {}
    for v in variables:
        total_var_means[v] = calc_mean(var_means[v], var_count[v])
        total_var_std[v] = calc_std(var_means[v], var_count[v],
                                    var_variance[v], total_var_means[v])

    t_inputs, t_non_seq_inputs, t_labels, t_binary_labels = [], [], [], []
    v_inputs, v_non_seq_inputs, v_labels, v_binary_labels = [], [], [], []
    for train, val in dfs:
        print("Building train sequences...")
        _t_inputs, _t_non_seq_inputs, _t_labels, _t_binary_labels = \
                build_sequences(train, seq_len, total_var_means, total_var_std)
        t_inputs.extend(_t_inputs)
        t_non_seq_inputs.extend(_t_non_seq_inputs)
        t_labels.extend(_t_labels)
        t_binary_labels.extend(_t_binary_labels)

        print("Building validation sequences...")
        _v_inputs, _v_non_seq_inputs, _v_labels, _v_binary_labels = \
                build_sequences(val, seq_len, total_var_means, total_var_std)
        v_inputs.extend(_v_inputs)
        v_non_seq_inputs.extend(_v_non_seq_inputs)
        v_labels.extend(_v_labels)
        v_binary_labels.extend(_v_binary_labels)
    print()

    # Convert to numpy arrays and save processed data
    save_dict = {"train": {"inputs": np.asarray(t_inputs),
                           "non_seq_inputs": np.asarray(t_non_seq_inputs),
                           "labels": np.asarray(t_labels),
                           "binary_labels": np.asarray(t_binary_labels)},
                 "val": {"inputs": np.asarray(v_inputs),
                         "non_seq_inputs": np.asarray(v_non_seq_inputs),
                         "labels": np.asarray(v_labels),
                         "binary_labels": np.asarray(v_binary_labels)}}
    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{name}_{seq_len}.pickle"
    with open(f"{output_dir}/{output_file}", 'wb') as sf:
        pickle.dump(save_dict, sf)
    print(f"File saved to {output_dir}/{output_file}!\n")
    for k,d in save_dict.items():
        print(f"Final {k} shapes:")
        for var,data in d.items():
            print(f"\t{var}: {data.shape}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Process csv into sequence")
    parser.add_argument('--dir',
                        type=str,
                        default=None,
                        help="Directory to full csv files from")
    parser.add_argument('--file',
                        type=str,
                        default=None,
                        help="File to pull data from")
    parser.add_argument('--seq',
                        type=int,
                        default=10,
                        help="Sequence length (default: 10)")
    parser.add_argument('--name',
                        type=str,
                        default="processed_data",
                        help="Output file name (default: processed_data)")
    args = parser.parse_args()

    if args.dir:
        filenames = glob.glob(f"{args.dir}/*.csv")
    elif args.file:
        filenames = [args.file]
    else:
        raise ValueError("Please specify a directory or file to pull from")
    main(filenames, name=args.name, seq_len=args.seq)
