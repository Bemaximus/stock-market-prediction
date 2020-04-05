import os
import pickle

import numpy as np
import pandas as pd

def main(filename, seq_len=10):
    # Get data
    df = pd.read_csv(filename)
    df = df[["Open", "High", "Low", "Close"]]
    df_len = len(df.index)

    inputs = [[] for _ in range(df_len - seq_len)]
    non_seq_inputs = []
    labels = []
    # Skip the first seq_len values because they do not have historic data
    for i in range(seq_len, df_len):
        start = i - seq_len
        norm_factor = df.iloc[start]["Open"]
        for index,row in df.iloc[start:i].iterrows():
            # Normalize each datapoint by the first opening value
            data = [v / norm_factor for v in row]
            inputs[i - seq_len].append(data)

        cur = df.iloc[i]
        non_seq_inputs.append(cur["Open"])
        labels.append(cur["Close"] / cur["Open"])

    # Convert to numpy arrays and save processed data
    save_dict = {"inputs": np.asarray(inputs),
                 "non_seq_inputs": np.asarray(non_seq_inputs),
                 "labels": np.asarray(labels)}
    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/processed_dataset.pickle", 'wb') as sf:
        pickle.dump(save_dict, sf)

if __name__ == "__main__":
    filename = "../../data/AAPL.csv"
    main(filename)
