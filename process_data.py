import os
import pickle

import glob
import numpy as np
import pandas as pd

def main(filenames, seq_len=10, out_name="processed_dataset.pickle"):
    all_inputs = []
    all_non_seq_inputs = []
    all_labels = []
    all_binary_labels = []
    for filename in filenames:
        print(f"Parsing {filename}...")
        # Get data
        df = pd.read_csv(filename)
        df = df[["Open", "High", "Low", "Close"]]
        df_len = len(df.index)

        inputs = [[] for _ in range(df_len - seq_len)]
        non_seq_inputs = []
        labels = []
        binary_labels = []
        # Skip the first seq_len values because they do not have historic data
        for i in range(seq_len, df_len):
            start = i - seq_len
            norm_factor = df.iloc[start]["Open"]
            for index,row in df.iloc[start:i].iterrows():
                # Normalize each datapoint by the first opening value
                data = [v / norm_factor if norm_factor > 0 else 0 for v in row]
                inputs[i - seq_len].append(data)

            cur = df.iloc[i]
            change = cur["Close"] / cur["Open"]
            non_seq_inputs.append(cur["Open"])
            labels.append(change)
            binary_labels.append(1 if change > 1 else 0)
        all_inputs.extend(inputs)
        all_non_seq_inputs.extend(non_seq_inputs)
        all_labels.extend(labels)
        all_binary_labels.extend(binary_labels)
        print(np.asarray(all_inputs).shape)

    # Convert to numpy arrays and save processed data
    save_dict = {"inputs": np.asarray(all_inputs),
                 "non_seq_inputs": np.asarray(all_non_seq_inputs),
                 "labels": np.asarray(all_labels),
                 "binary_labels": np.asarray(all_binary_labels)}
    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/{out_name}", 'wb') as sf:
        pickle.dump(save_dict, sf)
    print(f"File saved to {output_dir}/{out_name}!")
    print("Final shapes:")
    for k,v in save_dict.items():
        print(f"{k}: {v.shape}")

if __name__ == "__main__":
    # filenames = ["../../data/JNUG.csv"]
    # main(filenames, out_name="JNUG_dataset.csv")

    filenames = glob.glob("../../data/*.csv")
    main(filenames, out_name="large_dataset.csv")
