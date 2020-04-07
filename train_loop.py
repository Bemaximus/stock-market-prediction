from os import makedirs

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import numpy as np

from ml_lib.lstm_net import LSTM_Net
from ml_lib.metrics import calc_r2


class TrainLoop:
    def __init__(self,
                 inputs=None,
                 labels=None,
                 train_split=0.8,
                 val_split=0.1,
                 fc=[128],
                 input_dim=None,
                 output_dim=1,
                 hidden_dim=128,
                 n_layers=1,
                 dropout=0.5,
                 out_actv=None,
                 lr=0.005,
                 batch_size=128,
                 l2_penalty=0,
                 log_period=25,
                 save_period=100,
                 output_dir="/tmp/model_output",
                 wandb=False,
                 wandb_name=None,
                 wandb_project=None):
        # Build model, loss function and optimizer
        self.model = LSTM_Net(fc=fc,
                              input_dim=input_dim,
                              output_dim=output_dim,
                              hidden_dim=hidden_dim,
                              n_layers=n_layers,
                              dropout=dropout,
                              out_actv=out_actv)
        self.loss_func = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=lr,
                                          weight_decay=l2_penalty)

        # Hyperparameters
        self.batch_size = batch_size

        # Build DataLoaders
        self.train_loader, self.test_loader, self.val_loader = \
                self.build_loaders(inputs, labels, train_split, val_split)

        # Logging parameters
        self.log_period = log_period
        self.save_period = save_period
        self.wandb = wandb

        # Create output directory is it doesn't exist
        makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir

        # Setup wandb
        if wandb:
            import wandb
            if wandb_name != None:
                wandb.init(name=wandb_name,
                           project=wandb_project)
            else:
                wandb.init(project=wandb_project)
            wandb.watch(model)
            self.wandb = True
        else: self.wandb = False

    def build_loaders(self, inputs, labels, train_split, val_split):
        # Calculate test_size to pass to sklearn to get the split we want
        assert train_split + val_split <= 1
        val_test_size = 1 - train_split
        test_size = val_split / val_test_size
       
        # Split into test / validation / train
        train_inputs, test_inputs, train_labels, test_labels = \
            train_test_split(inputs, labels, test_size=val_test_size)
        val_inputs, test_inputs, val_labels, test_labels = \
            train_test_split(test_inputs, test_labels, test_size=test_size)

        # Build TensorDatasets
        train = TensorDataset(torch.from_numpy(train_inputs),
                              torch.from_numpy(train_labels))
        val = TensorDataset(torch.from_numpy(val_inputs),
                            torch.from_numpy(val_labels))
        test = TensorDataset(torch.from_numpy(test_inputs),
                             torch.from_numpy(test_labels))

        # Build DataLoaders
        train_loader = DataLoader(train, shuffle=True,
                                  batch_size=self.batch_size, drop_last=True)
        test_loader = DataLoader(test, shuffle=True,
                                 batch_size=self.batch_size, drop_last=True)
        val_loader = DataLoader(val, shuffle=True,
                                batch_size=self.batch_size, drop_last=True)

        return train_loader, test_loader, val_loader

    def train(self, epochs):
        self.model.train()
        self.step = 0
        # Setup logging metrics
        metrics = ("Loss", "R2")
        self.train_metrics = {m: [] for m in metrics}
        self.val_metrics = {m: [] for m in metrics}

        print("Training start!")
        for e in range(epochs):
            self.run_epoch(e)
        name = f'state_dict_final.pt'
        self.model.save(name, self.output_dir)

    def run_epoch(self, e, val=False):
        # Iterate over one epoch
        if val: self.model.eval()
        # Grab appropriate loader and metrics for evaluation type
        loader = self.val_loader if val else self.train_loader
        metrics = self.val_metrics if val else self.train_metrics
        for titles, labels in loader:
            if not val:
                self.step += 1
                self.model.zero_grad()

            # Get output and calculate loss
            output = self.model(titles)
            loss = self.loss_func(output, labels.float())

            # Calculate metrics
            r2 = calc_r2(output, labels)

            # Save logging metrics
            metrics["Loss"].append(loss.item())
            metrics["R2"].append(r2)

            # Calculate and apply gradient
            if not val:
                loss.backward()
                self.optimizer.step()

            # If log period, validate model and log
            if not val and self.step % self.log_period == 0:
                self.run_epoch(e, val=True)
                self.log(self.train_metrics, self.val_metrics, e)
                # Reset metrics after logging
                metrics = self.train_metrics = {key: [] for key in metrics}
                self.val_metrics = {key: [] for key in metrics}
            # If save period, save model
            if not val and self.step % self.save_period == 0:
                name = f'state_dict_{e}_{self.step}.pt'
                self.model.save(name, self.output_dir)
        if val: self.model.eval()
        return metrics
    
    def log(self, train_metrics, val_metrics, e):
        # Calculate averages
        train_metrics = {k:np.mean(m) for k,m in train_metrics.items()}
        val_metrics = {k:np.mean(m) for k,m in val_metrics.items()}

        # Print metrics
        print(f"Epoch: {e} | Step: {self.step}")
        for metric in train_metrics:
            train_avg, val_avg = train_metrics[metric], val_metrics[metric]
            print(f"{metric} | Train: {train_avg:.6f} | Val: {val_avg:.6f}")
        print("")

        # If wandb is running, log metrics
        # Hardcoded for now, think about changing in the future
        if self.wandb:
            wandb.log({"Train Loss": train_metrics["Loss"],
                       "Train R2": train_metrics["R2"],
                       "Validation Loss": train_metrics["Loss"],
                       "Validation R2": val_metrics["R2"],
                       "Step": step})
