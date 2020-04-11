from os import makedirs

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import numpy as np

from ml_lib.lstm_net import LSTM_Net
from ml_lib.metrics import METRIC_FUNCS

LOSS_FUNCS = {"MSE": nn.MSELoss,
              "BCE": nn.BCEWithLogitsLoss}

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
                 loss_func="MSE",
                 metrics=(),
                 log_period=25,
                 save_period=100,
                 output_dir="/tmp/model_output",
                 wandb=False,
                 wandb_name=None,
                 wandb_project=None):
        self.model = LSTM_Net(fc=fc,
                              input_dim=input_dim,
                              output_dim=output_dim,
                              hidden_dim=hidden_dim,
                              n_layers=n_layers,
                              dropout=dropout,
                              out_actv=out_actv)
        self.loss_func = LOSS_FUNCS[loss_func]()
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=lr,
                                          weight_decay=l2_penalty)

        self.batch_size = batch_size

        self.train_loader, self.test_loader, self.val_loader = \
                self.build_loaders(inputs, labels, train_split, val_split)

        self.metrics = ("Loss",) + metrics
        self.log_period = log_period
        self.save_period = save_period
        self.wandb = wandb

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
            wandb.watch(self.model)
            self.wandb = wandb
        else: self.wandb = None

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
        self.train_log = {m: [] for m in self.metrics}
        self.val_log = {m: [] for m in self.metrics}

        print("Training start!")
        for e in range(epochs):
            self.run_epoch(e)
        name = f'state_dict_final.pt'
        self.model.save(name, self.output_dir)

    def run_epoch(self, e, val=False):
        if val: self.model.eval()
        loader = self.val_loader if val else self.train_loader
        metrics_log = self.val_log if val else self.train_log
        for titles, labels in loader:
            if not val:
                self.step += 1
                self.model.zero_grad()

            output = self.model(titles)
            loss = self.loss_func(output, labels.float())

            metrics_log["Loss"].append(loss.item())
            for metric in self.metrics[1:]:
                # Lookup the function for the metric we are evaluating in
                # METRIC_FUNCS, then save the resulting value to the log of that
                # metric in metrics_log
                metrics_log[metric].append(METRIC_FUNCS[metric](labels, output))

            if not val:
                loss.backward()
                self.optimizer.step()

            if not val and self.step % self.log_period == 0:
                self.run_epoch(e, val=True)
                self.log(self.train_log, self.val_log, e)
                # Reset metrics after logging
                metrics_log = self.train_log = {key: [] for key in metrics_log}
                self.val_log = {key: [] for key in metrics_log}
            if not val and self.step % self.save_period == 0:
                name = f'state_dict_{e}_{self.step}.pt'
                self.model.save(name, self.output_dir)
        if val: self.model.eval()
    
    def log(self, train_log, val_log, e):
        train_metrics = {k:np.mean(m) for k,m in train_log.items()}
        val_metrics = {k:np.mean(m) for k,m in val_log.items()}

        print(f"Epoch: {e} | Step: {self.step}")
        for metric in self.metrics:
            train_avg, val_avg = train_metrics[metric], val_metrics[metric]
            print(f"{metric} | Train: {train_avg:.6f} | Val: {val_avg:.6f}")
        print("")

        if self.wandb:
            concat = dict([("Train " + k, v) for k,v in train_metrics.items()] +
                          [("Val " + k, v) for k,v in val_metrics.items()] +
                          [("Step", self.step)])
            self.wandb.log(concat)
