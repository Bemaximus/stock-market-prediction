from os import makedirs

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import numpy as np

from ml_lib.lstm_net import LSTM_Net


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

        # Build DataLoaders
        self.train_loader, self.test_loader, self.val_loader = \
                build_loaders(inputs, labels, train_split, val_split)

        # Hyperparameters
        self.batch_size = batch_size

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
        val_test_size = 1 - train_size
        test_size = val_size / val_test_size
       
        # Split into test / validation / train
        train_inputs, test_inputs, train_labels, test_labels = \
            train_test_split(inputs, labels, test_size=1-train_size)
        val_inputs, test_inputs, val_labels, test_labels = \
            train_test_split(test_inputs, test_labels, test_size=0.5)

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
        for e in range(epochs):
            run_epoch(e)
        name = f'state_dict_final.pt'
        self.model.save(name, self.output_dir)

    def run_epoch(self, e, val=False):
        # Setup logging metrics
        losses = []
        accuracy = []
        auc = []

        # Iterate over one epoch
        if val: self.model.eval()
        loader = self.val_loader if val else self.train_loader
        for titles, labels in loader:
            if not val:
                self.step += 1
                self.model.zero_grad()

            # Get output and calculate loss
            output = self.model(titles)
            loss = self.loss_func(output, labels.float())

            # Calculate and apply gradient
            if not val:
                loss.backward()
                self.optimizer.step()

            # Calculate logging metrics
            prediction = torch.round(output)
            labels = labels.float().view_as(prediction)
            correct = np.squeeze(prediction.eq(labels).numpy())

            # Save logging metrics
            losses.append(loss.item())
            accuracy.append(correct / self.batch_size)
            auc.append(roc_auc_score(labels, output.detach()))

            # If log period, validate model and log
            if not val and self.step % self.log_period == 0:
                val_losses, val_accuracy, val_auc = run_model(e, val=True)
                self.log(losses, accuracy, auc,
                         val_losses, val_accuracy, val_auc,
                         e)
            # If save period, save model
            if not val and self.step % self.save_period == 0:
                name = f'state_dict_{e}_{step}.pt'
                self.model.save(name, self.output_dir)
        if val: self.model.eval()
    
    def log(self, train_loss, train_acc, train_auc,
            val_loss, val_acc, val_auc, e):
        # Calculate Metrics
        avg_train_loss, avg_train_acc, avg_train_auc = \
                map(np.mean, (train_loss, train_acc, train_auc))
        avg_val_loss, avg_val_acc, avg_val_auc = \
                map(np.mean, (val_loss, val_acc, val_auc))

        # Print Metrics
        print(f"Epoch: {e} | Step: {self.step}")
        print(f"Loss | Train: {avg_train_loss:.6f} |  Val: {avg_val_loss:.6f}")
        print(f"Accuracy | Train: {avg_train_acc:.6f} | Val: {avg_val_acc:.6f}")
        print(f"AUROC | Train: {avg_train_auc:.6f} | Val: {avg_val_auc:.6f}")
        print("")

        # If wandb is running, log metrics
        if self.wandb:
            wandb.log({"Train Accuracy": avg_train_acc,
                       "Train Loss": avg_train_loss,
                       "Train AUC": avg_train_auc,
                       "Validation Accuracy": avg_val_acc,
                       "Validation Loss": avg_val_loss,
                       "Validation AUC": avg_val_auc,
                       "Step": step})
