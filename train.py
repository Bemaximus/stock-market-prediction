import os
import pickle
import argparse

import wandb
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from strategies.rnn.ml_lib.lstm_net import LSTM_Net

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

##############################
# TODO REFACTOR TO AN OBJECT #
##############################

def build_model(fc, hidden_dim, n_layers, lr, penalty):
    """
    Builds the LSTM model, as well as the loss function and optimizer

    Parameters:
        hidden_dim: Hidden dimension in the LSTM
        n_layers: Number of layers of the LSTM
        lr: Learning rate
        penalty: L2 regularization penalty
    """
    output_size = 1
    model = LSTM_Net(vocab_size,
                     output_size,
                     hidden_dim,
                     n_layers)
    bce = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=penalty)
    return model, bce, optimizer

def build_loaders(titles, labels, batch_size,
                  under_sample=None, over_sample=None):
    """
    Builds DataLoaders for train test validation. Splits data and over or
    undersamples if asked for

    Parameters:
        titles: numpy array of processed titles
        labels: numpy array of labels for corresponding titles
        batch_size: Batch size to batch data into
        under_sample: Nearest neighbhours hyperparemter for ENN. None if not
                      being used.
        over_sample: Minority class / majority class ratio for SMOTE. None if
                     not being used
    """
    train_titles, test_titles, train_labels, test_labels = \
        train_test_split(titles, labels, test_size=0.2)
    val_titles, test_titles, val_labels, test_labels = \
        train_test_split(test_titles, test_labels, test_size=0.5)

    steps = []
    if under_sample:
        steps.append(("Under", EditedNearestNeighbours(n_neighbors=under_sample)))
    if over_sample:
        steps.append(("Over", SMOTE(sampling_strategy=over_sample)))
    if under_sample or over_sample:
        pipeline = Pipeline(steps=steps)
        train_titles, train_labels = pipeline.fit_resample(train_titles,
                                                           train_labels)

    train = TensorDataset(torch.from_numpy(train_titles),
                          torch.from_numpy(train_labels))
    val = TensorDataset(torch.from_numpy(val_titles),
                        torch.from_numpy(val_labels))
    test = TensorDataset(torch.from_numpy(test_titles),
                         torch.from_numpy(test_labels))

    train_loader = DataLoader(train,
                              shuffle=True,
                              batch_size=batch_size,
                              drop_last=True)
    test_loader = DataLoader(test,
                             shuffle=True,
                             batch_size=batch_size,
                             drop_last=True)
    val_loader = DataLoader(val,
                            shuffle=True,
                            batch_size=batch_size,
                            drop_last=True)

    return train_loader, test_loader, val_loader

def logging(train_losses, train_num_correct, train_num_pred, train_auc,
            val_losses, val_num_correct, val_num_pred, val_auc,
            e, step, wandb):
    """
    Prints and logs training metrics

    Paramters:
        train_losses: List of train losses
        train_num_correct: Number of training correct predictions
        train_num_pred: Number of train predictions
        train_auc: List of train auc
        val_losses: List of validation losses
        val_num_correct: Number of validation correct predictions
        val_num_pred: Number of validation predictions
        val_auc: List of validation auc
        e: Current epoch
        step: Starting step
        wandb: WandB library. None if not being used
    """
    # Calculate Metrics
    avg_val_loss = np.mean(val_losses)
    val_accuracy = val_num_correct / val_num_pred
    avg_val_auc = np.mean(val_auc)
    avg_train_loss = np.mean(train_losses)
    train_accuracy = train_num_correct / train_num_pred
    avg_train_auc = np.mean(train_auc)

    # Print Metrics
    print(f"Epoch: {e} | Step: {step}")
    print(f"Train Accuracy: {train_accuracy:.6f} | Validation Accuracy: {val_accuracy:.6f}")
    print(f"Train Loss: {avg_train_loss:.6f} |  Validation Loss: {avg_val_loss:.6f}")
    print(f"Train AUROC: {avg_train_auc:.6f} | Validation AUROC: {avg_val_auc:.6f}")
    print("")

    # If wandb is running, log metrics
    if wandb:
        wandb.log({"Train Accuracy": train_accuracy,
                   "Train Loss": avg_train_loss,
                   "Train AUC": avg_train_auc,
                   "Validation Accuracy": val_accuracy,
                   "Validation Loss": avg_val_loss,
                   "Validation AUC": avg_val_auc,
                   "Step": step})

def run_model(e, step, log_period, save_period, output_dir, wandb, model,
              optimizer, bce, train_loader, val_loader, batch_size, val=False):
    """
    Runs the model on 1 epoch of data. Doesn't update, log or save if in
    validation mode.

    Parameters:
        e: Current epoch
        step: Starting step
        log_period: How many steps to weight before logging
        save_period: How many steps to weight before saving
        output_dir: Directory to save to
        wandb: WandB library. None if not being used
        model: Model being trained
        optimizer: Optimizer for the model
        bce: Loss equation for the model
        train_loader: Dataloader for training data
        val_loader: Dataloder for validation data
        batch_size: Data batch size
        val: Validation mode. Default False

    Returns:
        step: Ending step
        losses: List of losses over the epoch
        num_correct: Number of correct predictions
        num_pred: Number of predictions
        auc: List of auc over the epoch
    """
    # Setup logging metrics
    losses = []
    num_correct = 0
    num_pred = 0
    auc = []

    if val: model.eval()
    loader = val_loader if val else train_loader
    for titles, labels in loader:
        # Initialize LSTM hidden state
        model.hidden = model.init_hidden(batch_size)
        
        if not val:
            step += 1
            model.zero_grad()

        # Get output and calculate loss
        output = model(titles)
        loss = bce(output, labels.float())
        losses.append(loss.item())

        if not val:
            # Calculate and apply gradient
            loss.backward()
            optimizer.step()

        # Calculate logging metrics
        prediction = torch.round(output)
        labels = labels.float().view_as(prediction)
        auc.append(roc_auc_score(labels, output.detach()))
        correct = prediction.eq(labels)
        correct = np.squeeze(correct.numpy())
        num_correct += np.sum(correct)
        num_pred += output.size(0)

        if not val:
            # If log period, validate model and log
            if step % log_period == 0:
                _, val_losses, val_num_correct, val_num_pred, val_auc = \
                        run_model(e, step, log_period, save_period, output_dir,
                                  wandb, model, optimizer, bce,
                                  train_loader, val_loader, batch_size, val=True)
                logging(losses, num_correct, num_pred, auc,
                        val_losses, val_num_correct, val_num_pred, val_auc,
                        e, step, wandb)
            # If save period, save model
            if step % save_period == 0:
                _output_file = f'{output_dir}/state_dict_{e}_{step}.pt'
                print(f"Model saved to {_output_file}")
                torch.save(model.state_dict(), _output_file)
    if val: model.train()

    return step, losses, num_correct, num_pred, auc

def main(args):
    """
    Loads data, build and trains the model

    Parameters:
        args: Arguments passed to the script
    """
    # Unpack hyperparameters from args
    batch_size = args.batch_size
    epochs = args.epochs
    log_period = args.log_period
    save_period = args.save_period
    hidden_dim = args.hidden_dim
    n_layers = args.n_layers
    lr = args.lr
    penalty = args.penalty
    over_sample = args.over_sample
    under_sample = args.under_sample
    output_dir = args.output_dir

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Create train test validation splits
    train_loader, test_loader, val_loader = build_loaders(titles,
                                                          labels,
                                                          batch_size,
                                                          over_sample=over_sample,
                                                          under_sample=under_sample)

    # Create model
    model, bce, optimizer = build_model(word2idx, embeddings, hidden_dim,
                                        n_layers, lr, penalty)
    
    # Setup wandb if in arguments
    if args.wandb:
        import wandb
        if args.wandb_name != None:
            wandb.init(name=args.wandb_name,
                       project=args.wandb_project)
        else:
            wandb.init(project=args.wandb_project)
        wandb.watch(model)
    else: wandb = None
    
    # Train model!
    step = 0
    model.train()
    for e in range(epochs):
        step, losses, num_correct, num_pred, auc = \
                run_model(e, step, log_period, save_period, output_dir, wandb,
                          model, optimizer, bce, train_loader, val_loader,
                          batch_size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train LSTM Net")
    # Str arguments
    parser.add_argument('--output-dir',
                        type=str,
                        default='/tmp/aita',
                        help="directory to save model to (default:/tmp/aita)")
    parser.add_argument('--data',
                        type=str,
                        default='data/processed_dataset.pickle',
                        help="saved preprocessed data (default:\
                        data/processed_data.pickle)")

    # Numerical arguments
    parser.add_argument('--batch-size',
                        type=int,
                        default=512,
                        help='input batch size for training (default: 512)')
    parser.add_argument('--epochs',
                        type=int,
                        default=100,
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr',
                        type=float,
                        default=0.005,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--penalty',
                        type=float,
                        default=0,
                        help="L2 regularization coefficient (default: 0)")
    parser.add_argument("--hidden-dim",
                        type=int,
                        default=512,
                        help="LSTM hidden dimensions (default: 512)")
    parser.add_argument("--n-layers",
                        type=int,
                        default=1,
                        help="LSTM layers (default: 1)")
    parser.add_argument('--seed',
                        type=int,
                        default=1,
                        help='random seed (default: 1)')
    parser.add_argument('--log-period',
                        type=int,
                        default=50,
                        help='how many update steps to wait before logging\
                        (default: 100)')
    parser.add_argument('--save-period',
                        type=int,
                        default=500,
                        help='how many update steps to wait before saving\
                        (default: 500)')

    # WandB flags
    parser.add_argument('--wandb',
                        default=False,
                        action='store_true',
                        help="log the training with WandB. REQUIRES INSTALLING,\
                        CONFIGURING AND LOGGING INTO WANDB")
    parser.add_argument('--wandb-name',
                        type=str,
                        default=None,
                        help="WandB run name")
    parser.add_argument('--wandb-project',
                        type=str,
                        default="aita_classifier",
                        help="WandB project name (default: aita_classifier)")
    args = parser.parse_args()
    main(args)
