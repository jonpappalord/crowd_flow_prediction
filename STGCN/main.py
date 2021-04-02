import torch
from dataset import Dataset
import matplotlib.pyplot as plt
import numpy as np
from os.path import join as pjoin
import pandas as pd
import torch.nn as nn
from torch.utils.data import Subset

from stgcn import STGCN
from data_loader.data_utils import data_gen
from utils import get_normalized_adj
from utils.math_graph import weight_matrix

num_timesteps_input = 12
num_timesteps_output = 3

epochs = 15
batch_size = 50

def train_epoch(training_generator, batch_size):
    """
    Trains one epoch with the given data.
    :param training_input: Training inputs of shape (num_samples, num_nodes,
    num_timesteps_train, num_features).
    :param training_target: Training targets of shape (num_samples, num_nodes,
    num_timesteps_predict).
    :param batch_size: Batch size to use during training.
    :return: Average loss for this epoch.
    """
    # permutation = torch.randperm(training_input.shape[0])

    epoch_training_losses = []
    # Training
    for local_batch, local_labels in training_generator:
        net.train()
        optimizer.zero_grad()

        # Transfer to GPU
        local_batch, local_labels = local_batch.to(device, dtype=torch.float), local_labels.to(device, dtype=torch.float)

        out = net(A_wave, local_batch)
        loss = loss_criterion(out, local_labels)
        loss.backward()
        optimizer.step()
        epoch_training_losses.append(loss.detach().cpu().numpy())
    return sum(epoch_training_losses)/len(epoch_training_losses)

if __name__ == '__main__':
    torch.manual_seed(7)

    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True

    # Parameters
    params = {'batch_size': batch_size,
            'shuffle': False,
            'num_workers': 2}
    max_epochs = 100
    n = 228
    n_his = 9
    n_pred = 12

    # Loading weight matrix
    W = weight_matrix(pjoin('./data', f'W_{n}.csv'))

    # A_wave = get_normalized_adj(W)
    # A_wave = torch.from_numpy(A_wave)
    A_wave = torch.from_numpy(W)
    A_wave = A_wave.to(device=device, dtype=torch.float)

    data_file = f'V_{n}.csv'
    n_train, n_val, n_test = 34, 5, 5
    X = data_gen(pjoin('./data', data_file), (n_train, n_val, n_test), n, n_his + n_pred)

    train_dataset = Dataset(np.transpose(X['train'], (0, 2, 1, 3)), n_his)
    val_dataset = Dataset(np.transpose(X['val'], (0, 2, 1, 3)), n_his)
    test_dataset = Dataset(np.transpose(X['test'], (0, 2, 1, 3)), n_his)

    training_generator = torch.utils.data.DataLoader(train_dataset, **params)
    validation_generator = torch.utils.data.DataLoader(val_dataset, **params)


    net = STGCN(A_wave.shape[0],
                train_dataset.shape()[3],
                n_his,
                n_pred).to(device=torch.device('cuda'), dtype=torch.float)

    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    loss_criterion = nn.MSELoss()

    training_losses = []
    validation_losses = []
    validation_maes = []

    # Loop over epochs
    for epoch in range(epochs):
        loss = train_epoch(training_generator,
                           batch_size=batch_size)
        training_losses.append(loss)

        # Run validation
        with torch.set_grad_enabled(False):
            local_val = []
            for local_batch, local_labels in validation_generator:
                net.eval()
#                 local_batch, local_labels = local_batch.to(device), local_labels.to(device)
                local_batch, local_labels = local_batch.to(device, dtype=torch.float), local_labels.to(device, dtype=torch.float)
                
                out = net(A_wave, local_batch)
                val_loss = loss_criterion(out, local_labels).to(device="cpu")
                local_val.append(np.asscalar(val_loss.detach().numpy()))
            
            validation_losses.append(np.mean(local_val))
            std_val, mean_val = val_dataset.get_stats()['std'], val_dataset.get_stats()['mean']
            out_unnormalized = out.detach().cpu().numpy() #* std_val + mean_val #*stds[0]+means[0]
            target_unnormalized = local_labels.detach().cpu().numpy() #* std_val + mean_val #*stds[0]+means[0]
            mae = np.mean(np.absolute(out_unnormalized - target_unnormalized))
            validation_maes.append(mae)

            out = None
            local_batch = local_batch.to(device="cpu")
            local_labels = local_labels.to(device="cpu")

    print("Training loss: {}".format(training_losses[-1]))
    print("Validation loss: {}".format(validation_losses[-1]))
    print("Validation MAE: {}".format(validation_maes[-1]))
    plt.plot(training_losses, label="training loss")
    plt.plot(validation_losses, label="validation loss")
    plt.legend()
    plt.show()