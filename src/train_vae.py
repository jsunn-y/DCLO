import os
import random
import time
import numpy as np
from itertools import cycle
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm.auto import tqdm
from src.vae_model import VAE
from src.encoding_utils import *

def train(weights: np.array, model: nn.Module, device: torch.device, data_loader: DataLoader, pbar: tqdm) -> float:
    """Trains a VAE model.

    Args
    - model: nn.Module, VAE model, already placed on device
    - device: torch.device
    - data_loader: pyg.loader.DataLoader
    - optimizer: torch.optim.Optimizer
    - loss_fn: nn.Module

    Returns: loss
    - loss: float, avg loss across epoch
    """
    model.train()
    cum_losses = model.losses
    for key in cum_losses: cum_losses[key] = 0

    for step, (x, weights) in enumerate(data_loader):
        x = x.to(device)
        weights = weights.to(device)

        #forward pass
        model(x, weights)
        #backaward pass
        model.optimize()
        for key in model.losses: cum_losses[key] += model.losses[key]

    return cum_losses["reconst_loss"], cum_losses["kl_div"]

def evaluate(Xt: torch.tensor, model: nn.Module, device: torch.device):
    """
    Evaluates the quality of the reconstructions of the VAE. Currently does not support GPU or batching since that would not really accelerate things.
    """
    model.eval()
    with torch.no_grad():
        Xt = Xt.to(device)
        if model.variational:
            mu, var = model.encode(Xt)
        else:
            mu = model.encode(Xt)
        reconstructions = 1 * (torch.sigmoid(model.decode(mu)) > 0.5)
    
    original = []
    new = []
    differences = []
    for row1, row2 in zip(Xt.detach().cpu().numpy(), reconstructions.detach().cpu().numpy()):
        seq1 = encoding2seq(row1)
        seq2 = encoding2seq(row2)
        original.append(seq1)
        new.append(seq2)
        differences.append(hammingdistance(seq1, seq2))
    
    return differences

def start_training(Xt, save_path, data_config, vae_model_config, vae_train_config, device, weights):

    # Sample and fix a random seed if not set in train_config
    if 'seed' not in vae_train_config:
        vae_train_config['seed'] = random.randint(0, 9999)
    seed = vae_train_config['seed']
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

    # Initialize dataset
    dataset = TensorDataset(torch.tensor(Xt).float(), torch.tensor(weights).float())

    model = VAE(vae_model_config, data_config).to(device)
    
    #Initialize dataloaders
    train_loader = DataLoader(dataset, batch_size=vae_train_config['batch_size'],  shuffle=True)

    # Initialize optimizer
    model.init_optimizer(vae_train_config)

    # Start training
    pbar = tqdm()
    pbar.reset(vae_train_config['num_epochs'])
    pbar.set_description('Training VAE')

    for epoch in range(1, 1 + vae_train_config['num_epochs']):
        recon_loss, kl_div = train(weights, model, device, train_loader, pbar)
        loss = recon_loss + kl_div

        #update the best model after each epoch
        # if epoch == 1 or loss < best_loss:
        #     best_loss = loss
        #     torch.save(model.state_dict(), save_path + '/best.pth')
        #     print('Best model saved') 
        
        pbar.update()
        if epoch == 1 or epoch == vae_train_config['num_epochs']:
            n_samples = len(train_loader)
            #should kl div be scaled by the number of samples?
            tqdm.write(f'Epoch {epoch:02d}, recon_loss: {recon_loss/n_samples:.4f}, kl_div: {kl_div:.4f}')
    
    return model