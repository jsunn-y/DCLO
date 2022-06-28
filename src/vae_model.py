import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, model_config, data_config):
        #not sure what this line does tbh
        super(VAE, self).__init__()
        
        dropout = model_config['dropout']
        enc_dim1 = model_config['enc_dim1']
        enc_dim2 = model_config['enc_dim2']
        z_dim = model_config['z_dim']
        dec_dim1 = model_config['dec_dim1']
        dec_dim2 = model_config['dec_dim2']

        input_dim = data_config['sites']*12
        self.variational = model_config['kl_div_weight'] != 0

        self.losses = {}
        self.losses['reconst_loss'] = 0
        self.losses['kl_div'] = 0

        self.reconst_loss_weight = model_config["reconstruction_loss_weight"]
        self.kl_div_weight = model_config["kl_div_weight"]

        self.dropout = nn.Dropout(dropout)
        #encoder layers
        self.fce1 = nn.Linear(input_dim, enc_dim1)
        self.fce2 = nn.Linear(enc_dim1, enc_dim2)
        self.bne = nn.BatchNorm1d(enc_dim2)
        self.fce3 = nn.Linear(enc_dim2, z_dim)
        self.fcvar = nn.Linear(enc_dim2, z_dim)
        
        #decoder layers
        self.fcd1 = nn.Linear(z_dim, dec_dim1)
        self.fcd2 = nn.Linear(dec_dim1, dec_dim2)
        self.fcd3 = nn.Linear(dec_dim2, input_dim)
        
    def encode(self, x):
        h1 = F.relu(self.fce1(x))
        h1 = self.dropout(h1)
        h2 = F.relu(self.fce2(h1))
        h2 = self.bne(h2)
        
        if self.variational:
            return self.fce3(h2), self.fcvar(h2)
        else:
            return self.fce3(h2)
    
    def reparameterize(self, mu, log_var):
        log_var = torch.clamp(log_var, -16, 16)
        std = torch.exp(log_var/2)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h1 = F.relu(self.fcd1(z))
        h1 = self.dropout(h1)
        h2 = F.relu(self.fcd2(h1))
        return self.fcd3(h2)
    
    def init_optimizer(self, train_config):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=train_config['learning_rate'])

    def get_total_loss(self, losses):
        return losses['reconst_loss']*self.reconst_loss_weight + losses['kl_div']*self.kl_div_weight

    def optimize(self):
        self.optimizer.zero_grad()
        self.total_loss = self.get_total_loss(self.losses)
        self.total_loss.backward()
        self.optimizer.step()
    
    def forward(self, x, weights):
        if self.variational:
            mu, log_var = self.encode(x)
            z = self.reparameterize(mu, log_var)
            x_reconst = self.decode(z)
            self.losses['kl_div'] = self.compute_kl_div(mu, log_var)
        else:
            z = self.encode(x)
            x_reconst = self.decode(z)

        self.losses['reconst_loss'] = self.compute_reconst_loss(x, x_reconst, weights)

    @staticmethod
    def compute_reconst_loss(x, x_reconst, weights):
        reconst_loss = F.binary_cross_entropy(torch.sigmoid(x_reconst), x, reduction = 'none') #weights in the function is for class weights
        reconst_loss = torch.sum(reconst_loss, axis = 1)
        reconst_loss = reconst_loss * weights
        return torch.sum(reconst_loss)

    @staticmethod
    def compute_kl_div(mu, log_var):
        return - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())