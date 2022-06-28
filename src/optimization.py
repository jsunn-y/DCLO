import numpy as np
from scipy.stats import norm
import torch
from src.vae_model import VAE
from src.oracle import Oracle
from src.util import get_init_samples, get_samples, encoding2seq
from src.train_vae import start_training


class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

def run_dbas(save_path, data_config, vae_model_config, vae_train_config, opt_config, device, verbose=True, homoscedastic=False):
    
    """
    Runs weighted maximum likelihood optimization algorithms ('DbAS')
    """
    
    assert opt_config['type'] in ['dbas']
    iters = opt_config["iters"]
    sites = data_config["sites"]
    samples = opt_config["samples"]

    traj = np.zeros((iters, 7))
    oracle_samples = np.zeros((iters, samples))
    gt_samples = np.zeros((iters, samples))
    oracle_max_seq = None
    oracle_max = -np.inf
    gt_of_oracle_max = -np.inf
    y_star = -np.inf  

    np.random.seed(opt_config['seed'])
    oracle = Oracle(data_config)

    for t in range(iters):
        print('####  Iteration: ' + str(t+1) + '  ####')
        ### Take samples ###
        zt = np.random.randn(samples, vae_model_config['z_dim'])
        if t > 0:
            Xt_p = vae.decode(torch.tensor(zt).float())
            Xt = get_samples(Xt_p.detach().numpy())
        else:
            Xt = get_init_samples(samples, sites)
        
        ### Evaluate ground truth and oracle ###
        #is the variance of the zs score distribution a form of uncertainty?
        
        yt, yt_var = oracle.predict(Xt)
        
        ### Calculate weights for different schemes ###
        if t > 0:
            #weights for dbas
            y_star_1 = np.percentile(yt, opt_config['quantile']*100)
            if y_star_1 > y_star:
                y_star = y_star_1
            weights = norm.sf(y_star, loc=yt, scale=np.sqrt(yt_var))
        else:
            weights = np.ones(yt.shape[0])
            
        yt_max_idx = np.argmax(yt)
        yt_max = yt[yt_max_idx]

        if yt_max > oracle_max:
            oracle_max = yt_max
            try:
                oracle_max_seq = encoding2seq(Xt[yt_max_idx-1:yt_max_idx])[0]
            except IndexError:
                print(Xt[yt_max_idx-1:yt_max_idx])
        
        ### Record and print results ##
        #is this subsampling or sampling the same amount?
        if t == 0:
            rand_idx = np.random.randint(0, len(yt), samples)
            oracle_samples[t, :] = yt[rand_idx]
        if t > 0:
            oracle_samples[t, :] = yt
        
        traj[t, 3] = np.max(yt)
        traj[t, 4] = np.mean(yt)
        traj[t, 5] = np.std(yt)
        traj[t, 6] = np.mean(yt_var)
        
        if verbose:
            print(t, traj[t, 3], color.BOLD + str(traj[t, 4]) + color.END, traj[t, 5], traj[t, 6])
        
        ### Train model ###
        #changed code so that training starts in the first round
        # if t == 0:
        #     #set weights here
        #     pass
        #     # vae.encoder_.set_weights(vae_0.encoder_.get_weights())
        #     # vae.decoder_.set_weights(vae_0.decoder_.get_weights())
        #     # vae.vae_.set_weights(vae_0.vae_.get_weights())
        # else:
            
        cutoff_idx = np.where(weights < opt_config['cutoff'])
        Xt = np.delete(Xt, cutoff_idx, axis=0)
        yt = np.delete(yt, cutoff_idx, axis=0)
        weights = np.delete(weights, cutoff_idx, axis=0)

        vae = start_training(Xt, save_path, data_config, vae_model_config, vae_train_config, device, weights)
            # vae.fit([Xt], [Xt, np.zeros(Xt.shape[0])],
            #       epochs=it_epochs,
            #       batch_size=10,
            #       shuffle=False,
            #       sample_weight=[weights, weights],
            #       verbose=0)
    
    max_dict = {'oracle_max' : oracle_max, 
                'oracle_max_seq': oracle_max_seq}
    
    return traj, oracle_samples, max_dict
