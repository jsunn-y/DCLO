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
    uncertainty = opt_config["uncertainty"]

    traj = np.zeros((iters, 10))
    oracle_samples = np.zeros((iters, samples))
    gt_samples = np.zeros((iters, samples))
    oracle_max_seq = None
    oracle_max = -np.inf
    gt_of_oracle_max = -np.inf
    y_star = -np.inf  

    np.random.seed(opt_config['seed'])
    oracle = Oracle(data_config, opt_config)

    for t in range(iters):
        print('\n####  Iteration: ' + str(t+1) + '  ####')
        ### Take random normal samples ###
        zt = np.random.randn(samples, vae_model_config['z_dim'])
        if t > 0:
            Xt_p = vae.decode(torch.tensor(zt).float())
            Xt_new = get_samples(Xt_p.detach().numpy())
            Xt = Xt_new

            #can train the VAE with all the samples or just the new ones
            #print(Xt.shape)
            # print(Xt_new.shape)
            # Xt = np.concatenate((Xt_new, Xt), axis = 0)
            # print(Xt.shape)
        else:
            #np.random.seed(opt_config['seed'] + t)
            Xt = get_init_samples(samples, sites)
        
        ### Evaluate ground truth and oracle ###

        means, vars = oracle.predict(Xt)

        yt = means[:, 0]
        counts = means[:, 1]
        div = means[:, 2]

        yt_var = vars[:, 0]
        counts_var = vars[:, 1]
        div_var = vars[:, 2]
        
        ### Calculate weights for different schemes ###
        if t > 0:
            #weights for dbas
            #finds the y value that represents the desired percentile
            y_star_1 = np.percentile(yt, opt_config['quantile']*100)
            
            if y_star_1 > y_star:
                y_star = y_star_1

            print('Quantile Cutoff: %6.0f' % y_star)
            #uses the survival function (1 - cdf), shouldn't it be the opposite?
            #find what fraction of samples lie above y_star if the zs_distribution was modelled as a standard normal
            ###in the original paper, highly uncertain weights are penalized###
            #instead we penalize  the opposite (low-variance ZS score distributions), but really we should penalize diversity

            if uncertainty == True:
                weights = norm.sf(y_star, loc=yt, scale=np.sqrt(yt_var))
            else:
            ###ignore the uncertainty of the weights###
                weights = norm.sf(y_star, loc=yt)
        else:
            weights = np.ones(yt.shape[0])

        print('Sum of Weights: %3.0f' % np.sum(np.sum(weights)))

        yt_max_idx = np.argmax(yt)
        yt_max = yt[yt_max_idx]
        
        if yt_max > oracle_max:
            oracle_max = yt_max
            div_max = div[yt_max_idx]
            counts_max = counts[yt_max_idx]
            try:
                oracle_max_seq = encoding2seq(Xt[yt_max_idx])
            except IndexError:
                print(Xt[yt_max_idx])
        
        #is this subsampling or just reordering initially?
        if t == 0:
            rand_idx = np.random.randint(0, len(yt), samples)
            oracle_samples[t, :] = yt[rand_idx]
        #is it even used later on?
        if t > 0:
            oracle_samples[t, :] = yt[-samples:]
        
        #Keep track of training statistics
        traj[t, 0] = np.max(yt)
        traj[t, 1] = np.mean(yt)
        
        #for now just use the counts for the other max
        traj[t, 2] = counts[yt_max_idx]
        traj[t, 3] = np.mean(counts)

        traj[t, 4] = div[yt_max_idx]
        traj[t, 5] = np.mean(div)
        traj[t, 6] = np.std(yt)

        traj[t, 7] = np.mean(np.sqrt(yt_var))
        traj[t, 8] = np.mean(np.sqrt(counts_var))
        traj[t, 9] = np.mean(np.sqrt(div_var))
        
        ### Record and print results ##
        if verbose:
            print("Mean Score: %6.0f,  Mean Counts: %4.0f, Mean Diversity: %4.0f" % (traj[t, 1], traj[t, 3], traj[t, 5]))
            print("Std Score: %5.0f, Std Counts: %5.0f, Std Diversity: %5.0f" % (traj[t, 7], traj[t, 8], traj[t, 9]))
            print("Best Sequence of this Iteration: %s (Score: %6.0f, Counts: %4.0f, Diversity: %4.0f)" % (encoding2seq(Xt[yt_max_idx]), traj[t, 0], traj[t, 2], traj[t, 4]))

            print("Running Best: %s (Score: %6.0f, Counts: %4.0f,  Diversity: %4.0f)" % (oracle_max_seq,oracle_max, counts_max, div_max))
            
            # print(t, traj[t, 3], color.BOLD + str(traj[t, 4]) + color.END, traj[t, 5], traj[t, 6])
        
        ### Train model ###
        #changed code so that training starts in the first round
        # if t == 0:
        #     #set weights here
        #     pass
        #     # vae.encoder_.set_weights(vae_0.encoder_.get_weights())
        #     # vae.decoder_.set_weights(vae_0.decoder_.get_weights())
        #     # vae.vae_.set_weights(vae_0.vae_.get_weights())
        # else:
        
        #do not need to consider samples with a weight of zero (below the cutoff)
        cutoff_idx = np.where(weights < opt_config['cutoff'])
        Xt = np.delete(Xt, cutoff_idx, axis=0)
        yt = np.delete(yt, cutoff_idx, axis=0)
        weights = np.delete(weights, cutoff_idx, axis=0)

        #reset the weights?
        vae = start_training(Xt, save_path, data_config, vae_model_config, vae_train_config, device, weights)
    
    max_dict = {'oracle_max' : oracle_max, 
                'oracle_max_seq': oracle_max_seq,
                'diversity_max': div_max}
    
    return traj, oracle_samples, max_dict
