def weighted_ml_opt(X_train, oracles, ground_truth, vae_0, weights_type='dbas',
                    LD=20, iters=20, samples=500, homoscedastic=False, homo_y_var=0.1,
                    quantile=0.95, verbose=False, alpha=1, train_gt_evals=None,
                    cutoff=1e-6, it_epochs=10, enc1_units=50):
    
    """
    Runs weighted maximum likelihood optimization algorithms ('CbAS', 'DbAS',
    RWR, and CEM-PI)
    """
    
    assert weights_type in ['cbas', 'dbas','rwr', 'cem-pi']
    L = X_train.shape[1]
    vae = util.build_vae(latent_dim=LD,
                    n_tokens=20, seq_length=L,
                    enc1_units=enc1_units)

    traj = np.zeros((iters, 7))
    oracle_samples = np.zeros((iters, samples))
    gt_samples = np.zeros((iters, samples))
    oracle_max_seq = None
    oracle_max = -np.inf
    gt_of_oracle_max = -np.inf
    y_star = -np.inf  
    
    for t in range(iters):
        ### Take Samples ###
        zt = np.random.randn(samples, LD)
        if t > 0:
            Xt_p = vae.decoder_.predict(zt)
            Xt = util.get_samples(Xt_p)
        else:
            Xt = X_train
        
        ### Evaluate ground truth and oracle ###
        yt, yt_var = util.get_balaji_predictions(oracles, Xt)
        if homoscedastic:
            yt_var = np.ones_like(yt) * homo_y_var
        Xt_aa = np.argmax(Xt, axis=-1)
        if t == 0 and train_gt_evals is not None:
            yt_gt = train_gt_evals
        else:
            yt_gt = ground_truth.predict(Xt_aa, print_every=1000000)[:, 0]
        
        ### Calculate weights for different schemes ###
        if t > 0:
            if weights_type == 'cbas': 
                log_pxt = np.sum(np.log(Xt_p) * Xt, axis=(1, 2))
                X0_p = vae_0.decoder_.predict(zt)
                log_px0 = np.sum(np.log(X0_p) * Xt, axis=(1, 2))
                w1 = np.exp(log_px0-log_pxt)
                y_star_1 = np.percentile(yt, quantile*100)
                if y_star_1 > y_star:
                    y_star = y_star_1
                w2= scipy.stats.norm.sf(y_star, loc=yt, scale=np.sqrt(yt_var))
                weights = w1*w2 
            elif weights_type == 'cem-pi':
                pi = scipy.stats.norm.sf(max_train_gt, loc=yt, scale=np.sqrt(yt_var))
                pi_thresh = np.percentile(pi, quantile*100)
                weights = (pi > pi_thresh).astype(int)
            elif weights_type == 'dbas':
                y_star_1 = np.percentile(yt, quantile*100)
                if y_star_1 > y_star:
                    y_star = y_star_1
                weights = scipy.stats.norm.sf(y_star, loc=yt, scale=np.sqrt(yt_var))
            elif weights_type == 'rwr':
                weights = np.exp(alpha*yt)
                weights /= np.sum(weights)
        else:
            weights = np.ones(yt.shape[0])
            max_train_gt = np.max(yt_gt)
            
        yt_max_idx = np.argmax(yt)
        yt_max = yt[yt_max_idx]
        if yt_max > oracle_max:
            oracle_max = yt_max
            try:
                oracle_max_seq = util.convert_idx_array_to_aas(Xt_aa[yt_max_idx-1:yt_max_idx])[0]
            except IndexError:
                print(Xt_aa[yt_max_idx-1:yt_max_idx])
            gt_of_oracle_max = yt_gt[yt_max_idx]
        
        ### Record and print results ##
        if t == 0:
            rand_idx = np.random.randint(0, len(yt), samples)
            oracle_samples[t, :] = yt[rand_idx]
            gt_samples[t, :] = yt_gt[rand_idx]
        if t > 0:
            oracle_samples[t, :] = yt
            gt_samples[t, :] = yt_gt
        
        traj[t, 0] = np.max(yt_gt)
        traj[t, 1] = np.mean(yt_gt)
        traj[t, 2] = np.std(yt_gt)
        traj[t, 3] = np.max(yt)
        traj[t, 4] = np.mean(yt)
        traj[t, 5] = np.std(yt)
        traj[t, 6] = np.mean(yt_var)
        
        if verbose:
            print(weights_type.upper(), t, traj[t, 0], color.BOLD + str(traj[t, 1]) + color.END, 
                  traj[t, 2], traj[t, 3], color.BOLD + str(traj[t, 4]) + color.END, traj[t, 5], traj[t, 6])
        
        ### Train model ###
        if t == 0:
            vae.encoder_.set_weights(vae_0.encoder_.get_weights())
            vae.decoder_.set_weights(vae_0.decoder_.get_weights())
            vae.vae_.set_weights(vae_0.vae_.get_weights())
        else:
            cutoff_idx = np.where(weights < cutoff)
            Xt = np.delete(Xt, cutoff_idx, axis=0)
            yt = np.delete(yt, cutoff_idx, axis=0)
            weights = np.delete(weights, cutoff_idx, axis=0)
            vae.fit([Xt], [Xt, np.zeros(Xt.shape[0])],
                  epochs=it_epochs,
                  batch_size=10,
                  shuffle=False,
                  sample_weight=[weights, weights],
                  verbose=0)
    
    max_dict = {'oracle_max' : oracle_max, 
                'oracle_max_seq': oracle_max_seq, 
                'gt_of_oracle_max': gt_of_oracle_max}
    return traj, oracle_samples, gt_samples, max_dict