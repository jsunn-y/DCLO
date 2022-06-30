import numpy as np
from src.encoding_utils import *


def get_init_samples(n_samples, n_sites):
    """
    generates an initial sample of degenerate codon libraries
    """
    #write something to prevent it from sampling all zeros?
    initial = np.zeros((n_samples, n_sites*12))
    for i in range(n_samples):
        for j in range(n_sites*3):
            choice = (0, 0, 0, 0)

            #alternatively make sure the sum is greater than a certain amount
            while sum(choice) == 0:
                choice = np.random.choice(2, 4)
            initial[i, 4*j:4*(j+1)] = choice
    return initial

def get_samples(Xt_p):
    """Samples from a categorical probability distribution specifying the probability of a nucleotide being allowed at each position in a sequence"""
    Xt_sampled = np.zeros((Xt_p.shape[0], Xt_p.shape[1]))
    for i in range(Xt_p.shape[0]):
        for j in range(int(Xt_p.shape[1]/4)):
            probs = Xt_p[i, 4*j:4*(j+1)]
            samples = np.random.uniform(size = 4)
            boolean =  probs < samples
            choice = boolean*1
            if sum(choice == 0):
                # this is getting triggered at every site
                choice[np.argmax(probs)] = 1
            Xt_sampled[i, 4*j:4*(j+1)] = choice
    return Xt_sampled
    
    
    #old code
    # Xt_sampled = np.zeros_like(Xt_p)
    # for i in range(Xt_p.shape[0]):
    #     for j in range(Xt_p.shape[1]):
    #         p = Xt_p[i, j]
    #         #either 0 or 1
    #         k = np.random.choice(np.arange(2), p=[p, 1-p])
    #         Xt_sampled[i, j] = k
    
    #write something to prevent it from sampling 4 zeros?

    #return Xt_sampled


