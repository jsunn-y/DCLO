from multiprocessing import Pool
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Bio.Seq import Seq
from tqdm.auto import tqdm
from src.encoding_utils import *

class Oracle():
    """Maps from a degenerate mixed base library to the zs score distribtuion. Maintains mappings that have already been calculated."""
    def __init__(self, data_config):
        self.mappings = {}
        self.n_samples = data_config["samples"]
        self.diversity_thresh = data_config["diversity_thresh"]
        self.num_workers = 64

        df = pd.read_csv("/home/jyang4/repos/DCLO/data/GB1_all_triad.csv")
        
        ranked_zs = df["Score"].rank(ascending=False)
        self.avg_zs = np.mean(ranked_zs)
        #missing combos are now filled in 
        self.combo2zs_dict = dict(zip(df["Combo"].values, ranked_zs))

    def __encoding2nucleotides(self, encoding):
        """converts a numerical encoding of a mixed base library into its possible nucleotide sequences (sampled or fully iterated)"""
        encoding = encoding.reshape((-1, 4))
        #n_sites = encoding.shape[0]
        #seq = encoding2seq(encoding)
        
        #insert a seed here?
        
        #if seq in self.mappings.items():
        #    return self.mappings[seq]
        if False:
            pass
        else: 
            ###sample randomly from what is allowed###
            bseqs = []
            for i in range(self.n_samples):
                bseq = ""
                for row in encoding:
                    # indices = np.where(row == 1)
                    # choices = [index2base_dict[x] for x in indices[0]]
                    #print(tuple(row))
                    choices = encoding2choices_dict[tuple(row)]
                    bseq += np.random.choice(choices)
                bseqs.append(bseq)
        return bseqs
        
        ### Use the code below if you want to iterate through all possibilities ###
        
        # #need to append all the sequences
        # self.seqs = []

        # self.seq = ""
        # self.for_recursive(options, execute_function = self.make_seq, number_of_loops = n_sites)
        # self.seqs.append(self.seq)

        #return self.seqs
    
    def __nucleotides2zs(self, bseqs):
        #map nucleotides to their corresponding protein sequences
        #map all protein sequences to their corresponding zero shot scores
        #report the distribution
        aaseqs = [Seq(seq).translate() for seq in bseqs]
        zs_scores = []
        for seq in aaseqs:
            if seq in self.combo2zs_dict.keys():
                zs_scores.append(self.combo2zs_dict[seq])
            #assign the minimum for a stop codon
            elif '*' in seq:
                zs_scores.append(0)
            else:
                print('Missing ZS score')
                zs_scores.append(self.avg_zs)
        
        #proxy to increase diversity by considering noise as inverse variance
        #return np.mean(zs_scores), 1/(np.var(zs_scores) + 1)
        #print(zs_scores)
        #print(np.mean(zs_scores))

        diversity = len(np.unique(zs_scores))
        #enforce a certain level of diversity 
        if diversity < self.diversity_thresh:
            return 0, 0
        else:
            return np.mean(zs_scores), np.var(zs_scores)
    
    def predict(self, encodings): 
        results = np.zeros((encodings.shape[0], 2))
        pbar = tqdm()
        pbar.reset(len(encodings))
        pbar.set_description('Predicting')

        for i, row in enumerate(encodings):
            
            results[i, :] = self.__nucleotides2zs(self.__encoding2nucleotides(row))
            #if encoding2seq(row) == 'WSHYYHMSWNYS':
            #    exit()
            pbar.update()
        
        #add to self.mappings (doesn't make sense if you just)
        return results[:,0], results[:,1]

    
    with Pool(self.num_workers) as p:
        bseqs = p.map(self.sampler, [encoding]*self.n_samples)


    # def make_seq(nucleotide):
    #     self.seq += nucleotide

    # #more efficient way to do this using itertools?
    # def for_recursive(number_of_loops, range_list, execute_function, current_index=0, iter_list = []):

    #     if iter_list == []:
    #         iter_list = [0]*number_of_loops

    #     if current_index == number_of_loops-1:
    #         for iter_list[current_index] in range_list[current_index]:
    #             execute_function(iter_list)
    #     else:
    #         for iter_list[current_index] in range_list[current_index]:
    #             for_recursive(number_of_loops, iter_list = iter_list, range_list = range_list)