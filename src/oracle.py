from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool
from concurrent.futures import ProcessPoolExecutor
from unittest import result
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Bio.Seq import Seq
from tqdm.auto import tqdm
from src.encoding_utils import *

class Oracle():
    """Maps from a degenerate mixed base library to the zs score distribtuion. Maintains mappings that have already been calculated."""
    def __init__(self, data_config, opt_config):
        self.mappings = {}
        self.n_samples = data_config["samples"]
        self.sites = data_config["sites"]
        self.diversity_thresh = opt_config["diversity_thresh"]
        self.repeats = opt_config["num_repeats"]
        self.num_workers = opt_config["num_workers"]
        self.cutoff = 20**data_config["sites"]*(1- opt_config['top_fraction'])

        df = pd.read_csv("/home/jyang4/repos/DCLO/data/GB1_all_triad.csv")
        
        ranked_zs = df["Score"].rank(ascending=False)
        self.avg_zs = np.mean(ranked_zs)
        
        #missing combos are now filled in 
        self.combo2zs_dict = dict(zip(df["Combo"].values, ranked_zs))

    def __encoding2nucleotides(self, encoding):
        """converts a numerical encoding of a mixed base library into its possible nucleotide sequences (sampled or fully iterated)"""
        encoding = encoding.reshape((-1, self.sites))
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
        
        ### Use the code below if you want to iterate through all possibilities ### does not work yet
        
        # #need to append all the sequences
        # self.seqs = []

        # self.seq = ""
        # self.for_recursive(options, execute_function = self.make_seq, number_of_loops = n_sites)
        # self.seqs.append(self.seq)

        #return self.seqs
    
    def __nucleotides2zs(self, bseqs):
        """
        map nucleotides to their corresponding protein sequences
        map all protein sequences to their corresponding zero shot scores
        report stats about the distribution
        """
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
        zs_scores = np.array(zs_scores)
        uniques = np.unique(zs_scores)

        #treat it like all repeats are zero because they provide no value
        zs_score_avg = np.sum(uniques)/self.n_samples
        diversity = len(uniques)
        counts = len(np.unique(zs_scores[zs_scores > self.cutoff]))
        #enforce a certain level of diversity 
        #no longer need to calculate the variance of the nucleotides, only the variance of the sampling

        #if diversity < self.diversity_thresh:
        #    return 0, counts, diversity
        return zs_score_avg, counts, diversity
    
    def predict(self, encodings): 
        self.encodings = encodings
        self.batch_size = encodings.shape[0]
        
        #run the repeated encoding calculations in parallel 
        results = np.zeros((self.batch_size, 3, self.repeats))
        with Pool(self.num_workers) as p:
            # with tqdm() as pbar:
            #     pbar = tqdm()
            #     pbar.reset(self.batch_size*self.repeats)
            #     pbar.set_description('Predicting')
            seeds = range(self.repeats)
            pbar = tqdm(p.imap_unordered(self.predictor_all, seeds),
            total=self.repeats)
            pbar.set_description('Predicting through Oracle')

            #all_results = p.map(self.predictor_all, [encodings]*self.repeats)

            for i, result in enumerate(pbar):
                 results[:,:,i] = result
        
        means = np.mean(results, axis = 2)
        vars =  np.var(results, axis = 2)

        #all_results = np.zeros((self.batch_size, 2, self.repeats))

        ### Other option is to run each row in parallel ###
        #probably less efficient because each row is so quick
        # with Pool(self.num_workers) as p:
        #      all_results = p.map(self.predictor, [row for row in encodings])
        
        # all_results = np.array(all_results)

        return means, vars

    def predictor(self, encoding):
        return self.__nucleotides2zs(self.__encoding2nucleotides(encoding))

    def predictor_all(self, myseed):
        np.random.seed(myseed)
        results = np.zeros((self.batch_size, 3))
        for i, row in enumerate(self.encodings):
            results[i, :] = self.__nucleotides2zs(self.__encoding2nucleotides(row))
        return results
        
            #if encoding2seq(row) == 'WSHYYHMSWNYS':
            #    exit()
        
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