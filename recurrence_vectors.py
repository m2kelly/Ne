import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from . import essentials as es
import warnings
from copy import deepcopy
warnings.filterwarnings('ignore')
from .base_operations import Operations
from .markov_infsites import MarkovInfSites
from collections import defaultdict

class ReccurenceVectors(Operations):

    def __init__(self, name, directory, indices, non_cpgs, cpgs=None, collapse=True,
                 muts_dict_raw=None, occ_dict_raw=None, cpg_remove_percentage=0,
                 generations=None, prefix='', operations=None):

        Operations.__init__(self, name=name, directory=directory, collapse=collapse,
                            muts_dict_raw=muts_dict_raw, occ_dict_raw=occ_dict_raw,
                            cpg_remove_percentage=cpg_remove_percentage, prefix=prefix,
                            operations=operations)

        self.generations = generations
        self.indices = indices

        self.mutations_pool = self.occ_dict_raw['chr1'].index

        if cpgs is None:
            all_cpgs = es.mutation.get_cpg_muts()
            self.cpg_muts = [x for x in all_cpgs if str(x) in self.occ_dict_raw['chr1'].index]
        else: self.cpg_muts = [c for c in cpgs if str(c) in self.occ_dict_raw['chr1'].index]

        self.non_cpg_muts = non_cpgs
        self.remove_percentage = 0.00

        self.cpg_vector_original = None
        self.chosen_non_cpg_vector_original = None
        self.cpg_backward_vector_original = None
        self.chosen_non_cpg_backward_vector_original = None
        

        self.cpg_vector_mrkv_corrected = None
        self.chosen_non_cpg_vector_mrkv_corrected = None
        self.cpg_backward_vector_mrkv_corrected = None
        self.chosen_non_cpg_backward_vector_mrkv_corrected = None

        self.best_static = None
        self.best_bins = None
        self.CpG_remove_percentage = None
        self.no_recurrence = None

    

    @classmethod
    def get_mutations_from_labels(cls, labels, pool=None):
        mutations = [es.mutation(label=m) for m in labels]
        if pool is not None:
            mutations = [m for m in mutations if str(m) in pool]
        return mutations

    def export_arrays_before_after_mrkov(self, cpg_box, non_cpg_box,_b, _c, c, nc, cb, ncb, name):
        before_cpg = [np.median(cpg_box[i])/self.generations for i in range(len(cpg_box))]
        before_non_cpg = [np.median(non_cpg_box[i])/self.generations for i in range(len(non_cpg_box))]
        before_cpg_backward = [np.median(_b[i])/self.generations for i in range(len(_b))]
        before_chosen_non_cpg_backward = [np.median(_c[i])/self.generations for i in range(len(_c))]
        self.write_logs(f'for {name}:')
        self.write_logs(f'\tuncorrected cpg: {before_cpg}\n\tuncorrected non_cpg: {before_non_cpg}\n\tuncorrected cpg_backward: {before_cpg_backward}\n\tuncorrected chosen_non_cpg_backward: {before_chosen_non_cpg_backward}\n')
        self.cpg_vector_mrkv_corrected = c
        self.chosen_non_cpg_vector_mrkv_corrected = nc
        self.cpg_backward_vector_mrkv_corrected = cb
        self.chosen_non_cpg_backward_vector_mrkv_corrected = ncb
        percent_change_cpg = [(c[i] - before_cpg[i])/before_cpg[i] for i in range(len(c))]
        percent_change_non_cpg = [(nc[i] - before_non_cpg[i])/before_non_cpg[i] for i in range(len(nc))]
        percent_change_cpg_backward = [(cb[i] - before_cpg_backward[i])/before_cpg_backward[i] for i in range(len(cb))]
        percent_change_chosen_non_cpg_backward = [(ncb[i] - before_chosen_non_cpg_backward[i])/before_chosen_non_cpg_backward[i] for i in range(len(ncb))]

        self.write_logs(f'\tcorrected cpg: {c}\n\tcorrected non_cpg: {nc}\n\tcorrected cpg_backward: {cb}\n\tcorrected chosen_non_cpg_backward: {ncb}\n')
        self.write_logs(f'\tpercent change cpg: {percent_change_cpg}\n\tpercent change non_cpg: {percent_change_non_cpg}\
                        \n\tpercent change cpg_backward: {percent_change_cpg_backward}\n\tpercent change chosen_non_cpg_backward: {percent_change_chosen_non_cpg_backward}\n')

    def get_matrices(self):
        '''a lits of the occs and muts matrices from the smoothed dictionaries based
           on the indeces, editted so uses unsmoothed dicts
        '''
        #muts = deepcopy(self.muts_dict_smoothed); occs = deepcopy(self.occ_dict_smoothed)
        muts=es.rename_cols(self.muts_dict_raw)
        occs=es.rename_cols(self.occ_dict_raw)
        
        muts_matrices = []; occs_matrices = []
        for index in self.indices:
            index = list(index)
            inner_muts = muts.T.loc[index].sum()
            inner_occs = occs.T.loc[index].sum()

            muts_matrices.append(inner_muts)
            occs_matrices.append(inner_occs)

        return muts_matrices, occs_matrices

    def get_markov_input_matrices(self, muts_matrices, occs_matrices):
        '''returns an initial state, final state and a transition matrix for each matrix
        '''
        groups = []
        for muts, occs in zip(muts_matrices, occs_matrices):
            initial_state = {}
            for mut in es.get_muts_sig_ordered():
                if str(mut) not in occs.index: continue
                if mut.tri not in initial_state:
                    initial_state[mut.tri] =  occs[str(mut)]
                else:
                    initial_state[mut.tri] = max(initial_state[mut.tri], occs[str(mut)])

            initial_state = pd.Series(initial_state)

            final_state = pd.DataFrame(0.0,index=initial_state.index, columns=initial_state.index)
            for mut in es.get_muts_sig_ordered():
                if str(mut) not in muts.index: continue
                target = mut.get_backwards().tri
                if target not in final_state: target = es.get_rev_comp(target)
                if target not in final_state: raise ValueError('target not found')
                final_state.loc[mut.tri, target] += muts[str(mut)]
            #diagonal is 1 - sum of all other elements
            for i in final_state.index:
                final_state.loc[i, i] = 0
                final_state.loc[i, i] = initial_state[i] - final_state.loc[i].sum()

            inf_sites_guess = muts/occs
            inf_sites_guess = MarkovInfSites.series_tri_base_to_matrix(inf_sites_guess)

            groups.append({'initial_state': initial_state,
                           'final_state': final_state,
                           'inf_sites_guess': inf_sites_guess})

        return groups
    
    def genome_ratio_matrix(self):
        '''get genome wide weighting per category = p(XXX->Y)/SUM_y p(XXX->y)
        division by occ XXX in numerator and denomenator cancel out'''
        #index=XXX->Y, columns=bins
        muts=es.rename_cols(self.muts_dict_raw)      
        #sum all bins
        muts['muts']=muts.sum(axis=1)
        #extract trinuc context XXX from mut XXX->Y
        muts["trinuc"] = muts.index.astype(str).str[:3]
        # sum mutations over all alt bases for same trinuc
        muts["trinuc_sum"] = muts.groupby("trinuc")["muts"].transform("sum")
        # relative weight of each mut type within its trinuc context
        muts["weights"] = muts["muts"] / muts["trinuc_sum"]
        self.weights=muts['weights']
        return
        


        

    def get_vector_item_from_best_guess(self, best_guess, weights_matrix, occs_matrix=None, muts_matrix=None):
        '''returns the vector item from the best guess
        '''
       


        #editted so input a list of non cpgs and cpgs 
        #output dict with rate for each cpg and non cpg

        def get_weighted_mut(mutations):
            
            
            
            rates={}
            occs={}
            for mut in mutations:
                if str(mut) not in weights_matrix.index: mut = mut.get_rev_comp()
                
                if mut.get_backwards().tri not in best_guess[mut.tri]:
                    back_tri = es.get_rev_comp(mut.get_backwards().tri)
                else: back_tri = mut.get_backwards().tri
                value = best_guess.loc[mut.tri][back_tri]
                if mut.get_rev_comp()== mut.get_backwards(): continue
                
                occs[mut]= occs_matrix[str(mut)]  
                #totalocc += occs_matrix[str(mut)]/3  #add alt for each of 3 alts
                rates[mut]= value
                
            return rates,occs
            
        #self.genome_ratio_matrix() #get weights for occs when only allow mutations to one alt base
        muts_objects=es.get_mut_obj_list()
        muts_pool=[x for x in muts_objects if str(x) in self.mutations_pool]
        all_rates, all_occs=get_weighted_mut(muts_pool)
        
        return all_rates, all_occs
    def get_mrkv_corrected_vctrs(self, generations=None):
        if generations is None: generations = self.generations
        '''returns the best transition matrix for each group
        '''
        muts_matrices, occs_matrices = self.get_matrices()
        
        groups = self.get_markov_input_matrices(muts_matrices, occs_matrices)
        markov = MarkovInfSites(generations, self.prefix, self)
        orignial_prefix = self.prefix

        #initialise
        rates_dict = defaultdict(list)
        occs_dict=defaultdict(list)

       

        #helper function, append to dict of lists
        def append_dict_values(out, new):
            for key, value in new.items():
                out[key].append(value)

        for i, group in enumerate(groups):
            #self.write_logs(f'Calculating markov correction for group {i}')
            #make a new directory for each group
            markov.prefix = f'{orignial_prefix}/group_{i}/'
            if not os.path.exists(markov.prefix): os.mkdir(markov.prefix)
            markov.logs_f = markov.prefix + 'log.txt'
            group['inf_sites_guess'].to_csv(markov.prefix + 'observed_sub_matrix.csv', sep='\t')
            inf_sites_guess = group['inf_sites_guess']/generations
            inf_sites_guess.to_csv(markov.prefix + 'inf_sites_guess.csv', sep='\t')
            rates,occs = self.get_vector_item_from_best_guess(markov.get_best_guess(group['initial_state'],
                                                                                        group['final_state'],
                                                                                        group['inf_sites_guess']),
                                                                 muts_matrices[i]/occs_matrices[i],
                                                                 occs_matrices[i], muts_matrices[i])
            append_dict_values(rates_dict, rates)
            append_dict_values(occs_dict, occs)

        #optional convert bakc to standard dict
        rates_dict = dict(rates_dict)
        occs_dict = dict(occs_dict)
        
        self.write_logs(f'rates dict: {rates_dict}')
        
        return rates_dict, occs_dict


        