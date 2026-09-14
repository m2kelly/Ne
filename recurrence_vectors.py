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
        

        
        

    

    @classmethod
    def get_mutations_from_labels(cls, labels, pool=None):
        mutations = [es.mutation(label=m) for m in labels]
        if pool is not None:
            mutations = [m for m in mutations if str(m) in pool]
        return mutations

  
    def get_matrices(self):
        '''based on selected indices it extracts from the raw dicts a list
        of the summed occs and summed muts per category per big bin
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
        '''returns an initial state, final state and a inital guess for transition matrix for each matrix
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
    
     

    def get_vector_item_from_best_guess(self, best_guess, weights_matrix, occs_matrix=None, muts_matrix=None):
        '''returns the vector item from the best guess
        weights matrix=muts/occs
        works across full 64x 64 matrix
        '''
       


        #editted so input a list of non cpgs and cpgs 
        #output dict with rate for each cpg and non cpg

        def get_weighted_mut(mutations):
            #assuming at this point best guess matrix is ALREADY COLLAPSED-CHECK
            rates={}
            occs={}
            for mut in mutations:
                
                if mut.tri not in best_guess.index: mut = mut.get_rev_comp()
                
                if mut.get_backwards().tri not in best_guess[mut.tri]:
                    back_tri = es.get_rev_comp(mut.get_backwards().tri)
                else:
                    back_tri = mut.get_backwards().tri
                
                value = best_guess.loc[mut.tri][back_tri]
                #if mut.get_rev_comp()== mut.get_backwards(): continue
                
                occs[mut]= occs_matrix[str(mut)]  
                #totalocc += occs_matrix[str(mut)]/3  #add alt for each of 3 alts
                rates[mut]= value
                
            return rates,occs
            
        #self.genome_ratio_matrix() #get weights for occs when only allow mutations to one alt base
        muts_objects=es.get_mut_obj_list()
        muts_pool=[x for x in muts_objects if str(x) in self.mutations_pool]
        self.write_logs(f'muts pool {muts_pool}' )
        all_rates, all_occs=get_weighted_mut(muts_pool)
        
        return all_rates, all_occs
    
    def get_mrkv_corrected_vctrs(self, generations=None):
        '''
        returns the best transition matrix for each group by trying different transition matrices
        and minimizing the difference between the observed and expected final state
        '''
        if generations is None: generations = self.generations
        
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


        