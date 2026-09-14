import pickle
from sklearn.linear_model import LinearRegression
import pandas as pd
from . import essentials as es
import warnings
from copy import deepcopy
from scipy.stats import kendalltau
warnings.filterwarnings('ignore')
from .base_operations import Operations
import numpy as np
import matplotlib.pyplot as plt


#maria addition to try classical beam search (allowing addition, removal and replacement)
class BestNonCpGCandidatesBeam(Operations):

    def __init__(self, name, directory, best_smoothing, cpgs=None, non_cpg_pool=None, collapse=True,
                 muts_dict_raw=None, occ_dict_raw=None, cpg_remove_percentage=0, cutoff=0.9, prefix='',
                 operations=None):

        Operations.__init__(self, name=name, directory=directory, collapse=collapse,
                            muts_dict_raw=muts_dict_raw, occ_dict_raw=occ_dict_raw ,
                            cpg_remove_percentage=cpg_remove_percentage, prefix=prefix,
                            operations=operations)

        

        self.cpg_remove_percentage=cpg_remove_percentage
        if cpgs is None:
            all_cpgs = es.mutation.get_cpg_muts()
            self.cpg_muts = [x for x in all_cpgs if str(x) in self.occ_dict_raw['chr1'].index]
        else: self.cpg_muts = [c for c in cpgs if str(c) in self.occ_dict_raw['chr1'].index]
        print(f'cpg muts: {self.cpg_muts}')
        if non_cpg_pool is None:
            self.non_cpg_muts = [m for m in es.get_mut_obj_list() if str(m) in self.occ_dict_raw['chr1'].index]
            self.non_cpg_muts = [m for m in self.non_cpg_muts if m not in self.cpg_muts]
        else:
            self.non_cpg_muts = [m for m in non_cpg_pool if str(m) in self.occ_dict_raw['chr1'].index]
            self.non_cpg_muts = [m for m in self.non_cpg_muts if m not in self.cpg_muts]

        self.file_name = f'{self.prefix}best_correlations_smoothed_{name}.csv'
        self.best_candidates = None

    def score_group(self, group, occ_vector, muts_vector):
        cpg_muts=[str(m) for m in self.cpg_muts]
        cpg_vector=muts_vector.loc[cpg_muts,:].sum() / occ_vector.loc[cpg_muts,:].sum()
        

        group=[str(m) for m in group]
        non_cpg_vector=muts_vector.loc[group,:].sum() / occ_vector.loc[group,:].sum()
        

        tau = kendalltau(cpg_vector, non_cpg_vector)[0]

        if np.isnan(tau):
            return -1

        return round(tau, 5)


    def generate_neighbors(self, group):
        '''from a list of categories generate all distance one neighbour groups
        from adding/subtracting/swapping one mut'''
        neighbors = []

        group_set = set(group)

        # ADD
        for mut in self.non_cpg_muts:
            if mut not in group_set:
                neighbors.append(group + [mut])

        # REMOVE
        if len(group) > 1:
            for mut in group:
                neighbors.append(
                    [m for m in group if m != mut]
                )

        # SWAP-optional
        '''
        for removed in group:
            reduced = [m for m in group if m != removed]

            for added in self.non_cpg_muts:
                if added not in reduced:
                    if added != removed:
                        neighbors.append(reduced + [added])
        '''


        return neighbors

    @staticmethod
    def sort_by_cpg_non_cpg_random(cpg,non_cpg):
        print('chosing cpg, non cpg orders')
        '''
        sort cpg and non cpg vectors
        choose each positoin by randomly chossing next smallest cpg or next smappest non cpg
        avoiding sorting by only one, giving flick effect'''
        cpg_sorted = cpg.sort_values(ascending=True).index.tolist()
        non_cpg_sorted = non_cpg.sort_values(ascending=True).index.tolist()
        
        indexes=[]
        i=0
        j=0
        cpg_non_cpg_choices=[]
        for x in range(len(cpg)):
            choice = np.random.choice([0, 1])
            if choice==0:
                if i>=len(cpg_sorted): #only intems in j list left, so have to choose from there
                    choosen_index= non_cpg_sorted[j]
                    j+=1
                    cpg_sorted.remove(choosen_index) #remove index from other list to avoid choosing it again
                    cpg_non_cpg_choices.append(1)   #1 is added non cpg, 0 is added cpg
                else:
                    choosen_index=cpg_sorted[i] #chosen index is the next smallest cpg
                    i+=1   
                    non_cpg_sorted.remove(choosen_index) #remove index from other list to avoid choosing it again
                    cpg_non_cpg_choices.append(0)

            else:
                if j>=len(non_cpg_sorted): #only intems in i list left, so have to choose from there
                    choosen_index= cpg_sorted[i]
                    i+=1 
                    non_cpg_sorted.remove(choosen_index)
                    cpg_non_cpg_choices.append(0)
                else:
                    choosen_index=non_cpg_sorted[j]
                    j+=1
                    cpg_sorted.remove(choosen_index)
                    cpg_non_cpg_choices.append(1)

            indexes.append(choosen_index)
            
        #print(i,j, indexes, cpg_non_cpg_choices)
        
        return indexes

    @staticmethod
    def remove_based_on_rank(cpg_vector,non_cpg_vector,cutoff=0.9):

        '''function to remove bins based on rank difference between cpg and non cpg rates'''

        cpg_rank = cpg_vector.rank(method='average')
        non_cpg_rank = non_cpg_vector.rank(method='average')

        # absolute rank difference
        rank_diff = np.abs(cpg_rank - non_cpg_rank)
        if cutoff < 1:
            cut=rank_diff.quantile(cutoff)
        else:
            cut = cutoff
        # keep indices within cutoff
        keep_indices = rank_diff[rank_diff <= cut].index
        return keep_indices
    
    
    
    def filter_dicts_low_mut_rank(self,occ_dict,muts_dict,cpg_labels=None,non_cpg_labels=None,cpg_remove_percentage=0.0,cutoff=0.9):
        '''function to filter the muts and occ dicts based on the cpg and non cpg labels
        used after selecting the best non cpg candidates
        remove sparse bins with occ<100k (max=300k)
        sort bins based on cpg and non cpg rates
        keep only cutoff percentage of bins with smallest rank difference between cpg list and non cpg list ranks
        then remove the cpg_remove_percentage least mutable bins based on randomly sorting cpg and non cpg
        return ordered list of indices to be kept'''
        occ_dict = deepcopy(occ_dict)
        muts_dict = deepcopy(muts_dict)
        #dict must actually be vector
        muts_vector=es.rename_cols(muts_dict)
        occ_vector=es.rename_cols(occ_dict)
        print(f"Initial number of bins: {len(occ_vector.columns)}")

        #remove bins with low total occ, max ==300k
        mask = occ_vector.sum(axis=0) >= 100000
        occ_vector = occ_vector.loc[:, mask]
        muts_vector = muts_vector.loc[:, mask]
        print(f"Remaining bins after filtering low coverage: {len(occ_vector.columns)}")

        if cpg_labels==None:
            cpg_labels=[str(x) for x in self.cpg_muts]
        if non_cpg_labels==None:
            non_cpg_labels=[x for x in muts_vector.index if x not in cpg_labels] #as collapse trinucs

        #cpg pooled rate
        cpg_muts = muts_vector.loc[cpg_labels].sum()
        cpg_occs_copy = occ_vector.loc[cpg_labels].sum()
        cpg_rate = cpg_muts / cpg_occs_copy


        #non cpg pooled rate
        non_cpg_muts = muts_vector.loc[non_cpg_labels].sum()
        non_cpg_occs_copy = occ_vector.loc[non_cpg_labels].sum()
        non_cpg_rate = non_cpg_muts / non_cpg_occs_copy

        #first remove based on rank -so removing same bins from for every cpg removal percentage
        keep_rank=self.remove_based_on_rank(cpg_rate,non_cpg_rate,cutoff=cutoff)

        #then remove low mutability bins based on cpg removal percentage, so different bins are removed for each cpg removal percentage
        cpg_rate=cpg_rate.loc[keep_rank]
        non_cpg_rate=non_cpg_rate.loc[keep_rank]

        #sorting based on cpg and non-cpg rates
        sorted_indexes=self.sort_by_cpg_non_cpg_random(cpg_rate,non_cpg_rate)

        
        to_keep_more_mutable=sorted_indexes[int(len(sorted_indexes)*cpg_remove_percentage):]

        return to_keep_more_mutable
    
    
   
    def filter_low_mut(self,occ_dict,muts_dict,remove_low):
        print(f'remove low = {remove_low}')
        '''
        removing low mutability bins (from cpg remove percentage)
        based on summed mutabiltiy across all cpg and non cpg categories-aka before chosing categories
        keep bins in the intersection of cpg>remove low & non cpg>remove low
        first remove bins with zero mutability
        '''
        occ_vector=es.rename_cols(occ_dict)
        muts_vector=es.rename_cols(muts_dict)
        
        print(f"starting bin length: {len(occ_vector.columns)}")
        
        
        mask = (occ_vector > 0).all(axis=0)
        occ_vector = occ_vector.loc[:, mask]
        muts_vector = muts_vector.loc[:, mask]

        print(f"bin length after filtering low coverage: {len(occ_vector.columns)}")
        
        cpg_muts=[str(m) for m in self.cpg_muts]
        cpg_vector=muts_vector.loc[cpg_muts,:].sum().div(occ_vector.loc[cpg_muts,:].sum().replace(0, np.nan))
        cpg_vector.dropna(inplace=True)
        cpg_vector=cpg_vector[cpg_vector>0]
        print(f"cpg vector length after filtering na,0: {len(cpg_vector)}")

        non_cpg_muts=[str(m) for m in self.non_cpg_muts]
        non_cpg_vector=muts_vector.loc[non_cpg_muts,:].sum().div(occ_vector.loc[non_cpg_muts,:].sum().replace(0, np.nan))
        non_cpg_vector.dropna(inplace=True)
        non_cpg_vector=non_cpg_vector[non_cpg_vector>0]
        print(f"non-cpg vector length after filtering na,0: {len(non_cpg_vector)}")
        
        
        
        cpg_high=cpg_vector[cpg_vector>=cpg_vector.quantile(remove_low)].index
        non_cpg_high=non_cpg_vector[non_cpg_vector>=non_cpg_vector.quantile(remove_low)].index
        #take intersect of high cpg and high non cpg 
        low_ints=list(set(cpg_high) & (set(non_cpg_high)))
        print(f"bin length after filtering low mutability bins: {len(low_ints)}")
        return low_ints

    def get_best_candidates(self,cpg_labels=None, non_cpg_labels=None):
        '''main function to perform beam search
        iteravily try removing/adding and swapping 1 category from 10 best scoring categories in previous iteration
        if increase score keep going
        else stop
        else keep going for max iter iterations'''
        beam_width=4
        max_iter=10
        occ_dict = deepcopy(self.occ_dict_raw)
        muts_dict = deepcopy(self.muts_dict_raw)
                                                                                        
        muts_vector=es.rename_cols(muts_dict)
        occ_vector=es.rename_cols(occ_dict)
        
        #filer normal 
        #keep_indices=self.filter_dicts_low_mut_rank(occ_dict,muts_dict,cpg_remove_percentage=self.cpg_remove_percentage,cutoff=self.cutoff)
        
        #filter only on low mut
        keep_indices=self.filter_low_mut(occ_dict,muts_dict,self.cpg_remove_percentage)

        #testing removing top 20% most mutable bins, based on cpg and non-cpg rates, 
        # to see if it improves correlation and gives more stable candidates
        # as not affected by recurrence
        #test
        #keep_indices=keep_indices[:int(len(keep_indices)*0.8)]
    
        occ_vector=occ_vector.loc[:, keep_indices]
        muts_vector=muts_vector.loc[:, keep_indices]
        
        
        # initialize with singletons
        beam = []

        for mut in self.non_cpg_muts:

            group = [mut]
            
            score = self.score_group(group, occ_vector, muts_vector)
            beam.append((group, score))

        beam = sorted(beam, key=lambda x: x[1], reverse=True)
        beam = beam[:beam_width]

        best_group, best_score = beam[0]
        #TESTING WITH ONE CANDIDATE
        #return best_group[0]
        visited = set()

        for iteration in range(max_iter):

            print(f"Iteration {iteration}")

            candidates = []

            for group, _ in beam:
                #finds all neighbours (from swaps, additions or removals)
                neighbors = self.generate_neighbors(group)

                for new_group in neighbors:

                    key = tuple(sorted(str(m) for m in new_group))
                    if key in visited:
                        continue

                    visited.add(key)
                    #score_group for kendall tau
                    score = self.score_group(
                        new_group,
                        occ_vector,
                        muts_vector
                    )

                    candidates.append((new_group, score))

            if len(candidates) == 0:
                break
            #beam search, sort best candidates and keeps
            candidates = sorted(
                candidates,
                key=lambda x: x[1],
                reverse=True
            )

            beam = candidates[:beam_width]
            print(f'{iteration}:{beam}')
            #if imporved correlation then add new score
            if beam[0][1] > best_score:
                best_group, best_score = beam[0]
            #enforce a minimum size of 4, to match number of cpgs
            elif len(beam[0][0]) <5:
                best_group, best_score = beam[0]


            else:
                self.write_logs(f"No improvement after {iteration} iterations")
                break

        self.write_logs(f'Best non-cpg candidates:{best_group}, tau:{best_score}')
        
        #top_candidate = [es.mutation(label=m) for m in best_group]
        self.plot_group_linear_regression(best_group, occ_vector, muts_vector)
        self.best_candidates = best_group
        
        return best_group