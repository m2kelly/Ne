import pickle
import pandas as pd
from . import essentials as es
import warnings
from copy import deepcopy
from scipy.stats import kendalltau
warnings.filterwarnings('ignore')
from .base_operations import Operations

class BestNonCpGCandidates(Operations):

    def __init__(self, name, directory, best_smoothing, cpgs=None, non_cpg_pool=None, collapse=True,
                 muts_dict_raw=None, occ_dict_raw=None, cpg_remove_percentage=0, prefix='',
                 operations=None):

        Operations.__init__(self, name=name, directory=directory, collapse=collapse,
                            muts_dict_raw=muts_dict_raw, occ_dict_raw=occ_dict_raw ,
                            cpg_remove_percentage=cpg_remove_percentage, prefix=prefix,
                            operations=operations)

        self.best_window = best_smoothing
        self.smooth_dics()

        if cpgs is None:
            all_cpgs = es.mutation.get_cpg_muts()
            self.cpg_muts = [x for x in all_cpgs if str(x) in self.occ_dict_raw['chr1'].index]
        else: self.cpg_muts = [c for c in cpgs if str(c) in self.occ_dict_raw['chr1'].index]

        if non_cpg_pool is None:
            self.non_cpg_muts = [m for m in es.get_mut_obj_list() if str(m) in self.occ_dict_raw['chr1'].index]
            self.non_cpg_muts = [m for m in self.non_cpg_muts if m not in self.cpg_muts]
        else:
            self.non_cpg_muts = [m for m in non_cpg_pool if str(m) in self.occ_dict_raw['chr1'].index]
            self.non_cpg_muts = [m for m in self.non_cpg_muts if m not in self.cpg_muts]

        self.file_name = f'{self.prefix}best_correlations_smoothed_{name}.csv'
        self.best_candidates = None

    @classmethod
    def compare_mut_lists(cls, list1, list2):
        '''returns True if the two lists have the same mutations'''
        if len(list1) != len(list2):
            return False
        for mut in list1:
            if mut not in list2:
                return False
        return True

    @classmethod
    def group_was_done(cls, new_group, already_done):
        return any([cls.compare_mut_lists(new_group, group) for group in already_done])

    def get_added_new_corr(self, prev_best_corr, occ_dict=None, muts_dict=None):
        '''adds a new mutation to each of the previous best correlations,
        and returns the new correlations'''
        if occ_dict is None: occ_dict = deepcopy(self.occ_dict_smoothed)
        if muts_dict is None: muts_dict = deepcopy(self.muts_dict_smoothed)

        new_correlations = {}
        already_done = []
        for idx, group in enumerate(prev_best_corr.index):
            group = es.mutation.str_to_list(group)
            this_pool = [mut for mut in self.non_cpg_muts if mut not in group]

            for added_mut in this_pool:
                new_group = group + [added_mut]
                if self.group_was_done(new_group, already_done): continue

                already_done.append(new_group)

                cpg_vector, non_cpg_vector = es.condition_muts(muts_dict, occ_dict, self.cpg_muts, new_group)

                zipped = zip(cpg_vector, non_cpg_vector)
                zipped = sorted(zipped, key=lambda x: x[0])
                cpg_vector, non_cpg_vector = zip(*zipped)

                correlation = kendalltau(cpg_vector, non_cpg_vector)[0]
                correlation = round(correlation, 5)

                new_group_str = es.mutation.list_to_str(new_group)
                new_correlations[new_group_str] = correlation

            #print(f'---{idx}/{len(prev_best_corr.index)}---', end='\r')

        return pd.DataFrame(new_correlations.values(), index=new_correlations.keys(), columns=['correlation'])

    def _get_first_group(self):

        occ_dict = deepcopy(self.occ_dict_smoothed)
        muts_dict = deepcopy(self.muts_dict_smoothed)

        first_group = {}

        for mut in self.non_cpg_muts:

            cpg_vector, non_cpg_vector = es.condition_muts(muts_dict, occ_dict, self.cpg_muts, [mut])
            zipped = zip(cpg_vector, non_cpg_vector)
            zipped = sorted(zipped, key=lambda x: x[0])
            cpg_vector, non_cpg_vector = zip(*zipped)

            correlation = kendalltau(cpg_vector, non_cpg_vector)[0]
            correlation = round(correlation, 5)

            first_group[es.mutation.list_to_str([mut])] = [correlation]

        first_group = pd.DataFrame(first_group.values() , index=first_group.keys(), columns=['correlation'])
        first_group = first_group.sort_values('correlation', axis=0, ascending=False)

        first_group.to_csv(self.file_name, sep='\t')
        corrs = first_group.iloc[:50, :]
        return corrs

    def get_best_candidates(self):
        occ_dict = deepcopy(self.occ_dict_smoothed)
        muts_dict = deepcopy(self.muts_dict_smoothed)

        first_group = self._get_first_group()
        best_corr = first_group.sort_values('correlation', axis=0, ascending=False)
        corrs = best_corr.iloc[:50, :]

        top_candidate = deepcopy(best_corr.index[0])
        for i in range(1, (len(self.cpg_muts)*2)+1):
            print(f'---{i}---',end='\r')
            correlations = self.get_added_new_corr(corrs, occ_dict=occ_dict, muts_dict=muts_dict)

            correlations = correlations.sort_values('correlation', axis=0, ascending=False)
            #best_corr = best_corr.append(corrs)
            best_corr = pd.concat([best_corr, corrs], ignore_index=False)

            corrs = correlations.iloc[:10, :] #for the next iteration

            best_corr = best_corr.sort_values('correlation', axis=0, ascending=False)

            best_corr.to_csv(self.file_name, sep='\t', mode='w')
            if best_corr.index[0] == top_candidate and i!=1:
                break
            else:
                top_candidate = deepcopy(best_corr.index[0])

        self.write_logs(f'Best non-cpg candidates:{top_candidate}')
        top_candidate = top_candidate.split(',')
        top_candidate = [es.mutation(label=m) for m in top_candidate]
        self.best_candidates = top_candidate
        return top_candidate
