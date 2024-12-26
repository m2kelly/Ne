
import pickle
import pandas as pd
from copy import deepcopy
from . import essentials as es
import math
import numpy as np

class Operations():

    def __init__(self, name=None, directory=None, cpgs=None, non_cpg_pool=None, collapse=True, muts_dict_raw=None, occ_dict_raw=None, cpg_remove_percentage=0, prefix='', operations=None, silent=False):
        if operations is not None:
            self.__dict__ = deepcopy(operations.__dict__)
            return

        self.name = name
        self.directory = directory
        self.collapse = collapse
        self.cpgs = cpgs
        self.non_cpg_pool = non_cpg_pool
        self.silent = silent

        if muts_dict_raw is None or occ_dict_raw is None:
            self.muts_dict_raw, self.occ_dict_raw = self.get_muts_and_occs(name, directory)
        else:
            self.muts_dict_raw = muts_dict_raw
            self.occ_dict_raw = occ_dict_raw

        if collapse:
            self.muts_dict_raw = self.collapse_dicts(self.muts_dict_raw)
            self.occ_dict_raw = self.collapse_dicts(self.occ_dict_raw)

        self.muts_dict_CpGfiltered = None
        self.occ_dict_CpGfiltered = None

        self.cpg_remove_percentage = cpg_remove_percentage
        #self.filter_low_mut_CpGs()

        self.prefix = prefix
        self.logs_f = f'logs_{self.name}.txt'

        self.muts_dict_CpGfiltered_bootstrap = None
        self.occ_dict_CpGfiltered_bootstrap = None
        self.muts_dict_smoothed = None
        self.occ_dict_smoothed = None


    def filter_other_muts(self, muts_to_keep):
        to_keep = [str(m) for m in muts_to_keep]
        for chrom in self.muts_dict_raw:
            self.muts_dict_raw[chrom] = self.muts_dict_raw[chrom].loc[to_keep, :]
            self.occ_dict_raw[chrom] = self.occ_dict_raw[chrom].loc[to_keep, :]

        if self.muts_dict_CpGfiltered is None: return
        for chrom in self.muts_dict_raw:
            self.muts_dict_CpGfiltered[chrom] = self.muts_dict_CpGfiltered[chrom].loc[to_keep, :]
            self.occ_dict_CpGfiltered[chrom] = self.occ_dict_CpGfiltered[chrom].loc[to_keep, :]

    def filter_low_both(self, filter_small_bins):
        muts = deepcopy(self.muts_dict_raw); occs = deepcopy(self.occ_dict_raw)
        if filter_small_bins: muts, occs = es.filter_small_bins_dict(muts, occs)
        self.muts_dict_CpGfiltered, self.occ_dict_CpGfiltered = es.filter_low_both_cats(muts, occs,
                                                                                        self.cpg_remove_percentage,
                                                                                        self.cpgs, self.non_cpg_pool)


    def filter_low_mut_CpGs(self):
        muts = deepcopy(self.muts_dict_raw); occs = deepcopy(self.occ_dict_raw)
        self.muts_dict_CpGfiltered, self.occ_dict_CpGfiltered = es.filter_low_mut_CpGs(muts, occs,
                                                                            self.cpg_remove_percentage,
                                                                            self.cpgs)


    def bootstrap_filter(self):
        if self.muts_dict_CpGfiltered_bootstrap is None:
            self.muts_dict_CpGfiltered_bootstrap = deepcopy(self.muts_dict_smoothed)
            self.occ_dict_CpGfiltered_bootstrap = deepcopy(self.occ_dict_smoothed)

        valid_indices = np.where(self.occ_dict_CpGfiltered_bootstrap.sum()>0)[0]
        valid_indices = [i+1 for i in valid_indices]
        indices = np.random.choice(valid_indices, size=len(valid_indices), replace=True)

        self.muts_dict_smoothed = self.muts_dict_CpGfiltered_bootstrap[indices]
        self.occ_dict_smoothed = self.occ_dict_CpGfiltered_bootstrap[indices]

    def smooth_dics(self, muts_dict=None, occ_dict=None, best_smoothing=None,
                    cpg_remove_percentage=None):

        if muts_dict is None: muts_dict = self.muts_dict_CpGfiltered
        if occ_dict is None: occ_dict = self.occ_dict_CpGfiltered
        if best_smoothing is None: best_smoothing = self.best_window
        if cpg_remove_percentage is None: cpg_remove_percentage = self.cpg_remove_percentage


        self.muts_dict_smoothed, self.occ_dict_smoothed = es.smooth_dicts(muts_dict,
                                                                          occ_dict,
                                                                          window_size=best_smoothing)
        self.muts_dict_smoothed = es.rename_cols(self.muts_dict_smoothed)
        self.occ_dict_smoothed  = es.rename_cols(self.occ_dict_smoothed)


    def get_h_mono_and_l_mono(self, ignored_pairs=[], skip_cpgs=True, len_h=None, len_l=None):
        all_muts = None; all_occs = None
        for chrom in es.CHROMS:
            if chrom not in self.muts_dict_raw: continue
            if all_muts is None:
                all_muts = self.muts_dict_raw[chrom].sum(axis=1)
                all_occs = self.occ_dict_raw[chrom].sum(axis=1)
            else:
                all_muts += self.muts_dict_raw[chrom].sum(axis=1)
                all_occs += self.occ_dict_raw[chrom].sum(axis=1)

        cpgs = [str(m) for m in es.mutation.get_cpg_muts()]
        sorted_matrix = all_muts/all_occs
        sorted_matrix = sorted_matrix.sort_values(ascending=False)
        sorted_matrix = sorted_matrix.dropna()
        mirrored_mutations  = [str(m) for m in sorted_matrix.index if es.mutation(label=m).is_mirrored()]


        if skip_cpgs: sorted_matrix = sorted_matrix[~sorted_matrix.index.isin(cpgs)]
        sorted_matrix = sorted_matrix[~sorted_matrix.index.isin(mirrored_mutations)]

        sorted_matrix = sorted_matrix.dropna()

        muts = [es.mutation(label=str(m)) for m in sorted_matrix.index]
        ranges = {}

        for source in es.BASES:
            for target in es.BASES:
                if source == target: continue
                skip_pair = False
                for pair in ignored_pairs:
                    if len(pair) != 2: raise ValueError('ignored_pairs must be a list of tuples of length 2')
                    if source == pair[0] and target == pair[1]: skip_pair = True
                if skip_pair: continue
                min_mut = sorted_matrix[sorted_matrix.index.str.contains(rf'^.{source}.+{target}$')].min()
                max_mut = sorted_matrix[sorted_matrix.index.str.contains(rf'^.{source}.+{target}$')].max()
                if math.isnan(min_mut) or math.isnan(max_mut): continue
                ranges[(source, target)] = (min_mut, max_mut)

        h_source, h_target = max(ranges, key=lambda x: ranges[x][1])
        all_muts = [m for m in muts if m.tri[1] == h_source and m.base == h_target]

        if len_h is not None:
                h_mono = all_muts[:len_h]
                h_matrix = sorted_matrix[sorted_matrix.index.isin([str(m) for m in h_mono])]
        else:
            h_mono = all_muts[:int(len(all_muts)/2)]
            h_matrix = sorted_matrix[sorted_matrix.index.isin([str(m) for m in h_mono])]
            h_matrix = h_matrix[h_matrix > h_matrix.quantile(0.2)]

        h_mono = [es.mutation(label=m) for m in h_matrix.index]

        if len_l is not None:
            len_l = min(len_l, abs(len(all_muts) - len(h_mono)))
            l_mono = all_muts[-len_l:]
            l_matrix = sorted_matrix[sorted_matrix.index.isin([str(m) for m in l_mono])]

        else:
            l_mono = all_muts[int(len(all_muts)/2):]
            l_matrix = sorted_matrix[sorted_matrix.index.isin([str(m) for m in l_mono])]
            l_matrix = l_matrix[l_matrix < l_matrix.quantile(0.8)]

        l_mono = [es.mutation(label=m) for m in l_matrix.index]

        self.write_logs(f'h_non_cpgs({len(h_mono)}): {h_mono}')
        self.write_logs(f'l_non_cpgs({len(l_mono)}): {l_mono}')
        self.write_logs(f'sorted matrix: {sorted_matrix[sorted_matrix.index.isin([str(m) for m in h_mono+ l_mono])]}')

        return h_mono, l_mono


    @classmethod
    def get_muts_and_occs(cls, name, directory):

        full_directory = f'{directory}/{name}/'
        name_prefix = f'whole_genome_dict_{name}_unsmoothed'

        muts_dict_dir = f'{full_directory}/{name_prefix}_muts.pkl'
        occ_dict_dir = f'{full_directory}/{name_prefix}_occs.pkl'
        muts_dict_raw = pickle.load(open(muts_dict_dir, 'rb'))
        occ_dict_raw = pickle.load(open(occ_dict_dir, 'rb'))
        return muts_dict_raw, occ_dict_raw


    @classmethod
    def collapse_dicts(cls, dicts):
        #check if the dict is already collapsed
        if len(dicts['chr1'].T.columns) == len(es.get_muts_sig_ordered()):
            return dicts

        collapsed_dicts = {}
        frwrd_muts = es.get_muts_sig_ordered()
        for chrom in es.CHROMS:
            #check if chrom is in the dicts
            if chrom not in dicts: continue
            df = dicts[chrom]

            new_df = pd.DataFrame(columns=[str(f) for f in frwrd_muts])
            for frwrd_mut in frwrd_muts:
                new_df[str(frwrd_mut)] = df.T[str(frwrd_mut)] + df.T[str(frwrd_mut.get_rev_comp())]

            collapsed_dicts[chrom] = new_df.T
        return collapsed_dicts

    def write_logs(self, message):
        with open(self.logs_f, 'a') as f:
            f.write(f'{message}\n')
        print(message)

    def clear_logs(self):
        with open(self.logs_f, 'w') as f:
            f.write('')
