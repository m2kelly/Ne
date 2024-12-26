
import re
import scipy.stats
import pandas as pd
import numpy as np
#from parallel_pandas import ParallelPandas
from copy import deepcopy
#ParallelPandas.initialize(n_cpu=8, split_factor=1)

import math

BASES = ['A', 'C', 'G', 'T']
CONTEXTS = [a+b+c for a in BASES for b in BASES for c in BASES]
CHROMS = [f'chr{str(i)}' for i in range(1, 23)] + ['chrX', 'chrY']

def get_gene_name(line):

    attrs = line.split(";")[:-1]
    for attr in attrs:
        attr = attr.split()
        if attr[0] == "gene_name":
            return re.sub('\"', '', attr[1])

    return 'Not_a_gene'

def get_rev_comp(sequence):
    rev_comp_dict = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}
    rev_comp = ''
    for base in sequence:
        rev_comp += rev_comp_dict[base]
    return rev_comp[::-1]

def matrix_to_series(matrix):
    series = matrix.stack()
    series.index = series.index.map(lambda s: '->'.join(map(str, s)))
    return series



class mutation:
    def __init__(self, tri=None, base=None, label=None):

        if label is not None and tri is None and base is None:
            tri, base = label.split('->')
        if tri is None or base is None: raise ValueError('Not enough arguments')
        self.tri = tri
        self.base = base
        self.label = tri + '->' + base

    def get_backwards(self):
        tri = self.tri[0] + self.base + self.tri[-1]
        base = self.tri[1]
        return mutation(tri, base)

    def get_rev_comp(self):
        return mutation(get_rev_comp(self.tri), get_rev_comp(self.base))

    @staticmethod
    def list_to_str(muts):
        return f'{",".join([str(i) for i in muts])}'

    @staticmethod
    def str_to_list(string):

        muts = []
        for label in string.split(','):
            tri, base = label.split('->')
            muts.append(mutation(tri, base))
        return muts

    @staticmethod
    def get_cpg_muts():
        cpg_transitions = [mutation(tri, base) for tri in CONTEXTS for base in BASES if tri[1:]=='CG' and base == 'T']
        cpg_transitions += [mutation(tri, base) for tri in CONTEXTS for base in BASES if tri[1:]=='GC' and base == 'A']
        return cpg_transitions

    def is_mirrored(self):
        return self.get_backwards() == self.get_rev_comp()

    def is_cpg(self):
        return 'CG' in self.tri

    def is_transition(self):
        if self.tri[1] == 'A' and self.base == 'G': return True
        if self.tri[1] == 'G' and self.base == 'A': return True
        if self.tri[1] == 'C' and self.base == 'T': return True
        if self.tri[1] == 'T' and self.base == 'C': return True
        return False

    def __eq__(self, other):
        return self.tri == other.tri and self.base == other.base

    def __hash__(self):
        return hash(self.label)

    def __str__(self):
        return self.label

    def __repr__(self):
        return self.label

def get_mut_obj_list():
    return [mutation(tri, base) for tri in CONTEXTS for base in BASES if base != tri[1]]

def get_muts_sig_ordered():

    muts = get_mut_obj_list()
    sort_bases = ['C', 'T']
    muts_sorted = []
    for base in sort_bases:
        muts_group = [m for m in muts if m.tri[1] == base]
        muts_group.sort(key=lambda x: x.base)
        muts_sorted += muts_group
    return muts_sorted

def get_general_non_cpg_pool():
    '''return all non cpg mutations after removing mutations where backwords mutation==rev_comp mutation
    '''
    non_cpg_pool = [m for m in get_mut_obj_list() if not (m.is_cpg() and m.is_transition())] #remove cpg transitions
    non_cpg_pool = [m for m in non_cpg_pool if not m.is_mirrored()]
    return non_cpg_pool

def rename_cols(dict):

    global_df = pd.DataFrame()
    last_col = 0
    for chrom in CHROMS:
        if chrom not in dict: continue
        df = dict[chrom]
        df.columns = [i + last_col for i in range(1,len(df.columns)+1)]
        if not df.empty:
            last_col = df.columns[-1]
            global_df = pd.concat([global_df, df], axis=1)

    return global_df

def condition_muts(muts_vector, occ_vector, group1, group2,
                     filter_low_muts=False, drop_small_bins=False):

    if drop_small_bins:
        index = (occ_vector.sum()/3 > 29000)
        muts_vector = muts_vector.loc[:, index]
        occ_vector = occ_vector.loc[:, index]

    source_index = [str(mut) for mut in group1]
    source_vector = muts_vector.loc[source_index, :].sum()
    source_vector /= occ_vector.loc[source_index, :].sum()

    target_index = [str(mut) for mut in group2]
    target_vector = muts_vector.loc[target_index, :].sum()
    target_vector /= occ_vector.loc[target_index, :].sum()

    #remove the lowest 10% of vectors
    if filter_low_muts:
        source_vector = source_vector.sort_values()
        target_vector = target_vector.sort_values()
        source_vector = source_vector.iloc[int(len(source_vector)*0.05):]
        target_vector = target_vector.iloc[int(len(target_vector)*0.05):]

    #remove positions with 0 mutation rate in both vectors
    zero_source_index = source_vector.index[source_vector == 0]
    zero_target_index = target_vector.index[target_vector == 0]
    #zero_index = list(set(zero_source_index) & set(zero_target_index))
    zero_index = list(set(zero_source_index) | set(zero_target_index))
    source_vector.drop(zero_index, inplace=True)
    target_vector.drop(zero_index, inplace=True)

    #remove na values
    na_index = source_vector.index[source_vector.isna()]
    na_index = list(set(na_index) | set(target_vector.index[target_vector.isna()]))
    source_vector.drop(na_index, inplace=True, errors='ignore')
    target_vector.drop(na_index, inplace=True, errors='ignore')



    return source_vector, target_vector

def condition_muts_all(muts_vector, occ_vector, groups,
                     filter_low_muts=False, drop_small_bins=False):

    if drop_small_bins:
        index = (occ_vector.sum()/3 > 29000)
        muts_vector = muts_vector.loc[:, index]
        occ_vector = occ_vector.loc[:, index]

    vectors = []
    for group in groups:
        index = [str(mut) for mut in group]
        vector = muts_vector.loc[index, :].sum()
        vector /= occ_vector.loc[index, :].sum()
        vectors.append(vector)

    #remove the lowest 10% of vectors
    if filter_low_muts:
        filtered_vectors = []
        for vector in vectors:
            vector = vector.sort_values()
            vector = vector.iloc[int(len(vector)*0.05):]
            filtered_vectors.append(vector)


    #remove positions with 0 mutation rate in any vector
    zero_index = []
    for vector in vectors: zero_index += list(vector.index[vector == 0])

    zero_index = list(set(zero_index))
    for vector in vectors: vector.drop(zero_index, inplace=True)


    #remove na values
    na_index = []
    for vector in vectors: na_index += list(vector.index[vector.isna()])

    na_index = list(set(na_index))
    for vector in vectors: vector.drop(na_index, inplace=True, errors='ignore')

    return vectors

def smooth_dicts(*dicts, window_size=2):

    weights = scipy.signal.gaussian(window_size, std=window_size/3)


    def weighted_mean_with_nans(x):

        # Mask to exclude NaNs
        mask = ~np.isnan(x)

        # Apply the mask to weights and x
        valid_weights = weights[mask]
        valid_x = x[mask]

        # Calculate weighted mean
        if valid_weights.sum() < 0.5: #test this
            return np.nan
        return np.dot(valid_weights, valid_x) / valid_weights.sum()

    '''def weighted_mean_with_nans(x):

        sum_wights = 0
        weighted_mean = 0
        for i in range(len(x)):
            xx = list(x)
            if not math.isnan(xx[i]):
                sum_wights += weights[i]
                weighted_mean += weights[i]*xx[i]

        if sum_wights == 0: return None
        return weighted_mean/sum_wights
    '''
    smotthed_dicts = []
    for dict in dicts:
        smoothed_dict = {}
        for chrom in CHROMS:

            try: muts = dict[chrom].T
            except KeyError: continue

            nan_index = muts.index[muts.isna().sum(axis=1) > 0]
            zero_index = muts.index[muts.sum(axis=1) == 0]
            joined_index = list(set(nan_index) | set(zero_index))

            #smoothed_muts = muts.rolling(window_size, center=True, axis=0, win_type='gaussian', min_periods=1).mean(std=window_size/3)

            #muts = muts.fillna(value=0)
            #change type to float
            muts = muts.astype(float)

            smoothed_muts = muts.rolling(window_size, center=True, axis=0, min_periods=0).apply(weighted_mean_with_nans, engine='numba', raw=True)


            smoothed_muts = smoothed_muts.drop(smoothed_muts.index[nan_index])

            #smoothed_muts = smoothed_muts.T
            #smoothed_occ = smoothed_occ.T

            #remove nan values
            smoothed_muts = smoothed_muts.dropna(axis=1)

            smoothed_dict[chrom] = smoothed_muts.T

        smotthed_dicts.append(smoothed_dict)

    return smotthed_dicts

def filter_small_bins_dict(muts_dict_raw, occ_dict_raw, min_binsize=None):

    if min_binsize==None:
        all_occ = []
        for chrom in CHROMS:
            if chrom not in occ_dict_raw: continue
            occ = occ_dict_raw[chrom]
            all_occ.extend(list(occ.sum()))
        all_occ = [l for l in all_occ if l>0]
        min_binsize = np.percentile(np.array(all_occ), 5)

    muts_dict_filtered = {}
    occ_dict_filtered = {}
    for chrom in CHROMS:
        if chrom not in occ_dict_raw: continue
        muts = muts_dict_raw[chrom]
        occ = occ_dict_raw[chrom]

        index = (occ.sum() > min_binsize)
        inverse_index = [not i for i in index]
        #change reverese index to nan
        muts.loc[:, inverse_index] = None
        occ.loc[:, inverse_index] = None

        muts_dict_filtered[chrom] = muts
        occ_dict_filtered[chrom] = occ

    return muts_dict_filtered, occ_dict_filtered

def filter_small_bins_vector(muts_vector, occ_vector, min_binsize=0):

    index = (occ_vector.sum()/3 > min_binsize)
    muts_vector = muts_vector.loc[:, index]
    occ_vector = occ_vector.loc[:, index]

    return muts_vector, occ_vector

def filter_cpgs(muts_dict, occ_dict, threshold=0.005, keep=lambda x, y: x > y):

    muts_dict_filtered = {}
    occ_dict_filtered = {}
    for chrom in CHROMS:
        if chrom not in occ_dict: continue
        muts = muts_dict[chrom]
        occ = occ_dict[chrom]
        cpgs = [str(m) for m in get_mut_obj_list() if 'CG' in m.tri]
        cpgs = list(set(cpgs).intersection(set(occ.index)))
        #print(cpgs)

        cpg_counts = occ.loc[cpgs, :].sum()
        cpg_percent = cpg_counts/occ.sum()

        index = keep(cpg_percent, threshold)
        #index = cpg_percent > threshold

        muts_dict_filtered[chrom] = muts.loc[:, index]
        occ_dict_filtered[chrom] = occ.loc[:, index]

    return muts_dict_filtered, occ_dict_filtered

def filter_low_mut_CpGs(muts_dict_o, occ_dict_o, cpg_remove_percentage=0, cpgs=None):
    muts_dict = deepcopy(muts_dict_o)
    occ_dict = deepcopy(occ_dict_o)
    muts_dict_filtered = {}
    occ_dict_filtered = {}
    list_of_mutability = []
    for chrom in CHROMS:
        if chrom not in occ_dict: continue
        muts = muts_dict[chrom]
        occ = occ_dict[chrom]
        if cpgs is not None: cpgs_to_t = [str(m) for m in cpgs]
        else: cpgs_to_t = [str(m) for m in get_mut_obj_list() if 'CG' == m.tri[1:] and m.base == 'T']
        cpgs_to_t = list(set(cpgs_to_t).intersection(set(occ.index)))

        cpg_occs = occ.loc[cpgs_to_t, :].sum()
        cpg_muts = muts.loc[cpgs_to_t, :].sum()
        cpg_percent = cpg_muts.div(cpg_occs.replace(0, np.nan))
        cpg_percent = cpg_percent.dropna()
        cpg_percent = cpg_percent[cpg_percent > 0]
        cpg_percent = list(cpg_percent)
        list_of_mutability.extend(cpg_percent)
    #cuttoff = the value of the lowest thtrshold% of the mutability of the CpGs
    #sort list_of_mutability
    list_of_mutability.sort()
    cuttoff = list_of_mutability[int(len(list_of_mutability)*cpg_remove_percentage)]
    print(f'CpG cuttoff is {cuttoff}')

    for chrom in CHROMS:
        if chrom not in occ_dict: continue
        muts = muts_dict[chrom]; muts = muts.astype(float)
        occ = occ_dict[chrom]; occ = occ.astype(float)

        if cpgs is not None: cpgs_to_t = [str(m) for m in cpgs]
        else: cpgs_to_t = [str(m) for m in get_mut_obj_list() if 'CG' == m.tri[1:] and m.base == 'T']

        cpgs_to_t = list(set(cpgs_to_t).intersection(set(occ.index)))
        cpg_occs = occ.loc[cpgs_to_t, :].sum()
        cpg_muts = muts.loc[cpgs_to_t, :].sum()
        cpg_percent = cpg_muts/cpg_occs
        index = cpg_percent > cuttoff
        inverse_index = [not i for i in index]
        #change reverese index to nan
        muts.loc[:, inverse_index] = None
        occ.loc[:, inverse_index] = None

        muts_dict_filtered[chrom] = muts
        occ_dict_filtered[chrom] = occ

        #muts_dict_filtered[chrom] = muts.loc[:, index]
        #occ_dict_filtered[chrom] = occ.loc[:, index]

    return muts_dict_filtered, occ_dict_filtered

def get_filter_low_index(muts_dict_o, occ_dict_o, cpg_remove_percentage=0, cpgs=None):
    muts_dict = deepcopy(muts_dict_o)
    occ_dict = deepcopy(occ_dict_o)
    muts_dict_filtered = {}
    occ_dict_filtered = {}
    list_of_mutability = []
    for chrom in CHROMS:
        if chrom not in occ_dict: continue
        muts = muts_dict[chrom]
        occ = occ_dict[chrom]
        if cpgs is not None: cpgs_to_t = [str(m) for m in cpgs]
        else: cpgs_to_t = [str(m) for m in get_mut_obj_list() if 'CG' == m.tri[1:] and m.base == 'T']
        cpgs_to_t = list(set(cpgs_to_t).intersection(set(occ.index)))

        cpg_occs = occ.loc[cpgs_to_t, :].sum()
        cpg_muts = muts.loc[cpgs_to_t, :].sum()
        cpg_percent = cpg_muts.div(cpg_occs.replace(0, np.nan))
        cpg_percent = cpg_percent.dropna()
        cpg_percent = cpg_percent[cpg_percent > 0]
        cpg_percent = list(cpg_percent)
        list_of_mutability.extend(cpg_percent)
    #cuttoff = the value of the lowest thtrshold% of the mutability of the CpGs
    #sort list_of_mutability
    list_of_mutability.sort()
    cuttoff = list_of_mutability[int(len(list_of_mutability)*cpg_remove_percentage)]
    indeces = {}
    for chrom in CHROMS:
        if chrom not in occ_dict: continue
        muts = muts_dict[chrom]; muts = muts.astype(float)
        occ = occ_dict[chrom]; occ = occ.astype(float)

        if cpgs is not None: cpgs_to_t = [str(m) for m in cpgs]
        else: cpgs_to_t = [str(m) for m in get_mut_obj_list() if 'CG' == m.tri[1:] and m.base == 'T']

        cpgs_to_t = list(set(cpgs_to_t).intersection(set(occ.index)))
        cpg_occs = occ.loc[cpgs_to_t, :].sum()
        cpg_muts = muts.loc[cpgs_to_t, :].sum()
        cpg_percent = cpg_muts/cpg_occs
        index = cpg_percent > cuttoff
        indeces[chrom] = index
    return indeces

def filter_low_both_cats(muts_dict_o, occ_dict_o, cpg_remove_percentage=0, cpgs=None, non_cpgs=None):
    cpg_filter = get_filter_low_index(muts_dict_o, occ_dict_o, cpg_remove_percentage, cpgs)
    non_cpg_filter = get_filter_low_index(muts_dict_o, occ_dict_o, cpg_remove_percentage, non_cpgs)
    muts_dict = deepcopy(muts_dict_o)
    occ_dict = deepcopy(occ_dict_o)

    muts_dict_filtered = {}
    occ_dict_filtered = {}
    len_index = 0; len_cpg_index = 0; len_non_cpg_index = 0
    for chrom in CHROMS:

        if chrom not in occ_dict: continue
        muts = muts_dict[chrom]; muts = muts.astype(float)
        occ = occ_dict[chrom]; occ = occ.astype(float)
        commmon_index = cpg_filter[chrom] & non_cpg_filter[chrom]
        muts.loc[:, ~commmon_index] = None
        occ.loc[:, ~commmon_index] = None
        len_index += len([i for i in commmon_index if i])
        len_cpg_index += len([i for i in cpg_filter[chrom] if i])
        len_non_cpg_index += len([i for i in non_cpg_filter[chrom] if i])
        muts_dict_filtered[chrom] = muts
        occ_dict_filtered[chrom] = occ

    print(f'len index: {len_index}')
    print(f'len cpg index: {len_cpg_index}')
    print(f'len non cpg index: {len_non_cpg_index}')
    return muts_dict_filtered, occ_dict_filtered
