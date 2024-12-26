import pickle
import pandas as pd
import matplotlib.pyplot as plt
from . import essentials as es
import seaborn as sns
import warnings
from copy import deepcopy
from scipy.stats import kendalltau
from kneed import KneeLocator
from .base_operations import Operations


warnings.filterwarnings('ignore')

class BestSmoothing(Operations):

    def __init__(self, name, directory, cpgs=None, non_cpg_pool=None, smoothing_range=None, collapse=True,
                 muts_dict_raw=None, occ_dict_raw=None, cpg_remove_percentage=0, prefix='',
                 operations=None):


        Operations.__init__(self, name=name, directory=directory, collapse=collapse,
                            muts_dict_raw=muts_dict_raw, occ_dict_raw=occ_dict_raw,
                            cpg_remove_percentage=cpg_remove_percentage, prefix=prefix,
                            operations=operations, cpgs=cpgs, non_cpg_pool=non_cpg_pool)

        if cpgs is None:
            all_cpgs = es.mutation.get_cpg_muts()
            self.cpg_muts = [x for x in all_cpgs if str(x) in self.occ_dict_CpGfiltered['chr1'].index]
        else:
            self.cpg_muts = [c for c in cpgs if str(c) in self.occ_dict_CpGfiltered['chr1'].index]

        if non_cpg_pool is None:
            self.non_cpg_muts = [m for m in es.get_mut_obj_list() if str(m) in self.occ_dict_CpGfiltered['chr1'].index]
            self.non_cpg_muts = [m for m in self.non_cpg_muts if m not in self.cpg_muts]
        else:
            self.non_cpg_muts = [m for m in non_cpg_pool if str(m) in self.occ_dict_CpGfiltered['chr1'].index]
            self.non_cpg_muts = [m for m in self.non_cpg_muts if m not in self.cpg_muts]

        if smoothing_range is None: self.smoothing_range =  lambda: range(2, 101, 1)
        else: self.smoothing_range = smoothing_range

        self.smoothing_correlations_cpgs = []
        self.smoothing_correlations_cpgs_noncpgs = []
        self.smoothing_correlations_diff = []

        self.number_of_bins = []
        self.biggest_heatmap = None
        self.biggest_window = None
        self.best_window = None

    def calc_cpgs_correlations(self):
        self.write_logs('Calculating cpgs correlations')

        self.smoothing_correlations_cpgs = []
        self.number_of_bins = []

        #occ_dict = deepcopy(self.occ_dict_CpGfiltered)
        #muts_dict = deepcopy(self.muts_dict_CpGfiltered)
        best_correlation = -1*float('inf')
        for smoothing in self.smoothing_range():
            print(f'---{smoothing}---', end='\r')

            occ, muts = es.smooth_dicts(deepcopy(self.occ_dict_CpGfiltered), deepcopy(self.muts_dict_CpGfiltered), window_size=smoothing)
            muts = es.rename_cols(muts)
            occ = es.rename_cols(occ)

            heatmap = pd.DataFrame(0, index=[str(mut) for mut in self.cpg_muts],
                                    columns=[str(mut) for mut in self.cpg_muts])

            for idx, mut1 in enumerate(self.cpg_muts):
                for mut2 in self.cpg_muts[idx:]:
                    mut1vector, mut2vector = es.condition_muts(muts, occ, [mut1], [mut2], drop_small_bins=False)


                    zipped = zip(mut1vector, mut2vector)
                    zipped = sorted(zipped, key=lambda x: x[0])
                    mut1vector, mut2vector = zip(*zipped)

                    correlation = kendalltau(mut1vector, mut2vector)[0]

                    #approximate
                    correlation = round(correlation, 5)
                    heatmap.loc[str(mut1), str(mut2)] = correlation
                    heatmap.loc[str(mut2), str(mut1)] = correlation

            self.number_of_bins.append(len(mut1vector))


            correlation_sum = heatmap.sum().sum()
            self.smoothing_correlations_cpgs.append(correlation_sum)
            if correlation_sum > best_correlation:
                best_correlation = correlation_sum
                self.biggest_heatmap = heatmap
                self.biggest_window = smoothing

    def calc_cpgs_noncpgs_corrs(self):
        self.write_logs('Calculating cpgs non-cpgs correlations')


        for m in self.cpg_muts:
            if m in self.non_cpg_muts: self.non_cpg_muts.remove(m)

        self.smoothing_correlations_cpgs_noncpgs = []
        self.smoothing_correlations_diff = []

        for idx, smoothing in enumerate(self.smoothing_range()):
            print(f'---{smoothing}---', end='\r')
            occ, muts = es.smooth_dicts(deepcopy(self.occ_dict_CpGfiltered), deepcopy(self.muts_dict_CpGfiltered), window_size=smoothing)

            muts = es.rename_cols(muts)
            occ = es.rename_cols(occ)


            cpg_vec, noncpg_vec = es.condition_muts(muts, occ, self.cpg_muts, self.non_cpg_muts)

            zipped = zip(noncpg_vec, cpg_vec)
            zipped = sorted(zipped, key=lambda x: x[1])
            noncpg_vec, cpg_vec = zip(*zipped)

            correlation = kendalltau(noncpg_vec, cpg_vec)[0]


            #approximate
            correlation = round(correlation, 5)
            self.smoothing_correlations_cpgs_noncpgs.append(correlation)
            avg_cpg_corr = (self.smoothing_correlations_cpgs[idx]-len(self.cpg_muts))/((len(self.cpg_muts)**2)-len(self.cpg_muts))
            self.smoothing_correlations_diff.append(avg_cpg_corr- correlation)

    def get_best_smoothing_window(self):
        if not self.smoothing_correlations_cpgs: self.calc_cpgs_correlations()
        if not self.smoothing_correlations_cpgs_noncpgs: self.calc_cpgs_noncpgs_corrs()
        kn = KneeLocator(self.smoothing_range(), self.smoothing_correlations_cpgs_noncpgs, curve='concave', direction='increasing')
        self.best_window = kn.knee
        return(kn.knee)

    def get_best_smoothing_window_diff(self):
        if not self.smoothing_correlations_diff: self.calc_cpgs_noncpgs_corrs()
        kn = KneeLocator(self.smoothing_range()[5:], self.smoothing_correlations_diff[5:], curve='concave', direction='increasing')
        return(kn.knee)

    def plot_smoothing_range(self):
        plt.clf()
        plt.plot(self.smoothing_range(), self.smoothing_correlations_cpgs_noncpgs)
        plt.xlabel('Smoothing window size')
        plt.ylabel('coorelation between cpg and non-cpg (pooled)')
        #plt.show()
        return plt

    def plot_smoothing_diff(self):
        plt.clf()
        plt.plot(self.smoothing_range(), self.smoothing_correlations_diff)
        plt.xlabel('Smoothing window size')
        plt.ylabel('Average correlation amoung cpgs - correlation between cpg and non-cpg pooled')
        #plt.show()
        return plt

    def plot_max_corr_heatmap(self):
        plt.clf()
        sns.heatmap(self.biggest_heatmap, cmap='coolwarm', annot=True)
        plt.title(f'Best window size: {self.biggest_window} (gaussian, std=window_size/2)')
        #plt.show()
        return plt
