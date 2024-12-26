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

class ReccurenceVectors(Operations):

    def __init__(self, name, directory, best_smoothing, non_cpgs, cpgs=None, collapse=True,
                 muts_dict_raw=None, occ_dict_raw=None, cpg_remove_percentage=0,
                 generations=None, prefix='', operations=None):

        Operations.__init__(self, name=name, directory=directory, collapse=collapse,
                            muts_dict_raw=muts_dict_raw, occ_dict_raw=occ_dict_raw,
                            cpg_remove_percentage=cpg_remove_percentage, prefix=prefix,
                            operations=operations)

        self.generations = generations
        if self.muts_dict_smoothed is None:
            self.smooth_dics(best_smoothing=best_smoothing)

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
        self.indices = []

        self.cpg_vector_mrkv_corrected = None
        self.chosen_non_cpg_vector_mrkv_corrected = None
        self.cpg_backward_vector_mrkv_corrected = None
        self.chosen_non_cpg_backward_vector_mrkv_corrected = None

        self.best_static = None
        self.best_bins = None
        self.CpG_remove_percentage = None
        self.no_recurrence = None

    def calculate_best_parameters(self):
        self.calculate_chosen_vectors()

        self.write_logs(f'len of cpg vector: {len(self.cpg_vector_original)}')

        try:
            best_static, best_bins, indeces = self.get_best_static_percentile()

        except ValueError as e:
            print(e)
            self.write_logs(f'No best static percentile found')
            self.no_recurrence = True
            raise ValueError('No best static percentile found')

        self.write_logs(f'best static percentile: {best_static}, best number of bins: {best_bins}')
        self.best_static = best_static
        self.best_bins = best_bins
        self.indices = indeces
        self.plot_boxes()

    def get_recurrence_vectors(self):

        if self.best_static is None and (not self.no_recurrence): self.calculate_best_parameters()
        if self.cpg_vector_mrkv_corrected is not None:
            return self.cpg_vector_mrkv_corrected, self.chosen_non_cpg_vector_mrkv_corrected, \
                   self.cpg_backward_vector_mrkv_corrected, self.chosen_non_cpg_backward_vector_mrkv_corrected

        if self.no_recurrence: return

        cpg_box, non_cpg_box, cpg_backward_box, chosen_non_cpg_backward_box, _indeces = self.get_boxes()
        cpg = [np.median(i) for i in cpg_box]
        non_cpg = [np.median(i) for i in non_cpg_box]
        cpg_backward = [np.median(i) for i in cpg_backward_box]
        chosen_non_cpg_backward = [np.median(i) for i in chosen_non_cpg_backward_box]

        if self.generations is not None:
            cpg= [i/self.generations for i in cpg]; non_cpg = [i/self.generations for i in non_cpg]
            cpg_backward = [i/self.generations for i in cpg_backward]; chosen_non_cpg_backward = [i/self.generations for i in chosen_non_cpg_backward]

        self.write_logs(f'cpg: {cpg}\nnon_cpg: {non_cpg}\ncpg_backward: {cpg_backward}\nchosen_non_cpg_backward: {chosen_non_cpg_backward}')

        return cpg, non_cpg, cpg_backward, chosen_non_cpg_backward

    def calculate_chosen_vectors(self):

        def get_the_right_mut(mut):
            if str(mut) not in self.mutations_pool: mut = mut.get_rev_comp()
            if str(mut) in self.mutations_pool: return str(mut)
            return None

        muts = deepcopy(self.muts_dict_smoothed); occs = deepcopy(self.occ_dict_smoothed)

        chosen_non_cpg_backward_muts = [get_the_right_mut(m.get_backwards()) for m in self.non_cpg_muts
                                        if get_the_right_mut(m.get_backwards()) is not None]

        cpg_backward_muts = [get_the_right_mut(m.get_backwards()) for m in self.cpg_muts
                             if get_the_right_mut(m.get_backwards()) is not None]

        if len(cpg_backward_muts) == 0 or len(chosen_non_cpg_backward_muts) == 0:
            self.write_logs('No backward mutations in the pool, assuming backward mutations = 0')
            self.cpg_vector_original, self.chosen_non_cpg_vector_original= es.condition_muts_all(
                                                                            muts, occs,
                                                                            [self.cpg_muts, self.non_cpg_muts],
                                                                            filter_low_muts=False,
                                                                            drop_small_bins=False
                                                                            )
            self.cpg_backward_vector_original = pd.Series(0, index=self.cpg_vector_original.index)
            self.chosen_non_cpg_backward_vector_original = pd.Series(0, index=self.chosen_non_cpg_vector_original.index)
            return

        self.cpg_vector_original, self.chosen_non_cpg_vector_original, \
        self.cpg_backward_vector_original,\
        self.chosen_non_cpg_backward_vector_original = es.condition_muts_all(
                                                     muts, occs,
                                                     [self.cpg_muts, self.non_cpg_muts,
                                                     cpg_backward_muts,
                                                     chosen_non_cpg_backward_muts],
                                                     filter_low_muts=False,
                                                     drop_small_bins=False
                                                     )

    def get_boxes(self, number_of_bins=None, static_percentile=None, remove_percentage=None, CpG_remove_percentage=None):
        if number_of_bins is None: number_of_bins = self.best_bins
        if static_percentile is None: static_percentile = self.best_static
        if remove_percentage is None: remove_percentage = self.remove_percentage
        if CpG_remove_percentage is None: CpG_remove_percentage = self.CpG_remove_percentage

        #copy vectors
        cpg_vector = deepcopy(self.cpg_vector_original)
        chosen_non_cpg_vector = deepcopy(self.chosen_non_cpg_vector_original)
        cpg_backward_vector = deepcopy(self.cpg_backward_vector_original)
        chosen_non_cpg_backward_vector = deepcopy(self.chosen_non_cpg_backward_vector_original)

        cpg_box = []; non_cpg_box = []; cpg_backward_box = []; chosen_non_cpg_backward_box = []

        #get to-remove indicies
        to_remove_noncpg_index = chosen_non_cpg_vector.sort_values(ascending=True).iloc[
                int(len(chosen_non_cpg_vector)*0): int(len(chosen_non_cpg_vector)*remove_percentage)].index
        to_remove_cpg_index = cpg_vector.sort_values(ascending=True).iloc[
                int(len(cpg_vector)*0): int(len(cpg_vector)*(remove_percentage))].index

        if CpG_remove_percentage is not None:
            extra_to_remove_cpg_index = cpg_vector.sort_values(ascending=True).iloc[
                int(len(cpg_vector)*0): int(len(cpg_vector)*(CpG_remove_percentage))].index
        else: extra_to_remove_cpg_index = []

        indices_to_drop = to_remove_noncpg_index.union(to_remove_cpg_index).union(extra_to_remove_cpg_index)

        #drop these indices from all vectors
        cpg_vector = cpg_vector.drop(indices_to_drop).reindex()
        chosen_non_cpg_vector = chosen_non_cpg_vector.drop(indices_to_drop).reindex()
        cpg_backward_vector = cpg_backward_vector.drop(indices_to_drop).reindex()
        chosen_non_cpg_backward_vector = chosen_non_cpg_backward_vector.drop(indices_to_drop).reindex()

        indices = []
        #pool common indices to bins
        for i, iplus1 in self._xrange_plus1(static_percentile, number_of_bins):

            non_cpg_index = chosen_non_cpg_vector.sort_values(ascending=True).iloc[
                int(len(chosen_non_cpg_vector)*i): int(len(chosen_non_cpg_vector)*(iplus1))].index


            cpg_index = cpg_vector.sort_values(ascending=True).iloc[
                int(len(cpg_vector)*i): int(len(cpg_vector)*(iplus1))].index

            common_index = cpg_index.intersection(non_cpg_index)

            cpg_box.append(cpg_vector[common_index])
            non_cpg_box.append(chosen_non_cpg_vector[common_index])
            cpg_backward_box.append(cpg_backward_vector[common_index])
            chosen_non_cpg_backward_box.append(chosen_non_cpg_backward_vector[common_index])
            indices.append(common_index)
        #print(sum([len(i) for i in indices]))
        return cpg_box, non_cpg_box, cpg_backward_box, chosen_non_cpg_backward_box, indices

    def get_best_static_percentile(self, remove_percentage=None, CpG_remove_percentage=None):

        if remove_percentage is None: remove_percentage = self.remove_percentage
        best_static = 0.0; step = 0.05; static = 0.15; best_indeces = None

        while static < 0.8:
            best_bins = None
            for bins in range(4, 5):

                cpg_box, non_cpg_box, _b, _c, indeces = self.get_boxes(number_of_bins=bins,
                                                              static_percentile=static,
                                                              remove_percentage=remove_percentage,
                                                              CpG_remove_percentage=CpG_remove_percentage)

                if not all([len(cpg_box[i]) > 100 for i in range(len(cpg_box))]):
                    break

                if self.biases_by_ratios(cpg_box, non_cpg_box):
                    self.indices = indeces

                    if self.generations is not None:
                        self.write_logs(f'check biases by ratios after markove correction for static percentile: {static} and bins: {bins}')
                        self.plot_boxes(cpg_box, non_cpg_box, f'check_biases_by_ratios_static_{static}_bins_{bins}')
                        c, nc, cb, ncb = self.get_mrkv_corrected_vctrs(generations=self.generations)
                        self.plot_boxes(c, nc, f'check_biases_by_ratios_static_{static}_bins_{bins}_mrkv_corrected')
                        self.export_arrays_before_after_mrkov(cpg_box, non_cpg_box,_b, _c, c, nc, cb, ncb, f'static_{static}& bins_{bins}')
                        if not self.biases_by_ratios(c, nc, True):
                            self.indeces = None
                            self.write_logs(f'check failed, trying biases_ok')
                            if not self.biases_ok(c, nc, True):
                                self.write_logs(f'check failed')
                                break
                            else:
                                self.write_logs(f'bias ok passed')
                                self.write_logs(f'cpg_vector= {c}\nnon_cpg_vector= {nc}\ncpg_backward_vector= {cb}\nnon_cpg_backward_vector= {ncb}')
                                self.write_logs('resuming pipeline')
                        else:
                            self.write_logs(f'checks passed')
                            #self.export_arrays_before_after_mrkov(cpg_box, non_cpg_box,_b, _c, c, nc, cb, ncb, f'static_{static}& bins_{bins}')

                            best_bins = bins
                            best_static = static
                            best_indeces = indeces
                    else:

                        best_bins = bins
                        best_static = static
                        best_indeces = indeces

            if best_bins is None:

                static += step
                static = round(static, 4)
                self.write_logs(f'increasing static percentile to {static}')
                continue
            else:

                return best_static, best_bins, best_indeces

        raise ValueError('No best static percentile found')

    def biases_by_ratios(self, cpg_box, non_cpg_box, medians=False):
        if not medians:
            cpg_box = [np.median(cpg_box[i]) for i in range(len(cpg_box))]
            non_cpg_box = [np.median(non_cpg_box[i]) for i in range(len(non_cpg_box))]

        biases = [non_cpg_box[i]/cpg_box[i] for i in range(len(cpg_box))]
        #check that biaseses are sorted and increasing
        return all([biases[i] <= biases[i+1] for i in range(len(biases)-1)])

    def biases_ok(self, cpg_box, non_cpg_box, medians=False):
        if not medians:
            cpg_box = [np.median(cpg_box[i]) for i in range(len(cpg_box))]
            non_cpg_box = [np.median(non_cpg_box[i]) for i in range(len(non_cpg_box))]
        biases = [(non_cpg_box[i]/non_cpg_box[0]) - (cpg_box[i]/cpg_box[0])
                    for i in range(len(cpg_box))]


        return all([i >= 1000 for i in biases])

    def plot_boxes(self, cpg_box=None, non_cpg_box=None, name='boxes_plot'):

        plt.clf()
        if cpg_box is None or non_cpg_box is None:
            cpg_box, non_cpg_box, _b, _c, _indeces = self.get_boxes()

        cpg_box = [i/np.median(cpg_box[0]) for i in cpg_box]
        non_cpg_box = [i/np.median(non_cpg_box[0]) for i in non_cpg_box]

        width = 0.45

        ticks = range(len(cpg_box))
        fig, ax = plt.subplots(figsize=(10,10))
        if isinstance(cpg_box[0], float):
            cpg_box = [[i] for i in cpg_box]
            non_cpg_box = [[i] for i in non_cpg_box]



        ax.boxplot(cpg_box, positions=ticks, widths=width, showfliers=False, patch_artist=True, boxprops=dict(facecolor='r', color='r'))
        ax.boxplot(non_cpg_box, positions=[i+width/2 for i in ticks], widths=width, showfliers=False, patch_artist=True, boxprops=dict(facecolor='b', color='b'))
        #add the means
        ax.plot(ticks, [np.median(i) for i in cpg_box], 'ro')
        ax.plot([i+width/2 for i in ticks], [np.median(i) for i in non_cpg_box], 'bo')



        #xticks values are the normalized mu values, their positions are the window positions
        ax.set_xticks(ticks)
        #ax.set_xticklabels([np.average(i)/cpg_mu[0] for i in cpg_mu], rotation=90)
        #put the number of bins in each box
        for i in ticks:
            ax.text(i, 0.5, str(len(cpg_box[i])), color='r', rotation=90)

        ax.set_xlabel('pooled genomic windows sorted by their mutation rate')
        ax.set_ylabel('mu/mu0')
        #show legend
        #ax.legend(['CpG transitions', 'Non-CpG mutations'], fontsize='x-large')

        plt.savefig(f'{self.prefix}{name}.png')

    def _xrange(self, st, n):
        if st != 0: return [0] + list(np.linspace(st, 1, n+1)[:-1])
        else: return list(np.linspace(0, 1, n+1)[:-1])

    def _xrange_plus1(self, st, n):
        xrange = list(self._xrange(st,n)) + [1]
        result = []
        inner_xrange = list(self._xrange(st, n))
        for i in inner_xrange:
            result.append((i, xrange[xrange.index(i)+1]))
        return result

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
           on the indeces
        '''
        muts = deepcopy(self.muts_dict_smoothed); occs = deepcopy(self.occ_dict_smoothed)
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

            final_state = pd.DataFrame(0,index=initial_state.index, columns=initial_state.index)
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
        '''
        def get_weighted_mut(mutations):
            weights = []; values = []
            totalocc = 0; totalmut = 0
            for mut in mutations:
                if str(mut) not in weights_matrix.index: mut = mut.get_rev_comp()
                if mut.get_backwards().tri not in best_guess[mut.tri]:
                    back_tri = es.get_rev_comp(mut.get_backwards().tri)
                else: back_tri = mut.get_backwards().tri
                value = best_guess.loc[mut.tri][back_tri]
                if mut.get_rev_comp()== mut.get_backwards(): continue
                values.append(value)
                weights.append(weights_matrix[str(mut)])

                totalocc += occs_matrix[str(mut)]
                totalmut += value*occs_matrix[str(mut)]
            mutability = totalmut/totalocc
            return mutability
            return np.average(values, weights=weights)

        cpg_values = get_weighted_mut(self.cpg_muts)
        non_cpg_values = get_weighted_mut(self.non_cpg_muts)
        cpg_backward_values = get_weighted_mut([m.get_backwards() for m in self.cpg_muts])
        non_cpg_backward_values = get_weighted_mut([m.get_backwards() for m in self.non_cpg_muts])

        return cpg_values, non_cpg_values, cpg_backward_values, non_cpg_backward_values

    def get_mrkv_corrected_vctrs(self, generations):
        '''returns the best transition matrix for each group
        '''
        muts_matrices, occs_matrices = self.get_matrices()
        groups = self.get_markov_input_matrices(muts_matrices, occs_matrices)
        markov = MarkovInfSites(generations, self.prefix, self)
        orignial_prefix = self.prefix
        cpg_vector = []; non_cpg_vector = []; cpg_backward_vector = []; non_cpg_backward_vector = []
        for i, group in enumerate(groups):
            #self.write_logs(f'Calculating markov correction for group {i}')
            #make a new directory for each group
            markov.prefix = f'{orignial_prefix}/group_{i}/'
            if not os.path.exists(markov.prefix): os.mkdir(markov.prefix)
            markov.logs_f = markov.prefix + 'log.txt'
            group['inf_sites_guess'].to_csv(markov.prefix + 'observed_sub_matrix.csv', sep='\t')
            inf_sites_guess = group['inf_sites_guess']/self.generations
            inf_sites_guess.to_csv(markov.prefix + 'inf_sites_guess.csv', sep='\t')
            c, nc, cb, cnb = self.get_vector_item_from_best_guess(markov.get_best_guess(group['initial_state'],
                                                                                        group['final_state'],
                                                                                        group['inf_sites_guess']),
                                                                 muts_matrices[i]/occs_matrices[i],
                                                                 occs_matrices[i], muts_matrices[i])
            cpg_vector.append(c); non_cpg_vector.append(nc)
            cpg_backward_vector.append(cb); non_cpg_backward_vector.append(cnb)

        print(f'cpg_vector: {cpg_vector}\nnon_cpg_vector: {non_cpg_vector}\ncpg_backward_vector: {cpg_backward_vector}\nnon_cpg_backward_vector: {non_cpg_backward_vector}')

        return cpg_vector, non_cpg_vector, cpg_backward_vector, non_cpg_backward_vector
