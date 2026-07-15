import sys
import time
import numpy as np
from . import essentials as es
import pandas as pd
import cupy
from scipy.linalg import fractional_matrix_power
import matplotlib.pyplot as plt
from .base_operations import Operations
import warnings
warnings.filterwarnings('ignore')

class MarkovInfSites(Operations):

    def __init__(self, generations, prefix='', operations:Operations=None):

        if operations is not None: Operations.__init__(self, operations=operations)

        self.generations = generations
        self.prefix = prefix
        self.contexts = None#list(sorted(set([m.tri for m in es.get_muts_sig_ordered()])))
        self.final_state = None
        self.initial_state = None


    def get_best_guess(self, initial_state:pd.Series, final_state:pd.DataFrame, inf_sites_guess:pd.DataFrame):
        plt.clf()
        initial_state_cp, final_state_cp, inf_sites_guess_cp = self.get_cupy_matrices(initial_state,
                                                                                      final_state,
                                                                                      inf_sites_guess)
        #return pd.DataFrame(inf_sites_guess_cp.get(), columns=self.contexts, index=self.contexts)

        self.final_state = final_state_cp
        self.initial_state = initial_state_cp

        alpha = 0.9
        max_loop = 1100
        error = cupy.array([0.0]*max_loop)
        guessed_matrix = cupy.copy(inf_sites_guess_cp)
        start = time.time()
        #inner_error = []
        analyzed=False
        for i in range(max_loop):
            error_mat, errored_bases = self.get_error(initial_state_cp, guessed_matrix, final_state_cp)
            error[i] = errored_bases
            #inner_error.append(float(errored_bases))

            if i == 0: self.export(guessed_matrix, i, error)

            max_errors, max_error_poss = self.max_errors_and_poss(error_mat)

            studied = 0
            for j, max_error_pos in enumerate(zip(max_error_poss[0], max_error_poss[1])):

                if max_error_pos[0] == max_error_pos[1]: continue
                if guessed_matrix[max_error_pos] == 0: continue


                analyzed = False
                studied += 1
                if studied > 32: break

                for ratio in 1+alpha, 1-alpha:
                    try_matrix = self.get_try_cpmatrix(guessed_matrix, max_error_pos, ratio)
                    new_error_cpmat, _ = self.get_error(initial_state_cp, try_matrix, final_state_cp)
                    new_error_cpmat = cupy.absolute(new_error_cpmat)
                    if cupy.sum(new_error_cpmat) < cupy.sum(error_mat):

                        guessed_matrix = try_matrix

                        best_in_df = guessed_matrix
                        analyzed = True
                        break
                if analyzed: break

            if not analyzed:
                alpha *= 0.5
                if not self.silent:
                    self.write_logs(f'\talpha reduced to {alpha}')
                #inner_error = []

            '''if len(inner_error) > 2:

                delta_current_error = float(abs(inner_error[-1] - inner_error[-2]))
                delta_previous_error = float(abs(error[-2] - error[i-3]))
                if delta_previous_error != 0 and (abs(delta_current_error-delta_previous_error)/delta_previous_error<0.75):
                    alpha *= 0.5
                    self.write_logs(f'\talpha reduced to {alpha}')
                    inner_error = []'''

            #if i % 10 == 0: print(f'---{str(i)}--- iterations', end='\r')
            if i % 20 == 0 and i != 0:
                self.export(best_in_df, i, error)
                if not self.silent:
                    self.write_logs('\n')
                    self.write_logs(f'\talpah: {alpha}')

            if alpha < 1e-15 or error[i] <= 0.5:
                self.export(best_in_df, i, error)
                if not self.silent:
                    self.write_logs('\tconverged')
                    self.write_logs(f'\tit took {i} iterations')
                break
            if i == max_loop-1:
                self.export(best_in_df, i, error)
                if not self.silent:
                    self.write_logs('\tmax loop reached')
                    self.write_logs(f'\tit took {i} iterations')

        #print('**************')
        #print(time.time()-start)
        #print('**************')
        best_in_df_pd = pd.DataFrame(best_in_df.get(), columns=self.contexts, index=self.contexts)

        return best_in_df_pd

    def get_cupy_matrices(self, initial_state:pd.Series, final_state:pd.DataFrame, inf_sites_guess:pd.DataFrame):
        inf_sites_guess_extended = pd.DataFrame(0.0, index=initial_state.index, columns=initial_state.index)
        self.contexts = list(final_state.index)
        min_mut = 1.0
        for base in inf_sites_guess.columns:
            for source in inf_sites_guess.index:
                target = source[0] + base[0] + source[-1]
                if target not in inf_sites_guess.index: target = es.get_rev_comp(target)
                inf_sites_guess_extended.loc[source, target] = inf_sites_guess.loc[source, base]#/self.generations
                if inf_sites_guess.loc[source, base] < min_mut: min_mut = inf_sites_guess.loc[source, base]/self.generations

        #fill_diagonal
        for i in range(inf_sites_guess_extended.shape[0]):
            inf_sites_guess_extended.iloc[i, i] = 1 - inf_sites_guess_extended.iloc[i].sum()

        inf_sites_guess_extended = pd.DataFrame(fractional_matrix_power(inf_sites_guess_extended, 1/self.generations),
                                                index=inf_sites_guess_extended.index, columns=inf_sites_guess_extended.columns)
        #make any value less than minimum mut == 0:
        inf_sites_guess_extended[inf_sites_guess_extended < 10**-15] = 0

        for i in range(inf_sites_guess_extended.shape[0]):
            inf_sites_guess_extended.iloc[i, i] = 0
            inf_sites_guess_extended.iloc[i, i] = 1 - inf_sites_guess_extended.iloc[i].sum()

        inf_sites_guess_extended.to_csv('bla.csv', sep='\t')
        inf_sites_guess_extended = cupy.array(inf_sites_guess_extended, dtype=np.double)
        initial_state = cupy.array(initial_state, dtype=np.double)
        final_state = cupy.array(final_state, dtype=np.double)


        return initial_state.T, final_state, inf_sites_guess_extended

    @staticmethod
    def matrix_power(matrix, power):
        arr_cupy = cupy.copy(matrix)
        for i in range(power-1):
            arr_cupy = cupy.dot(arr_cupy, matrix)
        return arr_cupy

    def get_error(self, q0, trasition_cpmatrix, observed_cpmatrix):

        #arr_cupy = cupy.copy(cupy.transpose(trasition_cpmatrix))
        arr_cupy = cupy.copy(trasition_cpmatrix)
        p_n = cupy.linalg.matrix_power(arr_cupy, self.generations)
        #p_n = matrix_power(arr_cupy, self.generations)
        #print(all(np.round(p_n.get().flatten(), 7) == np.round(cupy.linalg.matrix_power(arr_cupy, generations).get().flatten(),7)))
        #assert all(np.round(p_n.get().flatten(), 7) == np.round(cupy.linalg.matrix_power(arr_cupy, generations).get().flatten(),7))

        qn = cupy.array([[0]*32]*32 , dtype=np.double)

        for c in self.contexts:
            q0_c = cupy.array([0]*32, dtype=np.double)
            q0_c[self.contexts.index(c)] = q0[self.contexts.index(c)]
            qn[self.contexts.index(c)] += cupy.dot(q0_c, p_n)

        #qn = cupy.transpose(qn)
        error_bases = cupy.subtract(qn, observed_cpmatrix)
        error_bases = cupy.absolute(error_bases)

        return error_bases, error_bases.sum().sum()

    @staticmethod
    def max_error_and_pos(error_cpmat):
        max_value = cupy.max(error_cpmat)
        max_pos = cupy.unravel_index(cupy.argmax(error_cpmat), error_cpmat.shape)
        #max_pos = cupy.argmax(error_cpmat)
        return max_value, max_pos

    @staticmethod
    def max_errors_and_poss(error_cpmat):
        flattened = error_cpmat.flatten()
        max_values = cupy.sort(flattened)[::-1]
        max_poss = cupy.argsort(flattened)[::-1]
        max_poss = cupy.unravel_index(max_poss, error_cpmat.shape)

        return max_values, max_poss

    @staticmethod
    def get_try_cpmatrix(guessed_cpmat, max_error_pos, ratio):

        try_cpmat = cupy.copy(guessed_cpmat)
        x = [(try_cpmat[i, i]) for i in range(32)]

        try_cpmat[max_error_pos] *= (ratio)
        try_cpmat[max_error_pos[0], max_error_pos[0]] = 0
        try_cpmat[max_error_pos[0], max_error_pos[0]] = 1 - cupy.sum(try_cpmat[max_error_pos[0]])
        #try_cpmat[max_error_pos[0]] = cupy.divide(try_cpmat[max_error_pos[0]], cupy.sum(try_cpmat[max_error_pos[0]]))
        return try_cpmat

    @staticmethod
    def series_tri_base_to_matrix(series):
        muts = [es.mutation(label=m) for m in series.index]
        labels = sorted(set(m.tri for m in muts))
        columns = sorted(set(m.base for m in muts))
        matrix = pd.DataFrame(0.0, index=labels, columns=columns)

        for mutation in muts:
            matrix.loc[mutation.tri, mutation.base] = series[mutation.label]

        return matrix


    def export(self, best_in_df, i, error):

        best_in_df_cp = cupy.copy(best_in_df.T)
        #best_in_df_cp = cupy.copy(best_in_df)
        best_in_df_cp = pd.DataFrame(best_in_df_cp.get(), columns=self.contexts, index=self.contexts).T
        best_in_df_cp.to_csv(f'{self.prefix}/best_in_df.csv', sep='\t')

        p_n = cupy.linalg.matrix_power(best_in_df, self.generations)
        #best_overall = (matrix_power(best_in_df, generations)).get()
        best_overall = pd.DataFrame(p_n.get(), columns=self.contexts, index=self.contexts)
        best_overall.to_csv(f'{self.prefix}/best_overallT.csv', sep='\t')

        observed_matrix = self.final_state.get()
        if not self.silent:
            self.write_logs(f'\terror abs: {abs(error[i])}')
            self.write_logs(f'\terror percentage: {abs(error[i]/observed_matrix.sum().sum())*100}%')


        plt.plot(error.get()[:i]/observed_matrix.sum().sum())
        plt.title(f'loop iteration: {i}')
        plt.savefig(f'{self.prefix}/error.png')

        q0 = self.initial_state
        qn = cupy.array([[0]*32]*32 , dtype=np.double)
        for c in self.contexts:
            q0_c = cupy.array([0]*32, dtype=np.double)
            q0_c[self.contexts.index(c)] = q0[self.contexts.index(c)]
            qn[self.contexts.index(c)] += cupy.dot(q0_c, p_n)


        best_overall = pd.DataFrame(qn.get(), columns=self.contexts, index=self.contexts)

        best_overall.to_csv(f'{self.prefix}/best_in_df_result.csv', sep='\t')
