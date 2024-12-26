from statistics import mean
from matplotlib import pyplot as plt
import numpy as np
import warnings
from .predictor import Predictor
from .graphpoint import GraphPointAbstract as GPA
from ..base_operations import Operations
import traceback
warnings.filterwarnings("ignore")
from joblib import Parallel, delayed

class PopSizeCalculator(Operations):

    def __init__(self, gens, cpg_subs, non_cpg_subs, cpg_subs_bckwrds, non_cpg_subs_bckwrds, directory='', operations=None):
        Operations.__init__(self, operations=operations)

        self.gens = gens
        self.cpg_subs = cpg_subs
        self.non_cpg_subs = non_cpg_subs
        self.cpg_subs_bckwrds = cpg_subs_bckwrds
        self.non_cpg_subs_bckwrds = non_cpg_subs_bckwrds
        self.best_pop = None
        self.min_error = None
        self.computed_points = [[], []] #holds pop vs error
        self.directory = directory

        self.predictor = Predictor()
        #self.subs_to_subs_per_gen()

    def subs_to_subs_per_gen(self):
        self.cpg_subs = [i/self.gens for i in self.cpg_subs]
        self.non_cpg_subs = [i/self.gens for i in self.non_cpg_subs]
        self.cpg_subs_bckwrds = [i/self.gens for i in self.cpg_subs_bckwrds]
        self.non_cpg_subs_bckwrds = [i/self.gens for i in self.non_cpg_subs_bckwrds]

    def base_error(self):

        cpg_ratios = [i/self.cpg_subs[0] for i in self.cpg_subs]
        non_cpg_ratios = [i/self.non_cpg_subs[0] for i in self.non_cpg_subs]
        error = sum([abs(cpg_ratios[i] - non_cpg_ratios[i]) for i in range(len(cpg_ratios))])
        error = abs(error/(len(cpg_ratios)-1))

        return error

    def get_muts(self, pop):

        # Parallelize the operation
        def parallel_predictor(subf, b, pop):
            try:
                return self.predictor.get_mu(GPA(0, b, int(pop), mean_sub_rate=subf))
            except Exception as e:
                print(traceback.format_exc())
                try: return self.predictor.get_mu(GPA(0, None, int(pop), mean_sub_rate=subf))
                except Exception as e:
                    print(traceback.format_exc())
                    print(f'Error: {e} for pop: {pop}, subf: {subf}, b: {b}')
                    return subf

        cpg_muts = Parallel(n_jobs=-1, verbose=0)(
            delayed(parallel_predictor)(subf, subb, pop) for subf, subb in zip(self.cpg_subs, self.cpg_subs_bckwrds)
        )

        non_cpg_muts = Parallel(n_jobs=-1, verbose=0)(
            delayed(parallel_predictor)(subf, subb, pop) for subf, subb in zip(self.non_cpg_subs, self.non_cpg_subs_bckwrds)
        )

        return cpg_muts, non_cpg_muts

    def get_error(self, pop):

        cpg_muts, non_cpg_muts = self.get_muts(pop)
        ratio_of_means = mean(cpg_muts)/mean(non_cpg_muts)

        ratios = []
        for i in range(len(cpg_muts)):
            ratios.append(abs((cpg_muts[i]/non_cpg_muts[i])-ratio_of_means))
        error = sum(ratios)/len(ratios)
        #error = sum([abs(ratios[i] - ratios[i-1]) for i in range(1, len(ratios))])/len(ratios)
        print(f'{pop}:{error}')
        return error

    def calc_best_pop(self):
        old_xs = self.computed_points[0]; old_ys = self.computed_points[1]
        best_pop, self.computed_points = self.predictor.minimize_by_search(self.get_error, [4, 8000000],
                                                     tol=10**-4, negative=False, breadth=8,
                                                     first_breadth=32, max_depth=25, log=False,
                                                     parallel=True, precesion=5000, old_xs=old_xs,
                                                     old_ys=old_ys)
        error = self.get_error(best_pop)
        print(f'best pop: {best_pop}, error: {error}')
        self.best_pop = best_pop
        self.min_error = error

    def _plot_bins(self, subs_1, subs_2, muts_1, muts_2, name):
        '''plot the bins'''
        plt.clf()
        fig, (ax1, ax2) = plt.subplots(1, 2)

        index = np.arange(len(subs_1))
        bar_width = 0.35

        rects1 = ax1.bar(index, subs_1, bar_width, color='r', label='CpG_subs')
        rects2 = ax1.bar(index + bar_width, subs_2, bar_width,
                        color='b', label='nonCpG_subs')

        #ax1.set_ylabel('Substitution Rate')
        ax1.set_title('Normalised substitution Rates', loc='left')
        ax1.legend()

        index = np.arange(len(muts_1))
        bar_width = 0.35

        rects1 = ax2.bar(index, muts_1, bar_width, color='r', label='CpG_muts')
        rects2 = ax2.bar(index + bar_width, muts_2, bar_width,
                        color='b', label='nonCpG_muts')

        ax2.set_title('Normalised mutation Rates', loc='left')
        ax2.legend()

        plt.savefig(name)

    def plot_correction(self, best_pop=None):

        if best_pop is None: best_pop = self.best_pop

        cpg_muts, non_cpg_muts = self.get_muts(best_pop)

        cpg_subs = np.array(self.cpg_subs); non_cpg_subs = np.array(self.non_cpg_subs)
        cpg_muts = np.array(cpg_muts); non_cpg_muts = np.array(non_cpg_muts)

        cpg_subs_by_mean = cpg_subs/mean(cpg_subs); non_cpg_subs_by_mean = non_cpg_subs/mean(non_cpg_subs)
        cpg_muts_by_mean = cpg_muts/mean(cpg_muts); non_cpg_muts_by_mean = non_cpg_muts/mean(non_cpg_muts)

        self._plot_bins(cpg_subs_by_mean, non_cpg_subs_by_mean, cpg_muts_by_mean, non_cpg_muts_by_mean, f'{self.directory}/best_pop_by_mean.png')

        cpg_subs_by_zero = cpg_subs/cpg_subs[0]; non_cpg_subs_by_zero = non_cpg_subs/non_cpg_subs[0]
        cpg_muts_by_zero = cpg_muts/cpg_muts[0]; non_cpg_muts_by_zero = non_cpg_muts/non_cpg_muts[0]

        self._plot_bins(cpg_subs_by_zero, non_cpg_subs_by_zero, cpg_muts_by_zero, non_cpg_muts_by_zero, f'{self.directory}/best_pop_by_zero.png')

        perc_change_cpg = [(cpg_muts[i]-cpg_subs[i])/cpg_subs[i] for i in range(len(cpg_subs))]
        perc_change_noncpg = [(non_cpg_muts[i]-non_cpg_subs[i])/non_cpg_subs[i] for i in range(len(non_cpg_subs))]

        self.write_logs(f'CpG subs: {cpg_subs}')
        self.write_logs(f'CpG muts: {cpg_muts}')
        self.write_logs(f'perc change CpG: {perc_change_cpg}')
        self.write_logs(f'non CpG subs: {non_cpg_subs}')
        self.write_logs(f'non CpG muts: {non_cpg_muts}')
        self.write_logs(f'perc change nonCpG: {perc_change_noncpg}')


    def plot_error_points(self):
        '''scattter plot of computed points'''
        #clear plt
        plt.clf()
        plt.scatter(self.computed_points[0], self.computed_points[1])
        plt.xlabel('Population Size')
        plt.ylabel('Error')
        plt.savefig(f'{self.directory}/error_points.png')

        try:

            plt.clf()
            plt.scatter(self.computed_points[0], np.log10(np.array(self.computed_points[1])))
            plt.xlabel('Population Size')
            plt.ylabel('Error log10')
            plt.savefig(f'{self.directory}/error_points_log.png')
        except Exception as e:
            print(f'Error in log plot: {e}')

