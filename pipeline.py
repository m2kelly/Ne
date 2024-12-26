import numpy as np
from .pop_size import PopSizeCalculator
from .base_operations import Operations
from .best_smoothing import BestSmoothing
from .best_non_cpg_candidates import BestNonCpGCandidates
from .recurrence_vectors import ReccurenceVectors
from . import essentials as es
from .noncpgs import get_h_non_cpgs_and_l_non_cpgs
import os
import shutil


class Pipeline(BestSmoothing, BestNonCpGCandidates, ReccurenceVectors, Operations):

    def __init__(self, name, directory, cpgs=None, non_cpg_pool=None, smoothing_range=None, collapse=True, generations=None ,operations=None):

        Operations.__init__(self, name=name, directory=directory, collapse=collapse, operations=operations)

        self.cpgs = cpgs
        self.non_cpg_pool = non_cpg_pool
        self.smoothing_range = smoothing_range
        self.best_window = None
        self.best_candidates = None

        self.cpg_subs = None
        self.non_cpg_subs = None
        self.cpg_subs_bckwrds = None
        self.non_cpg_subs_bckwrds = None

        self.generations = generations

        self.clear_logs()

    def get_best_smoothing_window(self):
        self.write_logs('Getting best smoothing window')

        BestSmoothing.__init__(self, name=self.name, directory=self.directory, cpgs=self.cpgs,
                               smoothing_range=self.smoothing_range, collapse=self.collapse,
                               muts_dict_raw=self.muts_dict_raw, occ_dict_raw=self.occ_dict_raw,
                               cpg_remove_percentage=self.cpg_remove_percentage, prefix=self.prefix,
                               operations=self)

        BestSmoothing.get_best_smoothing_window(self)
        self.write_logs(f'Best smoothing window:{self.best_window}')

    def get_best_non_cpg_candidates(self):
        if self.best_window is None:
            self.get_best_smoothing_window()
        self.write_logs('Getting best non cpg candidates')

        BestNonCpGCandidates.__init__(self, name=self.name, directory=self.directory,
                                      best_smoothing=self.best_window,
                                      cpgs=self.cpgs, non_cpg_pool=self.non_cpg_pool,
                                      collapse=self.collapse,
                                      muts_dict_raw=self.muts_dict_raw,
                                      occ_dict_raw=self.occ_dict_raw,
                                      cpg_remove_percentage=self.cpg_remove_percentage,
                                      prefix=self.prefix, operations=self)
        best_candidates = BestNonCpGCandidates.get_best_candidates(self)
        self.best_candidates = best_candidates
        return best_candidates

    def get_recurrence_vectors(self):
        if self.best_candidates is None:
            self.best_candidates = self.get_best_non_cpg_candidates()
        self.write_logs('Getting recurrence vectors')

        ReccurenceVectors.__init__(self, self.name, self.directory,
                                   best_smoothing=self.best_window,
                                   non_cpgs=self.best_candidates, cpgs=self.cpgs,
                                   collapse=self.collapse,
                                   muts_dict_raw=self.muts_dict_raw,
                                   occ_dict_raw=self.occ_dict_raw,
                                   cpg_remove_percentage=self.cpg_remove_percentage,
                                   generations=self.generations,
                                   prefix=self.prefix,
                                   operations=self)

        c, nc, cb, ncb = ReccurenceVectors.get_recurrence_vectors(self)
        self.cpg_subs = c; self.non_cpg_subs = nc
        self.cpg_subs_bckwrds = cb; self.non_cpg_subs_bckwrds = ncb
        self.write_logs(f'cpg_subs:{c}\nnon_cpg_subs:{nc}\ncpg_subs_bckwrds:{cb}\nnon_cpg_subs_bckwrds:{ncb}')
        return c, nc, cb, ncb

    def get_best_pop(self):
        if self.cpg_subs is None or self.non_cpg_subs is None or \
        self.cpg_subs_bckwrds is None or self.non_cpg_subs_bckwrds is None:
            self.write_logs('recurrence vectors not calculated')
            self.get_recurrence_vectors()
        self.write_logs('Calculating best population size')

        popcalc = PopSizeCalculator(gens=self.generations, cpg_subs=self.cpg_subs,
                         non_cpg_subs=self.non_cpg_subs, cpg_subs_bckwrds=self.cpg_subs_bckwrds,
                         non_cpg_subs_bckwrds=self.non_cpg_subs_bckwrds,
                         directory=self.prefix, operations=self)
        popcalc.calc_best_pop()
        self.write_logs(f'Best population size:{popcalc.best_pop}')
        popcalc.plot_correction()
        popcalc.plot_error_points()
        self.write_logs(f'pop_error values:{popcalc.computed_points}')

        return popcalc.best_pop, popcalc.min_error

    def run_with_exports(self, CpG_remove_percentage=0, continue_remove=False, filterboth=False, remove_step=0.05, filter_small_bins=False):

        analysed = False
        if self.non_cpg_pool is None:
            self.non_cpg_pool = es.get_general_non_cpg_pool()

        while CpG_remove_percentage<=0.75:
            try:
                directory = f'cpg_remove_percentage_{CpG_remove_percentage}/'
                self.prefix = directory
                try: os.mkdir(directory)
                except FileExistsError: pass

                self.cpg_remove_percentage = CpG_remove_percentage
                if not filterboth: self.filter_low_mut_CpGs()
                else: self.filter_low_both(filter_small_bins)
                self.get_best_smoothing_window()
                self.plot_smoothing_range().savefig(f'{self.prefix}smoothing_range_plot.png')
                self.plot_smoothing_diff().savefig(f'{self.prefix}smoothing_diff_plot.png')
                self.plot_max_corr_heatmap().savefig(f'{self.prefix}max_corr_heatmap.png')
                self.get_best_non_cpg_candidates()
                self.get_recurrence_vectors()
                best_pop, min_error = self.get_best_pop()
                self.write_logs(f'best pop:{best_pop}, min error:{min_error}')
                #copy logs to output directory
                shutil.copy(self.logs_f, f'{self.prefix}/{self.logs_f}')
                analysed = True
                if not continue_remove: break

                CpG_remove_percentage += remove_step
                self.write_logs(f'increasing CpG remove percentage to {CpG_remove_percentage}')

            except ValueError as e:
                if str(e) == 'No best static percentile found':
                    CpG_remove_percentage += remove_step
                    self.write_logs(f'increasing CpG remove percentage to {CpG_remove_percentage}')
                else: raise e
        if not analysed:
            self.write_logs('No best static percentile found for any CpG remove percentage')

        return

    def run_until_min_error(self, CpG_remove_percentage=0, zero_back=False, step=0.05):

        analysed = False
        min_error_reached = np.inf; best_pop_computed = None; best_remove_perc = None
        if self.non_cpg_pool is None:
            self.non_cpg_pool = es.get_general_non_cpg_pool()
        while CpG_remove_percentage<=0.75:
            try:
                directory = f'cpg_remove_percentage_{CpG_remove_percentage}/'
                self.prefix = directory
                try: os.mkdir(directory)
                except FileExistsError: pass

                self.cpg_remove_percentage = CpG_remove_percentage
                self.filter_low_both(True)
                self.get_best_smoothing_window()
                self.plot_smoothing_range().savefig(f'{self.prefix}smoothing_range_plot.png')
                self.plot_smoothing_diff().savefig(f'{self.prefix}smoothing_diff_plot.png')
                self.plot_max_corr_heatmap().savefig(f'{self.prefix}max_corr_heatmap.png')
                self.get_best_non_cpg_candidates()
                self.get_recurrence_vectors()
                if zero_back:
                    self.cpg_subs_bckwrds = [0]*len(self.cpg_subs)
                    self.non_cpg_subs_bckwrds = [0]*len(self.non_cpg_subs)
                best_pop, min_error = self.get_best_pop()
                #copy logs to output directory
                shutil.copy(self.logs_f, f'{self.prefix}/{self.logs_f}')
                analysed = True
                if min_error < min_error_reached:
                    min_error_reached = min_error
                    best_pop_computed = best_pop
                    best_remove_perc = CpG_remove_percentage
                    self.write_logs(f'new min error reached:{min_error_reached}')
                    CpG_remove_percentage += step
                    self.write_logs(f'increasing CpG remove percentage to {CpG_remove_percentage}')

                elif min_error < min_error_reached*3:
                    self.write_logs(f'no new min error reached, but error in range')
                    CpG_remove_percentage += step
                    self.write_logs(f'increasing CpG remove percentage to {CpG_remove_percentage}')
                else:
                    self.write_logs(f'no new min error reached')
                    self.write_logs(f'best pop:{best_pop_computed}, min error:{min_error_reached}, best remove%: {best_remove_perc}')
                    CpG_remove_percentage += step
                    break

            except ValueError as e:
                if str(e) == 'No best static percentile found':
                    if analysed ==True: return
                    CpG_remove_percentage += step
                    self.write_logs(f'increasing CpG remove percentage to {CpG_remove_percentage}')
                else: raise e
        if not analysed:
            self.write_logs('No best static percentile found for any CpG remove percentage')

        return

    def run_non_cpg_pipeline(self):
        self.write_logs('Running non cpg pipeline')
        cpgs, non_cpgs = get_h_non_cpgs_and_l_non_cpgs(self.muts_dict_raw, self.occ_dict_raw)
        self.write_logs(f'h_monos:{cpgs}\nl_monos:{non_cpgs}')
        self.cpgs = cpgs
        self.non_cpg_pool = non_cpgs
        self.run_with_exports()

    def run_mono_nuc_pipeline(self, ignored_pairs=[], skip_cpgs=True, len_h=None, len_l=None, CpG_remove_percentage=0, function=None, **kwargs):
        self.write_logs('Running mono nuc pipeline')
        if function is None: function = self.run_with_exports
        if ignored_pairs: self.write_logs(f'Ignoring pairs:{ignored_pairs}')
        cpgs, non_cpgs = self.get_h_mono_and_l_mono(ignored_pairs=ignored_pairs, skip_cpgs=skip_cpgs,
                                                    len_h=len_h, len_l=len_l)

        self.cpgs = cpgs
        self.non_cpg_pool = non_cpgs
        function(CpG_remove_percentage=CpG_remove_percentage, **kwargs)

    def run_test_pipeline(self):

        CpG_remove_percentage = 0

        while CpG_remove_percentage<0.6:
            try:
                directory = f'cpg_remove_percentage_{CpG_remove_percentage}/'
                self.prefix = directory
                try: os.mkdir(directory)
                except FileExistsError: pass

                self.cpg_remove_percentage = CpG_remove_percentage
                self.filter_low_mut_CpGs()
                #self.get_best_smoothing_window()
                #self.plot_smoothing_range().savefig(f'{self.prefix}smoothing_range_plot.png')
                #self.plot_smoothing_diff().savefig(f'{self.prefix}smoothing_diff_plot.png')
                #self.plot_max_corr_heatmap().savefig(f'{self.prefix}max_corr_heatmap.png')
                self.best_window = 5
                #self.get_best_non_cpg_candidates()
                self.best_candidates = [es.mutation(label=l) for l in ['CTG->A','ATT->A','CTT->A','TTT->A','TTA->A','CTC->A','CTA->A','GTT->A','TTG->A','GTG->A','GTA->A','TTC->A','GTC->A','ATC->G']]
                self.get_recurrence_vectors()
                self.get_best_pop()
                exit()
            except ValueError as e:
                if str(e) == 'No best static percentile found':
                    CpG_remove_percentage += 0.05
                    self.write_logs(f'increasing CpG remove percentage to {CpG_remove_percentage}')
                else: raise e

        self.write_logs('No best static percentile found for any CpG remove percentage')

    def bootstrap(self, CpG_remove_percentage, replicates=100, smoothing_window=None, best_candidates=None):

        analysed = False
        if self.non_cpg_pool is None:
            self.non_cpg_pool = es.get_general_non_cpg_pool()

            directory = f'cpg_remove_percentage_{CpG_remove_percentage}/'

            self.prefix = directory
            try: os.mkdir(directory)
            except FileExistsError: pass

            self.cpg_remove_percentage = CpG_remove_percentage
            self.filter_low_both(True)
            self.best_window = smoothing_window
            self.best_candidates = best_candidates
            self.smooth_dics()

            for i in range(replicates):
                self.bootstrap_filter()
                self.get_recurrence_vectors()
                best_pop, min_error = self.get_best_pop()
                self.write_logs(f'best pop:{best_pop}, min error:{min_error}')
                #copy logs to output directory
                shutil.copy(self.logs_f, f'{self.prefix}/{self.logs_f}')
