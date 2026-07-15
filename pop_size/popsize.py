from statistics import mean
from matplotlib import pyplot as plt
from ..essentials import mutation
import numpy as np
import warnings
from .predictor import Predictor
from .graphpoint import GraphPointAbstract as GPA
from ..base_operations import Operations
import traceback
warnings.filterwarnings("ignore")
from joblib import Parallel, delayed
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize_scalar
from scipy.optimize import brute
from sklearn.linear_model import PoissonRegressor
import statsmodels.api as sm
from scipy.optimize import minimize

BASES = ['A', 'C', 'G', 'T']

class PopSizeCalculator(Operations):

    def __init__(self, gens, cpg_subs, non_cpg_subs, cpg_subs_bckwrds, non_cpg_subs_bckwrds,cpg_occs, non_cpg_occs, cpg_occs_bckwrds, non_cpg_occs_bckwrds, directory='', operations=None):
        Operations.__init__(self, operations=operations)

        self.gens = gens
        self.cpg_subs = cpg_subs
        self.non_cpg_subs = non_cpg_subs
        self.cpg_subs_bckwrds = cpg_subs_bckwrds
        self.non_cpg_subs_bckwrds = non_cpg_subs_bckwrds
        self.cpg_occs = cpg_occs
        self.non_cpg_occs = non_cpg_occs
        self.cpg_occs_bckwrds = cpg_occs_bckwrds
        self.non_cpg_occs_bckwrds = non_cpg_occs_bckwrds
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
                try:
                    return self.predictor.get_mu_scalar(GPA(0, b, int(pop), mean_sub_rate=subf))
                except Exception as e:
                    try: return self.predictor.get_mu_scalar(GPA(0, None, int(pop), mean_sub_rate=subf))
                    except Exception as e:
                        print(traceback.format_exc())
                        print(f'Error: {e} for pop: {pop}, subf: {subf}, b: {b}')
                        return np.inf

        cpg_muts = Parallel(n_jobs=-1, verbose=0)(
            delayed(parallel_predictor)(subf, subb, pop) for subf, subb in zip(self.cpg_subs, self.cpg_subs_bckwrds)
        )

        non_cpg_muts = Parallel(n_jobs=-1, verbose=0)(
            delayed(parallel_predictor)(subf, subb, pop) for subf, subb in zip(self.non_cpg_subs, self.non_cpg_subs_bckwrds)
        )

        return cpg_muts, non_cpg_muts
    
    def get_muts_per_cat(self, pop):
        
        # Parallelize the operation
        def parallel_predictor(pop,mut,sub_matrix,subf=0,b=0): #4aleles uses sub matrix not subf, b
            try:
                mu=self.predictor.get_mu(GPA(0, b, int(pop), mean_sub_rate=subf),sub_matrix,mut.tri[1],mut.base)
                self.write_logs(mut)
                self.write_logs(mu)
                return mu

            except Exception as e:
                print(traceback.format_exc())
                try:
                    return self.predictor.get_mu_scalar(GPA(0, b, int(pop), mean_sub_rate=subf))
                except Exception as e:
                    try: return self.predictor.get_mu_scalar(GPA(0, None, int(pop), mean_sub_rate=subf))
                    except Exception as e:
                        print(traceback.format_exc())
                        print(f'Error: {e} for pop: {pop}, subf: {subf}, b: {b}')
                        return np.inf
                    
        def make_4allele_matrix(context):
            """
            Make 4x4 mutation-rate matrix for fixed flanks context[0] _ context[2].

            Example:
                context = 'ACT'
                row C, col T uses key 'ACT->T'
            """
            left = context[0]
            right = context[2]

            matrix = {
                ref: {alt: None for alt in BASES}
                for ref in BASES
            }

            for ref in BASES:
                tri = f"{left}{ref}{right}"

                for alt in BASES:
                    if alt == ref:
                        matrix[ref][alt] = 0.0
                        continue
                    key=mutation(tri,alt)
                    
                    if key in self.rates_dict:
                        matrix[ref][alt] = self.rates_dict[key]
                    else:
                        key_rc = key.get_rev_comp()
                        
                        matrix[ref][alt] = self.rates_dict[key_rc]

                
            return matrix
        
        def get_n_runs(matrix):
            """
            Get number of runs from first list-valued matrix entry.
            Checks that all list-valued entries have same length.
            """
            n_runs = None

            for ref in BASES:
                for alt in BASES:
                    value = matrix[ref][alt]

                    if isinstance(value, list):
                        if n_runs is None:
                            n_runs = len(value)
                        elif len(value) != n_runs:
                            raise ValueError("Not all matrix entries have same list length")

            if n_runs is None:
                return 1

            return n_runs


        def get_matrix_run(matrix, i):
            """
            Convert matrix of lists into one scalar 4x4 matrix for run i.
            """
            run_matrix = {
                ref: {alt: 0.0 for alt in BASES}
                for ref in BASES
            }

            for ref in BASES:
                for alt in BASES:
                    value = matrix[ref][alt]

                    if isinstance(value, list):
                        run_matrix[ref][alt] = value[i]
                    else:
                        run_matrix[ref][alt] = value

            return run_matrix
            
        cpgs=[]
        non_cpgs=[]
        for cpg,non_cpg in self.cpg_non_cpg_dict.items():
            
            #make 4 allele mut matrix for each cpg, non cpg context, need mu_frwd and mu_back for all alts at middle base
            cpg_matrix=make_4allele_matrix(cpg.tri)
            non_cpg_matrix=make_4allele_matrix(non_cpg.tri)

            n_cpg=get_n_runs(cpg_matrix)
            n_non_cpg=get_n_runs(non_cpg_matrix)

            
            #edit so input one rate in matrix at a time (now matrix of list)
            cpg_muts = Parallel(n_jobs=-1, verbose=0)(
                delayed(parallel_predictor)(pop,cpg,get_matrix_run(cpg_matrix, i)) for i in range(n_cpg)
            )

            non_cpg_muts = Parallel(n_jobs=-1, verbose=0)(
                delayed(parallel_predictor)(pop,non_cpg,get_matrix_run(non_cpg_matrix, i)) for i in range(n_non_cpg)
            )
            cpgs.append(cpg_muts)
            non_cpgs.append(non_cpg_muts)

        #take mean at eahc index across litsd, of lists of lists
        
        cpg_muts = [np.mean(vals) for vals in zip(*cpgs)]
        non_cpg_muts = [np.mean(vals) for vals in zip(*non_cpgs)]
        #self.write_logs(f'cpg_muts: {cpg_muts}, non_cpg_muts: {non_cpg_muts}')
        return cpg_muts, non_cpg_muts
    

    def get_error(self, pop):
    
        cpg_muts, non_cpg_muts = self.get_muts(pop)
        if np.isfinite(cpg_muts).all() and np.isfinite(non_cpg_muts).all():
            x = np.array(non_cpg_muts).reshape((-1, 1)) # must be 2D
            y = np.array(cpg_muts)
            weights=range(1, len(non_cpg_muts) + 1) #weight by more mutable bins, as trust them more, and want to fit line based on them
            #sample_weight=weights
            #fit_intercept=False
            model = LinearRegression().fit(x, y,sample_weight=weights)
            #R2  assumes mean centerong,not true for mut rate
            error = 1-model.score(x, y,sample_weight=weights)


            intercept = model.intercept_
            coeff = model.coef_[0]
            self.write_log = f'pop: {pop}, error: {error}, intercept: {intercept}, coeff: {coeff}\n'
            print(f'{pop}:{error}')
            return error
        else:
            return 1 #max error 
        
    def get_error_regress_per_category(self, pop):
    
        cpg_muts, non_cpg_muts = self.get_muts_per_cat(pop)
        if np.isfinite(cpg_muts).all() and np.isfinite(non_cpg_muts).all():
            x = np.array(non_cpg_muts).reshape((-1, 1)) # must be 2D
            y = np.array(cpg_muts)
            weights=range(1, len(non_cpg_muts) + 1) #weight by more mutable bins, as trust them more, and want to fit line based on them
            #sample_weight=weights
            #fit_intercept=False
            model = LinearRegression(fit_intercept=False).fit(x, y,sample_weight=weights)
            #R2  assumes mean centerong,not true for mut rate
            error = 1-model.score(x, y,sample_weight=weights)


            intercept = model.intercept_
            coeff = model.coef_[0]
            self.write_log = f'pop: {pop}, error: {error}, intercept: {intercept}, coeff: {coeff}\n'
            print(f'{pop}:{error}')
            return error
        else:
            return 1 #max error 
        
    def get_error_test(self, pop):
    
        cpg_muts, non_cpg_muts = self.get_muts(pop)
        if np.isfinite(cpg_muts).all() and np.isfinite(non_cpg_muts).all():
            x = np.array(non_cpg_muts).reshape((-1, 1)) # must be 2D
            y = np.array(cpg_muts)
            weights=range(1, len(non_cpg_muts) + 1) #weight by more mutable bins, as trust them more, and want to fit line based on them
            #sample_weight=weights
            #fit_intercept=Falsefit
            model = LinearRegression(fit_intercept=False).fit(x, y,sample_weight=weights)
            #R2  assumes mean centerong,not true for mut rate
            error = 1-model.score(x, y,sample_weight=weights)


            intercept = model.intercept_
            coeff = model.coef_[0]
            self.write_log = f'pop: {pop}, error: {error}, intercept: {intercept}, coeff: {coeff}\n'
            print(f'{pop}:{error}')
            return error
        else:
            return 1 #max error 
        
    

    #standard method-quick quadratic convergence where possible
    def calc_best_pop_local(self):
        res = minimize_scalar(
        self.get_error,
        bracket=[50_000, 500_000],
        method="brent",
        options={"xtol": 1e-6}
        )
        self.best_pop = res.x
        self.min_error = res.fun
        print(f'best pop: {self.best_pop}, error: {self.min_error}')

        #global optimiser
    def calc_best_pop(self):

        res = brute(
        self.get_error,
        ((10_000, 800_000),),
        Ns=20,
        full_output=False,
        workers=10
        )

        self.best_pop = float(res)
        self.min_error = self.get_error(res)
        print(f'best pop: {self.best_pop}, error: {self.min_error}')
        return
    
    def calc_best_pop_regress_per_category(self):

        res = brute(
        self.get_error_regress_per_category,
        ((10_000, 800_000),),
        Ns=20,
        full_output=False,
        workers=10
        )

        self.best_pop = float(res)
        self.min_error = self.get_error_regress_per_category(res)
        print(f'best pop: {self.best_pop}, error: {self.min_error}')
        return

    def _plot_bins_lineplot(self, subs_1, subs_2, muts_1, muts_2, name):
        '''plot the bins'''
        plt.clf()
        fig, (ax1, ax2) = plt.subplots(1, 2)

        ax1.plot(subs_1, label='CpG_subs', color='r')
        ax1.plot(subs_2, label='nonCpG_subs', color='b')

        index = np.arange(len(subs_1))
        

        #ax1.set_ylabel('Substitution Rate')
        ax1.set_title('Normalised substitution Rates', loc='left')
        ax1.legend()

        ax1.plot(muts_1, label='CpG_muts', color='r')
        ax1.plot(muts_2, label='nonCpG_muts', color='b')

        ax2.set_title('Normalised mutation Rates', loc='left')
        ax2.legend()

        plt.savefig(name)

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

    def plot_regression(self,cpg_subs, non_cpg_subs, cpg_muts, non_cpg_muts, name_subs,name_muts):
        '''plot the regression for muts and subs'''
        plt.clf()
        x = np.array(non_cpg_subs).reshape(-1, 1)
        y = np.array(cpg_subs)
        model = LinearRegression(fit_intercept=False).fit(x, y)
        y_fit = model.predict(x)
        plt.plot(x, y_fit, color='black', label='Fit no intcp')

        model = LinearRegression().fit(x, y)
        y_fit = model.predict(x)
        plt.plot(x, y_fit, color='blue', label='Fit with intcp')
        plt.scatter(x,y)
        plt.xlabel('non CpG subs'); plt.ylabel('CpG subs')
        plt.legend()
        
        plt.savefig(name_subs)

        plt.clf()
        x = np.array(non_cpg_muts).reshape(-1, 1)
        y = np.array(cpg_muts)
        model = LinearRegression(fit_intercept=False).fit(x, y)
        y_fit = model.predict(x)
        plt.plot(x, y_fit, color='black', label='Fit no intcp')

        model = LinearRegression().fit(x, y)
        y_fit = model.predict(x)
        plt.plot(x, y_fit, color='blue', label='Fit with intcp')

        plt.scatter(x,y)
        plt.xlabel('non CpG muts'); plt.ylabel('CpG muts')
        plt.legend()
        plt.savefig(name_muts)

        

    def plot_correction(self, best_pop=None):

        if best_pop is None: best_pop = self.best_pop

        cpg_muts, non_cpg_muts = self.get_muts_per_cat(best_pop)

         #cpg_subs is a dict, take mean across all cpgs
        cpg_subs = np.array([np.mean(vals) for vals in zip(*self.cpg_subs.values())])
        non_cpg_subs = np.array([np.mean(vals) for vals in zip(*self.non_cpg_subs.values())])
        
        cpg_muts = np.array(cpg_muts); non_cpg_muts = np.array(non_cpg_muts)

        self.plot_regression(cpg_subs, non_cpg_subs, cpg_muts, non_cpg_muts, f'{self.directory}/subs_regression.png', f'{self.directory}/muts_regression.png')
        cpg_subs_by_mean = cpg_subs/mean(cpg_subs); non_cpg_subs_by_mean = non_cpg_subs/mean(non_cpg_subs)
        cpg_muts_by_mean = cpg_muts/mean(cpg_muts); non_cpg_muts_by_mean = non_cpg_muts/mean(non_cpg_muts)

        self._plot_bins(cpg_subs_by_mean, non_cpg_subs_by_mean, cpg_muts_by_mean, non_cpg_muts_by_mean, f'{self.directory}/best_pop_by_mean.png')
        self._plot_bins_lineplot(cpg_subs_by_mean, non_cpg_subs_by_mean, cpg_muts_by_mean, non_cpg_muts_by_mean, f'{self.directory}/best_pop_by_mean_lineplot.png')
        
        cpg_subs_by_zero = cpg_subs/cpg_subs[0]; non_cpg_subs_by_zero = non_cpg_subs/non_cpg_subs[0]
        cpg_muts_by_zero = cpg_muts/cpg_muts[0]; non_cpg_muts_by_zero = non_cpg_muts/non_cpg_muts[0]

        self._plot_bins(cpg_subs_by_zero, non_cpg_subs_by_zero, cpg_muts_by_zero, non_cpg_muts_by_zero, f'{self.directory}/best_pop_by_zero.png')

        perc_change_cpg = [(cpg_muts[i]-cpg_subs[i])/cpg_subs[i] for i in range(len(cpg_subs))]
        perc_change_noncpg = [(non_cpg_muts[i]-non_cpg_subs[i])/non_cpg_subs[i] for i in range(len(non_cpg_subs))]

        
        self.write_logs(f'perc change mut-sub/sub CpG: {perc_change_cpg}')
        
        self.write_logs(f'perc change mut-sub/sum nonCpG: {perc_change_noncpg}')


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
            

