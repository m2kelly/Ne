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
        
#########2 allele model functions
    def get_muts(self, pop):

        '''
        calculate cpg and non cpg muts for each category seperatly using 2 allele model '''
        # Parallelize the operation
        def parallel_predictor(subf, b, pop):
            try:
                return self.predictor.get_mu_2_allele(GPA(0, b, int(pop), mean_sub_rate=subf))
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

        cpg_muts = Parallel(n_jobs=4, verbose=0)(
            delayed(parallel_predictor)(subf, subb, pop) for subf, subb in zip(self.cpg_subs, self.cpg_subs_bckwrds)
        )

        non_cpg_muts = Parallel(n_jobs=4, verbose=0)(
            delayed(parallel_predictor)(subf, subb, pop) for subf, subb in zip(self.non_cpg_subs, self.non_cpg_subs_bckwrds)
        )

        return cpg_muts, non_cpg_muts

    def get_error(self, pop):
        '''
        2 allele model 
        calculate muts per inputted N guess
        then get error in cpg/non cpg mut ratio'''
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


            #intercept = model.intercept_
            #coeff = model.coef_[0]
            
            # self.write_log = f'pop: {pop}, error: {error}, intercept: {intercept}, coeff: {coeff}\n'
            print(f'{pop}:{error}')
            self.write_logs(f'{pop}:{error}')
            return error
        else:
            return 1 #max error 

    def get_error_test(self, pop,allow_intcp=True):
    
        cpg_muts, non_cpg_muts = self.get_muts(pop)
        if np.isfinite(cpg_muts).all() and np.isfinite(non_cpg_muts).all():
            x = np.array(non_cpg_muts).reshape((-1, 1)) # must be 2D
            y = np.array(cpg_muts)
            weights=range(1, len(non_cpg_muts) + 1) #weight by more mutable bins, as trust them more, and want to fit line based on them
            #weights=[x for x in non_cpg_muts]
            #,sample_weight=weights
            if allow_intcp:
                model = LinearRegression().fit(x, y,sample_weight=weights)
                            #R2  assumes mean centerong,not true for mut rate
                            #assume intcpt>=0 
                if model.intercept_ > 0:
                    error = 1-model.score(x, y,sample_weight=weights)
                else:
                    model_no_intcp = LinearRegression(fit_intercept=False).fit(x, y,sample_weight=weights)
                    error = 1-model_no_intcp.score(x, y,sample_weight=weights)         
            else:
                model_no_intcp = LinearRegression(fit_intercept=False).fit(x, y,sample_weight=weights)
                error = 1-model_no_intcp.score(x, y,sample_weight=weights) 

            
            print(f'{pop}:{error}')
            return error
        else:
            return 1 #max error 

    def calc_best_pop(self):
        '''
        2 allele model
        find the best pop size by regression of cpg and non cpg mut rate for each N guess
        runs one regression from the mean of all cpgs and mean of all non cpgs
        runs a grid search
        steps gives search resolution from coarse to exact'''
            
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
    

##########4 alelle model functions           
    def get_muts_per_cat(self, pop,take_mean=True):
        '''
        4 allele model
        get the muts for each cpg and non cpg category in cpg_non_cpg_dict
        if take_mean=True, return the mean of all cpgs and mean of all non cpgs, else return the list of muts for each category
        '''
        # Parallelize the operation
        def parallel_predictor(pop,mut,sub_matrix,subf=0,b=0): #4aleles uses sub matrix not subf, b
            try:
                mu=self.predictor.get_mu(GPA(0, b, int(pop), mean_sub_rate=subf),sub_matrix,mut.tri[1],mut.base)
                #self.write_logs(mut)
                #self.write_logs(mu)
                return mu

            except Exception as e:
                print(traceback.format_exc())
                return np.inf
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

        def process_cat(mut_label):
            matrix=make_4allele_matrix(mut_label.tri)
            n=get_n_runs(matrix)
            muts = Parallel(n_jobs=-1, verbose=0)(
                            delayed(parallel_predictor)(pop,mut_label,get_matrix_run(matrix, i)) for i in range(n)
                        )
            return muts
            
        cpgs=[]
        non_cpgs=[]
        #to avoid recounting the same category twice if it appears twice
        cpg_labels=[]
        non_cpg_labels=[]
        for cpg,non_cpg in self.cpg_non_cpg_dict.items():
            
            if cpg in cpg_labels:
                cpgs.append(cpgs[cpg_labels.index(cpg)]) #already processed
            else:
                cpgs.append(process_cat(cpg))
            cpg_labels.append(cpg)

            if type(non_cpg) is list:
                non_cpg_list=non_cpg
                for non_cpg in non_cpg_list:
                    
                    if non_cpg in non_cpg_labels:
                        #if already porcessed this mut category
                        non_cpgs.append(non_cpgs[non_cpg_labels.index(non_cpg)])
                    else:
                        #else process-find mu
                        non_cpgs.append(process_cat(non_cpg))
                    non_cpg_labels.append(non_cpg)
            else:
                if non_cpg in non_cpg_labels:
                    #if already porcessed this mut category
                    non_cpgs.append(non_cpgs[non_cpg_labels.index(non_cpg)])
                else:
                    #else process-find mu
                    non_cpgs.append(process_cat(non_cpg))
                non_cpg_labels.append(non_cpg)

            
        #take mean at each index across lits, of lists of lists-when using more than one cpg
        if take_mean:
            if len(set(cpg_labels))>1:
                cpg_muts = [np.mean(vals) for vals in zip(*cpgs)]
            else:
                cpg_muts=cpgs[0]
            if len(set(non_cpg_labels))>1:
                #take bin across categories
                non_cpg_muts = [np.mean(vals) for vals in zip(*non_cpgs)]
            else:
                non_cpg_muts=non_cpgs[0]
            #self.write_logs(f'cpg_muts: {cpg_muts}, non_cpg_muts: {non_cpg_muts}')
            return cpg_muts, non_cpg_muts
        else:
            return cpgs,non_cpgs
     
    def get_error_regress_per_category(self, pop):
    
        cpg_muts_lists, non_cpg_muts_lists = self.get_muts_per_cat(pop,take_mean=False)
        total_error=0
        for i in range(len(cpg_muts_lists)):
            cpg_muts = cpg_muts_lists[i]
            non_cpg_muts = non_cpg_muts_lists[i]
        
            if np.isfinite(cpg_muts).all() and np.isfinite(non_cpg_muts).all():
                x = np.array(non_cpg_muts).reshape((-1, 1)) # must be 2D
                y = np.array(cpg_muts)
                weights=range(1, len(non_cpg_muts) + 1) #weight by more mutable bins, as trust them more, and want to fit line based on them
                #sample_weight=weights
                #fit_intercept=False
                model = LinearRegression().fit(x, y,sample_weight=weights)
                #R2  assumes mean centerong,not true for mut rate
                error = 1-model.score(x, y,sample_weight=weights)
                total_error+=error

                #intercept = model.intercept_
                #coeff = model.coef_[0]
            else:
                total_error+=10 #max error
        self.write_log = f'pop: {pop}, error: {total_error}'
        print(f'{pop}:{total_error}')
        return total_error

    def get_error_regress_once(self, pop):
        '''
        4 allele model
        one regression fo rmean of cpg and mean of non cpg
        infer N based on assuming constant cpg/non_cpg ratio'''
        cpg_muts, non_cpg_muts = self.get_muts_per_cat(pop)
        if np.isfinite(cpg_muts).all() and np.isfinite(non_cpg_muts).all():
            x = np.array(non_cpg_muts).reshape((-1, 1)) # must be 2D
            y = np.array(cpg_muts)
            weights=range(1, len(non_cpg_muts) + 1) #weight by more mutable bins, as trust them more, and want to fit line based on them
            #weights=[x*x for x in non_cpg_muts]
            #sample_weight=weights
            #fit_intercept=False
            #model = LinearRegression().fit(x, y,sample_weight=weights)
            #R2  assumes mean centerong,not true for mut rate
            #assume intcpt>=0 
            #if model.intercept_ > 0:
                #error = 1-model.score(x, y,sample_weight=weights)
            #else:
            model_no_intcp = LinearRegression(fit_intercept=False).fit(x, y,sample_weight=weights)
            error = 1-model_no_intcp.score(x, y,sample_weight=weights)         


            #intercept = model.intercept_
            #coeff = model.coef_[0]
            #self.write_log = f'pop: {pop}, error: {error}, intercept: {intercept}, coeff: {coeff}\n'
            self.write_logs(f'{pop}:{error}')
            
            return error
        else:
            return 1 #max error 
              
    def calc_best_pop_regress_once(self,pop_min=1_000, pop_max=800_000,steps=[40_000, 4_000, 1_000]):
            
            '''
            4 allele model
            find the best pop size by regression of cpg and non cpg mut rate for each N guess
            runs one regression from the mean of all cpgs and mean of all non cpgs
            runs a grid search
            steps gives search resolution from coarse to exact'''
    
            
            # Avoid recalculating an N already tested at another level.
            error_cache = {}
    
            def evaluate(pop):
                pop = int(pop)
    
                if pop not in error_cache:
                    error_cache[pop] = self.get_error_regress_once(pop)
    
                return error_cache[pop]
    
            lower = pop_min
            upper = pop_max
            best_pop = None
    
            for step in steps:
                candidates = list(range(lower, upper + 1, step))
    
                # range() may not land exactly on the upper boundary.
                if candidates[-1] != upper:
                    candidates.append(upper)
    
                best_pop = min(candidates, key=evaluate)
    
                self.write_logs(
                    f"step={step}, best population={best_pop}, "
                    f"error={evaluate(best_pop)}"
                )
    
                # At the next resolution, search around this level's winner.
                lower = max(pop_min, best_pop - step)
                upper = min(pop_max, best_pop + step)
    
            self.best_pop = best_pop
            self.min_error = evaluate(best_pop)
    
            self.write_logs(
                f"best pop: {self.best_pop}, "
                f"error: {self.min_error}, "
                f"unique evaluations: {len(error_cache)}"
            )
            
    def calc_best_pop_regress_per_category(self,pop_min=1_000, pop_max=800_000,steps=[40_000, 4_000, 1_000]):
        '''
        4 allele model
        find the best pop size by regression of cpg and non cpg mut rate for each N guess
        runs a regression per each cpg:non cpg list in cpg_non_cpg_dict
        runs a grid search
        steps gives search resolution from coarse to exact
        '''
            
        

        # Avoid recalculating an N already tested at another level.
        error_cache = {}

        def evaluate(pop):
            pop = int(pop)

            if pop not in error_cache:
                error_cache[pop] = self.get_error_regress_per_category(pop)

            return error_cache[pop]

        lower = pop_min
        upper = pop_max
        best_pop = None

        for step in steps:
            candidates = list(range(lower, upper + 1, step))

            # range() may not land exactly on the upper boundary.
            if candidates[-1] != upper:
                candidates.append(upper)

            best_pop = min(candidates, key=evaluate)

            self.write_logs(
                f"step={step}, best population={best_pop}, "
                f"error={evaluate(best_pop)}"
            )

            # At the next resolution, search around this level's winner.
            lower = max(pop_min, best_pop - step)
            upper = min(pop_max, best_pop + step)

        self.best_pop = best_pop
        self.min_error = evaluate(best_pop)

        self.write_logs(
            f"best pop: {self.best_pop}, "
            f"error: {self.min_error}, "
            f"unique evaluations: {len(error_cache)}"
        )
        

##########Plotting functions
    def plot_regression(self,best_pop,cpg_subs, non_cpg_subs, cpg_muts, non_cpg_muts, name_subs,name_muts):
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
        plt.title(f'N={best_pop}')
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
        plt.title(f'N={best_pop}')
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

        self.plot_regression(best_pop,cpg_subs, non_cpg_subs, cpg_muts, non_cpg_muts, f'{self.directory}/subs_regression.png', f'{self.directory}/muts_regression.png')
        

        

