import pickle
import numpy as np
from .pop_size import PopSizeCalculator
from scipy.optimize import brute
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
from .base_operations import Operations
from . import essentials as es
from pathlib import Path
#from .predictor import Predictor
#from .graphpoint import GraphPointAbstract as GPA
#import .graphpoint as graphpoint

#now breaking when input backward sub rates=sub rate 

#run with python -m NeParallel.run_get_N_2_allele_from_4_allele_subs

name='vary2_non_cpg_G_'
expected_N=100
data_name=f'rates_N_{expected_N}_T_300N_1000N_segg_vary2.pkl'
allow_intcp=False

data_dir='/home/dweghorngroup/test_avg_rate/beta_4_allele_renewal_beam/slim_clean/4_allele_data'
plot_dir=f"/home/dweghorngroup/test_avg_rate/beta_4_allele_renewal_beam/slim_clean/2_allele_model_results/N_{expected_N}/{name}"


list_of_rates=f'{data_dir}/{data_name}'

Path(plot_dir).mkdir(exist_ok=True,parents=True)

with open(list_of_rates, "rb") as f:
    rates_dict = pickle.load(f)

rates_dict={key:np.array(value) for key,value in rates_dict.items()}
#keys format A->C

#per site
cpg_subs=rates_dict['A->C']+rates_dict['A->G']+rates_dict['A->T']
cpg_subs_bckwrds=rates_dict['C->A']+rates_dict['G->A']+rates_dict['T->A']
non_cpg_subs=rates_dict['T->G']+rates_dict['T->A']+rates_dict['T->C']
non_cpg_subs_bckwrds=rates_dict['G->T']+rates_dict['A->T']+rates_dict['C->T']

non_cpg_subs=rates_dict['G->T']+rates_dict['G->A']+rates_dict['G->C']
non_cpg_subs_bckwrds=rates_dict['T->G']+rates_dict['A->G']+rates_dict['C->G']



cpg_occs=[]
non_cpg_occs=[]
cpg_occs_bckwrds=[]
non_cpg_occs_bckwrds=[]

operations = Operations(collapse=False,muts_dict_raw={},occ_dict_raw={},directory=plot_dir,name=name)



def calc_best_pop(self):
    pop_min = 4
    pop_max = 5000

    # Search resolutions, from coarse to exact integer N.
    steps = [1000,100,10,5,1]

    # Avoid recalculating an N already tested at another level.
    error_cache = {}

    def evaluate(pop):
        pop = int(pop)

        if pop not in error_cache:
            error_cache[pop] = self.get_error_test(pop,allow_intcp=allow_intcp)

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
    


def run_inference(cpg_subs,non_cpg_subs,cpg_subs_bckwrds,non_cpg_subs_bckwrds,label):
    popcalc = PopSizeCalculator(gens=1, cpg_subs=cpg_subs, non_cpg_subs=non_cpg_subs, cpg_subs_bckwrds=cpg_subs_bckwrds,
                            non_cpg_subs_bckwrds=non_cpg_subs_bckwrds,cpg_occs=cpg_occs,
                            non_cpg_occs=non_cpg_occs, cpg_occs_bckwrds=cpg_occs_bckwrds, 
                            non_cpg_occs_bckwrds=non_cpg_occs_bckwrds,
                            directory='', operations=operations)
    run_dir=plot_dir + '/' + str(label)
    Path(run_dir).mkdir(exist_ok=True,parents=True)
    popcalc.directory=run_dir

    calc_best_pop(popcalc)

    #load results
    N=popcalc.best_pop
    error=popcalc.min_error
    cpg_muts, non_cpg_muts = popcalc.get_muts(N)
    popcalc.plot_regression(N,cpg_subs, non_cpg_subs, cpg_muts, non_cpg_muts, f'{popcalc.directory}/subs_regression.png', f'{popcalc.directory}/muts_regression.png')
            
    #poting results with simulated N        
    cpg_muts, non_cpg_muts = popcalc.get_muts(expected_N)
    popcalc.plot_regression(expected_N,cpg_subs, non_cpg_subs, cpg_muts, non_cpg_muts, f'{popcalc.directory}/subs_regression.png', f'{popcalc.directory}/muts_regression_simulated_N.png')
      

    popcalc.write_logs(f'repeat:{label},N={popcalc.best_pop}')
    print("Best N:", popcalc.best_pop)
    print("Minimum error:", popcalc.min_error)
    return N,error

N_estimates=[]
errors=[]

runs=[f'per_site_allow_intcp_{allow_intcp}_weight_bin' for allow_intcp in [True,False]]


for allow_intcp in [True,False]:
    i=f'per_site_allow_iintcp_{allow_intcp}_weight_bin'
    #N,e=run_inference(cpg_dict[i],non_cpg_dict[i],cpg_bckwrds_dict[i],non_cpg_bckwrds_dict[i],i)
    N,e=run_inference(cpg_subs,non_cpg_subs,cpg_subs_bckwrds,non_cpg_subs_bckwrds,i)
    N_estimates.append(N)
    errors.append(e)

results_df=pd.DataFrame({'repeats':runs,'N':N_estimates,'error':errors})
results_df.to_csv(plot_dir+'/summary.csv',index=False)
print(results_df)


