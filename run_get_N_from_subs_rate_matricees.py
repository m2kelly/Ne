#!/usr/bin/env python3

import pickle
import numpy as np
from .pop_size import PopSizeCalculator
from . import essentials as es
from .base_operations import Operations
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#run with
#python -m NeParallel.run_get_N_from_subs_rate_matricees
# ---------------- inputs ----------------
name='vary1_full_greens_escape'
expected_N=100
data_name=f'rates_N_{expected_N}_T_300N_1000N_segg_vary.pkl'



data_dir='/home/dweghorngroup/test_avg_rate/beta_4_allele_renewal_beam/slim_clean/4_allele_data'
plot_dir=f"/home/dweghorngroup/test_avg_rate/beta_4_allele_renewal_beam/slim_clean/4_allele_model_results/N_{expected_N}/{name}"


list_of_rates=f'{data_dir}/{data_name}'
Path(plot_dir).mkdir(exist_ok=True,parents=True)
output_summary=f"{plot_dir}/results_per_non_cpg.csv"
generations = 1   # already divided by generations 

no_sims=10
BASES=['A','C','G','T']
operations = Operations(collapse=False,muts_dict_raw={},occ_dict_raw={},name=name)


#4 allele 300N to 1000N
with open(list_of_rates, "rb") as f:
    list_of_rates = pickle.load(f)
rates_dict={es.mutation(tri='A'+key[0]+'G',base=key[3]):value for key,value in list_of_rates.items()}

print(rates_dict)
cpg_label=es.mutation(tri='AAG',base='C')

#if running a combo of (nearly) all non cpgs
non_cpg_pool=list(rates_dict.keys())
non_cpg_pool.remove(cpg_label)

for b in BASES:
    non_cpg_pool.remove(es.mutation(tri='A'+b+'G',base=b)) #cannot have b->b muts
    '''
    if b!='A':
        #trying removing, as worst agreement wuth slim/maths
        non_cpg_labels.remove(es.mutation(tri='A'+b+'G',base='A'))
    if b not in ['C','A']:
        print('A'+b+'G','C')
        non_cpg_labels.remove(es.mutation(tri='A'+b+'G',base='C'))
    '''



# Pull only these mutation categories from rates_dict
cpg_subs = {
    mut: rates_dict[mut]
    for mut in [cpg_label]
}



# ---------------- population-size regression ----------------
def run_get_N(non_cpg_labels,plot_dir):
    '''assuming always the same 1 cpg categpry and rerunning for diffn non cpg combos'''
    cpg_non_cpg_dict={cpg_label:non_cpg_labels}
    non_cpg_subs = {
        mut: rates_dict[mut]
        for mut in non_cpg_labels
    }
    popcalc = PopSizeCalculator(
        gens=generations,
        cpg_subs=cpg_subs,
        non_cpg_subs=non_cpg_subs,

        cpg_subs_bckwrds=[],
        non_cpg_subs_bckwrds=[],

        cpg_occs=[],
        non_cpg_occs=[],
        cpg_occs_bckwrds=[],
        non_cpg_occs_bckwrds=[],

        directory="./",
        operations=operations,
    )
    popcalc.write_logs(f'non cpg categories: {non_cpg_labels}')
    popcalc.cpg_non_cpg_dict = cpg_non_cpg_dict
    popcalc.rates_dict = rates_dict
    popcalc.directory=plot_dir

    popcalc.calc_best_pop_regress_once(pop_min=4, pop_max=10_000,steps=[1_000,100,25,5,1])
    popcalc.plot_correction()

    #plot expected N
    cpg_muts, non_cpg_muts = popcalc.get_muts_per_cat(expected_N)
    plt.clf()
    x = np.array(non_cpg_muts).reshape(-1, 1)
    y = np.array(cpg_muts)
    model = LinearRegression(fit_intercept=False).fit(x, y)
    y_fit = model.predict(x)
    plt.plot(x, y_fit, color='black', label='Fit no intcp')
    model = LinearRegression().fit(x, y)
    y_fit = model.predict(x)
    plt.plot(x, y_fit, color='blue', label='Fit with intcp')
    plt.title(f'N={expected_N}')
    plt.scatter(x,y)
    plt.xlabel('non CpG muts'); plt.ylabel('CpG muts')
    plt.legend()
    plt.grid()
    plt.savefig(f'{popcalc.directory}/muts_regression_simulated_N.png') 

    popcalc.write_logs(f'result: {non_cpg_labels} N={popcalc.best_pop}')
    print("Best N:", popcalc.best_pop)
    print("Minimum error:", popcalc.min_error)
    return popcalc.best_pop,popcalc.min_error

N_estimates=[]
errors=[]

#regress per category
'''
non_cpg_dir= f'{plot_dir}/regress_per_cat_intcp'
Path(non_cpg_dir).mkdir(exist_ok=True,parents=True)
N,e=run_get_N(non_cpg_labels=non_cpg_pool,plot_dir=non_cpg_dir)
N_estimates.append(N)
errors.append(e)
'''


non_cpg_dir= f'{plot_dir}/all_muts'
Path(non_cpg_dir).mkdir(exist_ok=True,parents=True)
N,e=run_get_N(non_cpg_labels=non_cpg_pool,plot_dir=non_cpg_dir)
N_estimates.append(N)
errors.append(e)



for non_cpg in non_cpg_pool:
    
    non_cpg_dir= f'{plot_dir}/{str(non_cpg)}'
    Path(non_cpg_dir).mkdir(exist_ok=True,parents=True)
    N,e=run_get_N(non_cpg_labels=[non_cpg],plot_dir=non_cpg_dir)
    N_estimates.append(N)
    errors.append(e)

results_df=pd.DataFrame({'non_cpg':['all']+non_cpg_pool,'N':N_estimates,'error':errors})
results_df.to_csv(output_summary,index=False)
print(results_df)


'''
for non_cpg in non_cpg_pool:
    for non_cpg_2 in non_cpg_pool:
        if non_cpg==non_cpg_2:
            continue
        non_cpg_dir= f'{plot_dir}/{str(non_cpg)}_{str(non_cpg_2)}'
        Path(non_cpg_dir).mkdir(exist_ok=True,parents=True)
        N,e=run_get_N(non_cpg_labels=[non_cpg,non_cpg_2],plot_dir=non_cpg_dir)
        N_estimates.append(N)
        errors.append(e)
'''