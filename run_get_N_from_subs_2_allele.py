#!/usr/bin/env python3

import pickle

from .pop_size import PopSizeCalculator
from . import essentials as es
from .base_operations import Operations
from pathlib import Path
import pandas as pd
#run with
#python -m NeParallel.run_get_N_from_subs_rate_matricees
# ---------------- inputs ----------------

rates_file = "/home/dweghorngroup/data/slim_4_allele/N_10000_sims_mean_rate_per_bin.pkl"
plot_dir="/home/dweghorngroup/test_avg_rate/beta_4_allele_renewal_beam/slim/plots"
output_summary="/home/dweghorngroup/test_avg_rate/beta_4_allele_renewal_beam/slim/results_per_non_cpg.csv"
generations = 1   # already divided by generations 

no_sims=10
BASES=['A','C','G','T']
operations = Operations(collapse=False,muts_dict_raw={},occ_dict_raw={},name='run_per_non_cpg')

# ---------------- load ----------------

with open(rates_file, "rb") as f:
    rates_list = pickle.load(f)


'''expected rates dict, keys are mut objects and entries are lists of mut rates '''

#just change so is a mut object
rates_dict={es.mutation(tri='A'+a+'G',base=b):[rates_list[i].loc[a,b] for i in range(no_sims)] for a in BASES for b in BASES }
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

    popcalc.calc_best_pop_regress_per_category()
    popcalc.plot_correction()
    popcalc.write_logs(f'result: {non_cpg} N={popcalc.best_pop}')
    print("Best N:", popcalc.best_pop)
    print("Minimum error:", popcalc.min_error)
    return popcalc.best_pop,popcalc.min_error

N_estimates=[]
errors=[]

for non_cpg in non_cpg_pool:
    non_cpg_dir= f'{plot_dir}/{str(non_cpg)}'
    Path(non_cpg_dir).mkdir(exist_ok=True,parents=True)
    N,e=run_get_N(non_cpg_labels=[non_cpg],plot_dir=non_cpg_dir)
    N_estimates.append(N)
    errors.append(e)

results_df=pd.DataFrame({'non_cpg':non_cpg_pool,'N':N_estimates,'error':errors})
results_df.to_csv(output_summary,index=False)
print(results_df)