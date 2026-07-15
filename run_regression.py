from .poisson_regress_subs import Regress_subs
from .base_operations import Operations
from . import essentials as es

'''
run with python -m NeParallel.run_regression
so runs like package and relative imports work '''
target='Anc3'
anc='Anc1'
divg=8.4
gen=23.375

target='hg38'
anc='Anc4'
divg=6.2
gen=24.5

input='/home/dweghorngroup/data/homo_8_primate'

if '.' in target:
    target=target.split('.')[0]
if '.' in anc:
    anc=anc.split('.')[0]

no_gen=int((divg/gen)*1000000)

name=f'cactus_{target}_{anc}'
directory=f'{input}/'
generations=no_gen

regress=Regress_subs(name=name,directory=directory)
#regress.non_cpg_pool=['ACA->T','CCC->A','CCA->A','GCC->A','CCA->T','CCT->A','GCA->A','ACT->T']

regress.bin_and_plot(cpg_remove=0.2,cutoff=0.8)
#regress.run_cpg_all_non_cpg()
#regress.regress_to_choose()
#regress.regress_cpg_to_non_cpg()
#regress.regress_collapse_groups(regress.cpg_muts)
#regress.regress_collapse_groups(regress.non_cpg_pool)