from statistics import mean
from matplotlib import pyplot as plt
import numpy as np
import warnings
from .predictor import Predictor
from .. import essentials as es
import traceback
from .graphpoint import GraphPointAbstract as GPA
from . import graphpoint
warnings.filterwarnings("ignore")
from joblib import Parallel, delayed
from sklearn.linear_model import LinearRegression
import time
import pickle
import pandas as pd
'''
run with python -m NeParallel.pop_size.compare_sub_slim'''

N_power=3
no_sims=10

predictor = Predictor()
plt.rcParams.update({'font.size': 16})

BASES=['A','C','G','T']
'''
script to compare calculated sub rates vs slim rates 
'''

list_of_rates='/home/dweghorngroup/test_avg_rate/beta_4_allele_renewal_beam/slim/4_allele_data/rates_N_1000_T_300N_1000N_segg.pkl'

#4 allele 300N to 1000N
with open(list_of_rates, "rb") as f:
    slim_dict = pickle.load(f)



#varying mut rate


#muts=np.logspace(-5, -3, no_sims)
#muts=np.linspace(1e-8, 1e-7, no_sims)
mu_power=N_power+2
#low mu
#mu_power+=1

muts=np.linspace(10**(-mu_power-1), 10**(-mu_power), no_sims)
alt_muts=10*muts
N_e = int(10**N_power)
BASES = ["A", "C", "G", "T"]


mu_dicts=[
            {
                "A": {"A": 0.0, "C": 10*mu,     "G": mu, "T": mu},
                "C": {"A": mu, "C": 0.0, "G": mu, "T": mu},
                "G": {"A": mu,     "C":mu,  "G": 0, "T": mu},
                "T": {"A": mu,     "C": mu,  "G": mu, "T": 0.0},
            }
    for mu in muts]


#_current_first_order_occupation
subs_4_allele_matrices=[predictor.get_sub_rate_diffusion_4_allele_renewal(GPA(0,0, pop_size=int(N_e)),
            mu_matrix=mu_dict) for mu_dict in mu_dicts]
#each run returns nested dict of sub rates 
print('finished computing 4 allele')
print(time.time())


#last_dict={a+'->'+b:subs_4_allele_matrices[no_sims-1][a][b] for a in BASES for b in BASES }
#print(pd.Series(last_dict).sort_values())
rates_dict={a+'->'+b:[subs_4_allele_matrices[i][a][b] for i in range(no_sims)] for a in BASES for b in BASES }

#take the mean of the matrices
bins=[x for x in range(no_sims)]

def plot_category(cat,slim_subs,diffusion_subs,muts):
    slim_subs=np.array(slim_subs)
    diffusion_subs=np.array(diffusion_subs)
    '''
    plt.figure()
    plt.plot(bins,slim_subs,color='black',alpha=0.5,label=cat+' slim sub rate')
    plt.plot(bins,diffusion_subs,color='green',alpha=0.5,label=cat+' diffusion sub rate')
    plt.plot(bins, muts, color="red", label="input mut rate")
    print(cat,slim_subs/diffusion_subs)
    plt.legend()
    plt.title(cat)
    plt.tight_layout()
    '''
    return slim_subs/diffusion_subs
    #plt.show()

cats=[a+'->'+b for a in BASES for b in BASES if a!=b]
ratios=[]
for cat in cats:
    slim_subs=slim_dict[cat]
    diffusion_subs=rates_dict[cat]
    if cat=='A->C':
        ratios.append(plot_category(cat,slim_subs,diffusion_subs,alt_muts))

    else:   
        ratio=plot_category(cat,slim_subs,diffusion_subs,muts)
        ratios.append(ratio)

linestyles = ["-", "--", "-.", ":"]
plt.figure()
i=0
for cat,ratio in zip(cats,ratios):
    plt.plot(bins,ratio,alpha=0.5,linestyle=linestyles[i%len(linestyles)],linewidth=3,label=cat+' ratio')
    i+=1

plt.axhline(y=1.0, color='black', alpha=1, label='y=1.0')
plt.legend(fontsize=10)
plt.tight_layout()
plt.title('slim sub rate / diffusion sub rate')
plt.xlabel('bin')
plt.savefig(f'{N_e}_slim_vs_diffusion.png')
plt.show()