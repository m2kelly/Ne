from statistics import mean
from matplotlib import pyplot as plt
import numpy as np
import warnings
from .predictor import Predictor
import traceback
from .graphpoint import GraphPointAbstract as GPA
from . import graphpoint
warnings.filterwarnings("ignore")
from joblib import Parallel, delayed
from sklearn.linear_model import LinearRegression
import time
'''
run with python -m NeParallel.pop_size.test_run_get_sub_4_alleles'''

N_power=5
no_sims=10
output_fig=f'/home/dweghorngroup/test_avg_rate/beta_4_allele_renewal_beam/4_alleles_subs/N_small_mu_1e{N_power}.svg'
predictor = Predictor()
plt.rcParams.update({'font.size': 16})
import pandas as pd
BASES=['A','C','G','T']
'''
updating so get sub for all pairs of categories at each bin and plot all trajectories'''

#changing N
'''

mu=1e-8

#create example mu matrix
mu_matrix={'A':{'C':mu,'G':mu,'T':mu},
           'C':{'A':10*mu,'G':mu,'T':mu},
           'G':{'A':mu,'C':mu,'T':mu},
           'T':{'A':mu,'C':mu,'G':mu}}
  
Ns=np.logspace(4, 6, 20)


#beta approximation gives exactly same result-much faster1
#subs = [predictor.get_sub_rate(GPA(mu,mu, pop_size=int(N_e))) for N_e in Ns]
subs_beta=[predictor.get_sub_rate_diffusion_beta(GPA(mu,10*mu, pop_size=int(N_e))) for N_e in Ns]
print(subs_beta)
#subs_2=[predictor.get_sub_rate_diffusion_prob_hit(GPA(mu,mu, pop_size=int(N_e))) for N_e in Ns]
#print(subs_2)
subs_4_allele=[predictor.get_sub_rate_4allele_edge_renewal(GPA(mu,mu, pop_size=int(N_e)),a='A',b='C',mu_matrix=mu_matrix) for N_e in Ns]
print(subs_4_allele)
#lt.scatter(Ns, subs_2,color='black',alpha=0.2,label='diffusion prob hit')
plt.scatter(Ns,subs_beta,color='red',alpha=0.2,label='diffusion beta')
plt.scatter(Ns,subs_4_allele,color='blue',alpha=0.2,label='4-allele edge')
plt.axhline(y=mu,label='inputed mut rate')
plt.xlabel('N_e')
plt.ylabel('subs')
plt.title(f'mu {mu}')
plt.legend()

plt.savefig(F'/home/dweghorngroup/run_get_mu/sub_mut_N_E_change_wfes_diff_waiting.svg',bbox_inches='tight')
plt.show()
'''

#varying mut rate


#muts=np.logspace(-5, -3, no_sims)
#muts=np.linspace(1e-8, 1e-7, no_sims)
mu_power=N_power+3
muts=np.linspace(10**(-mu_power-1), 10**(-mu_power), no_sims)
N_e = int(10**N_power)
BASES = ["A", "C", "G", "T"]
mu_dicts=[
            {
                "A": {"A": 0.0, "C": 10*mu,     "G": mu, "T": mu},
                "C": {"A": mu, "C": 0.0, "G": mu, "T": mu},
                "G": {"A": mu,     "C": mu,  "G": 0.0, "T": mu},
                "T": {"A": mu,     "C": mu,  "G": mu, "T": 0.0},
            }
    for mu in muts]
print(mu_dicts[0])
subs_beta=[predictor.get_sub_rate_diffusion_beta(GPA(mu,mu, pop_size=int(N_e))) for mu in muts]
#get_sub_rate_diffusion_prob_hit
subs_2_neutral=[predictor.get_sub_rate_diffusion_beta(GPA(mu,mu, pop_size=int(N_e))) for mu in muts]
subs_2_high_low=[predictor.get_sub_rate_diffusion_beta(GPA(10*mu,mu, pop_size=int(N_e))) for mu in muts]
subs_2_low_high=[predictor.get_sub_rate_diffusion_beta(GPA(mu,mu*10, pop_size=int(N_e))) for mu in muts]

print('finished computing 2 allele')
print(time.time())
subs_4_allele_matrices=[predictor.get_sub_rate_diffusion_4_allele_renewal(GPA(0,0, pop_size=int(N_e)),
            mu_matrix=mu_dict) for mu_dict in mu_dicts]
#each run returns nested dict of sub rates 
print('finished computing 4 allele')
print(time.time())
print('last marix')

last_dict={a+'->'+b:subs_4_allele_matrices[no_sims-1][a][b] for a in BASES for b in BASES }
print(pd.Series(last_dict).sort_values())
rates_dict={a+'->'+b:[subs_4_allele_matrices[i][a][b] for i in range(no_sims)] for a in BASES for b in BASES }
#print(rates_dict)
#take the mean of the matrices
bins=[x for x in range(no_sims)]
fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(bins, muts, color="red", label="input mut rate")
ax.plot(bins, [10 * x for x in muts], color="red", linestyle="--", label="input mut rate A->C")

linestyles = ["-", "--", "-.", ":"]
i=0
for a in BASES:
    for b in BASES:
        if a == b:
            continue
        #if a!='A': continue
        ax.plot(
            bins,
            rates_dict[a + "->" + b],
            alpha=0.5,
            label=a + "->" + b + " sub rate",
            linestyle=linestyles[i%len(linestyles)],
            linewidth=2
        )
        i+=1

ax.plot(bins, subs_2_neutral, color="black", linestyle="--",label="2 alelles mu->mu")
ax.plot(bins, subs_2_high_low, color="black", linestyle="--",label="2 alelles 10mu->mu")
ax.plot(bins, subs_2_low_high, color="black", linestyle="--",label="2 alelles mu->10mu")
ax.set_xlabel("bin")
ax.set_ylabel("rate")

ax.legend(
    loc="center left",
    bbox_to_anchor=(1.02, 0.5),
    borderaxespad=0
)

fig.tight_layout()
plt.title('N= '+ str(N_e))
fig.savefig(output_fig, bbox_inches="tight", dpi=300)
plt.show()


#add 2 allele trajectories 


'''


plt.figure()
#plt.scatter(muts, subs_2,color='black',alpha=0.2,label='diffusion prob hit')
plt.scatter(muts,subs_beta,color='red',alpha=0.2,label='diffusion beta')
plt.scatter(muts,subs_4_allele,color='blue',alpha=0.2,label='4-allele edge')
plt.plot(muts,muts,label='inputed mut rate')
plt.xlabel('mu')
plt.ylabel('subs')
plt.title(f'Ne {N_e}')
plt.legend()

plt.savefig(F'/home/dweghorngroup/run_get_mu/sub_mut_N_E_change_wfes_diff_4allele_mu.svg',bbox_inches='tight')
plt.show()
'''
