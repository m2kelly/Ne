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
run with python -m NeParallel.pop_size.compare_sub_slim_2_allele'''

N_power=2
no_sims=10

predictor = Predictor()
plt.rcParams.update({'font.size': 16})

BASES=['A','C','G','T']
'''
script to compare calculated sub rates vs slim rates 
'''

cpg_subs=[9.700135339530697e-05, 0.0001846543073927325, 0.00026714127212595896, 0.00034540404523002215, 0.00041891693544705193, 0.0004865798837072454, 0.0005508961474039859, 0.0006141430759257099, 0.0006765516564118276, 0.0007346271242363386]
cpg_subs_bckwrds=[9.483850244869565e-06, 1.8344688359018944e-05, 2.6238613259454207e-05, 3.2987748823422e-05, 3.9534625290324006e-05, 4.5290635686253855e-05, 4.9682236806880525e-05, 5.365161829659818e-05, 5.722476568478805e-05, 5.956171705043715e-05]

#from less generrations
non_cpg_subs=[1.0010732266477595e-05, 1.99330997130442e-05, 2.9276502902046252e-05, 3.870107320566423e-05, 4.819949843695813e-05, 5.675948434201272e-05, 6.628041949192711e-05, 7.524434293149229e-05, 8.405811951627675e-05, 9.251108676963504e-05]

slim_dict={'high->low':cpg_subs,'low->high':cpg_subs_bckwrds,'low->low':non_cpg_subs}

mu=1e-8


#varying mut rate


#muts=np.logspace(-5, -3, no_sims)
#muts=np.linspace(1e-8, 1e-7, no_sims)
mu_power=N_power+2
muts=np.linspace(10**(-mu_power-1), 10**(-mu_power), no_sims)
alt_muts=10*muts
N_e = int(10**N_power)



subs_2_neutral=[predictor.get_sub_rate_diffusion_beta(GPA(mu,mu, pop_size=int(N_e))) for mu in muts]
subs_2_high_low=[predictor.get_sub_rate_diffusion_beta(GPA(10*mu,mu, pop_size=int(N_e))) for mu in muts]
subs_2_low_high=[predictor.get_sub_rate_diffusion_beta(GPA(mu,mu*10, pop_size=int(N_e))) for mu in muts]

rates_dict={'high->low':subs_2_high_low,'low->high':subs_2_low_high,'low->low':subs_2_neutral}

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

cats=rates_dict.keys()
ratios=[]
for cat in cats:
    slim_subs=slim_dict[cat]
    diffusion_subs=rates_dict[cat]
    if cat=='high->low':
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
plt.show()