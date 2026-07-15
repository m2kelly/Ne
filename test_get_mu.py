'''
idea
input mut rate, apply get subs then solve 12 equatinos to get mu back'''

#run with python -m NeParallel.test_get_mu.py

import numpy as np
from .pop_size import PopSizeCalculator
from scipy.optimize import brute
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
from .base_operations import Operations
from .pop_size import Predictor
from .pop_size import GPA

mu=1e-8
N_e=int(1e5)
mu_matrix={'A':{'C':mu,'G':mu,'T':mu},
           'C':{'A':10*mu,'G':mu,'T':mu},
           'G':{'A':mu,'C':mu,'T':mu},
           'T':{'A':mu,'C':mu,'G':mu}}

predictor = Predictor()
point=GPA(0,0, pop_size=N_e)
sub_matrix=predictor.get_sub_rate_diffusion_4_allele_renewal(point, mu_matrix)
print(sub_matrix)

recovered_mu_matrix=predictor.get_mu(point,sub_matrix,'','',return_matrix=True)
print(recovered_mu_matrix)
#then extract mu matrix back from sub
