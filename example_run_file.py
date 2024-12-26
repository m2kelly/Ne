''' Run from outside this directory
'''
from popsize_pipeline import Pipeline

p = Pipeline(name='bins_after_defense', directory='../../', generations=160000)
p.run_until_min_error(step=0.01, CpG_remove_percentage=0.0)
