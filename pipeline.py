from copy import deepcopy

import numpy as np
from .pop_size import PopSizeCalculator
from .base_operations import Operations
from .best_non_cpg_candidates import BestNonCpGCandidatesBeam
from .regress_subs import RegressSubs
from .recurrence_vectors import ReccurenceVectors
from . import essentials as es
import os
import shutil
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.ndimage import gaussian_filter1d
import pandas as pd

'''
creating a simplified pipeline
remove bins where occ<100k
pick best non cpg categories using beam search


remove least mutable bins, randommly choosing cppg vs non cpg
1 remove cpg bins based on rank of cpg vs non cpg mutability 

bin remaining bins into 20 bins
calculate reucrrence vectors from 20 bins
find N by regresing cpg vs no cpg recurrence vectors
'''

class Pipeline(Operations):

    def __init__(self, name, directory, cpgs=None, non_cpg_pool=None, smoothing_range=None, collapse=True, generations=None ,operations=None):

        Operations.__init__(self, name=name, directory=directory, collapse=collapse, operations=operations)

        self.cpgs = cpgs
        self.non_cpg_pool = non_cpg_pool
        all_cpgs = es.mutation.get_cpg_muts()
        self.cpg_muts = [x for x in all_cpgs if str(x) in self.occ_dict_raw['chr1'].index]
        
        #TESTING WITH ONE CATEORY
        self.cpg_muts=[self.cpg_muts[0]]
        self.cpgs=self.cpg_muts
        
        self.smoothing_range = smoothing_range
        self.best_window = None
        self.best_candidates = None

        self.cpg_subs = None
        self.non_cpg_subs = None
        self.cpg_subs_bckwrds = None
        self.non_cpg_subs_bckwrds = None

        self.generations = generations
        self.cutoff = 0.8
        self.clear_logs()

    
    
    def bin_vectors(self,beam,bins=5):
        '''
        bin the cpg and non cpg vectors based on mutability
        returns a list of lists of indicies for each bin, based on orginial occ dict raw indicies
        '''
        filtered_indices_ordered=beam.filter_dicts_low_mut_rank(self.occ_dict_raw,self.muts_dict_raw,non_cpg_labels=self.best_candidates,cpg_remove_percentage=self.cpg_remove_percentage,cutoff=self.cutoff)
        
        #filtered_indices_ordered=beam.filter_dicts_low_mut_ints_ranks(self.occ_dict_raw,self.muts_dict_raw,non_cpg_labels=self.best_candidates,remove_low=self.cpg_remove_percentage,cutoff=self.cutoff)
        
        #to remove using 2d gaussian filter
        #filtered_indices_ordered=beam.filter_dicts_2d_gaussian(self.occ_dict_raw,self.muts_dict_raw,cpg_labels=[str(x) for x in self.cpg_muts],non_cpg_labels=self.best_candidates,cpg_remove_percentage=self.cpg_remove_percentage,cutoff=self.cutoff)

        indices = np.array_split(filtered_indices_ordered, bins)

        return indices
       
   
    

    def plot_bins(self):
        '''
        plot binned rates of cpg vs non cpg, before recurrence correction'''
        
        muts_vector=es.rename_cols(self.muts_dict_raw)
        occ_vector=es.rename_cols(self.occ_dict_raw)
        cpg=[]
        non_cpg=[]
        cpg_labels=[str(x) for x in self.cpg_muts]
        non_cpg_labels=self.best_candidates

        cpg_muts = muts_vector.loc[cpg_labels].sum()/self.generations
        cpg_occs_copy = occ_vector.loc[cpg_labels].sum()

        non_cpg_muts = muts_vector.loc[non_cpg_labels].sum()/self.generations
        non_cpg_occs_copy = occ_vector.loc[non_cpg_labels].sum()
        
        #all indices, to plot raw rates
        filt_indices=list(self.indices[0])
        for indices in self.indices[1:]:
            filt_indices.extend(list(indices))

        #plt raw rates
        x=np.array(non_cpg_muts.loc[filt_indices]/non_cpg_occs_copy.loc[filt_indices]).reshape(-1,1)
        y=np.array(cpg_muts.loc[filt_indices]/cpg_occs_copy.loc[filt_indices])


        
        model=LinearRegression(fit_intercept=False)
        model.fit(x,y)
        y_pred=model.predict(x)
        plt.clf()
        plt.plot(x, y_pred, color='red', linewidth=2,label=f'R^2={model.score(x,y):.2f},coef={model.coef_[0]:.2f}')
        plt.scatter(x,y,alpha=0.1)
        plt.ylabel('CpG subs, before recurrence correction')
        plt.xlabel('Non-CpG subs, before recurrence correction')
        plt.legend()
        plt.savefig(f'{self.prefix}/raw_rates.png')
        
        for bin in self.indices:
            cpg.append(cpg_muts.loc[bin].sum()/cpg_occs_copy.loc[bin].sum())
            non_cpg.append(non_cpg_muts.loc[bin].sum()/non_cpg_occs_copy.loc[bin].sum())
        
        #linear regress
        x=np.array(non_cpg).reshape(-1,1)
        y=np.array(cpg)
        model=LinearRegression(fit_intercept=False)
        model.fit(x,y)
        y_pred=model.predict(x)
        plt.clf()
        plt.plot(x, y_pred, color='red', linewidth=2,label=f'R^2={model.score(x,y):.2f},coef={model.coef_[0]:.2f}')
        plt.scatter(x,y)
        plt.ylabel('CpG subs, before recurrence correction')
        plt.xlabel('Non-CpG subs, before recurrence correction')
        plt.legend()
        plt.savefig(f'{self.prefix}/binned_rates.png')
        
    @staticmethod
    def gaussian_smooth_series(s: pd.Series, sigma=1, mode="nearest") -> pd.Series:
        #‘nearest’ (a a a a | a b c d | d d d d)
        #The input is extended by replicating the last pixel
        # bins used =2*radius + 1
        return pd.Series(
            gaussian_filter1d(s.astype(float).to_numpy(), sigma=sigma, mode=mode,radius=4),
            index=s.index,
            name=s.name,
        )        

   
    def smooth_dict(self,dict):
        '''
        optional smoothing of the occ and mut dicts, to reduce noise in low mutability bins'''
        smoothed_dict={}
        for chr,df in dict.items():
            smoothed_dict[chr] = df.apply(self.gaussian_smooth_series,axis=1)
        return smoothed_dict
    
    @staticmethod
    def extract_non_cpg_per_context(mut):
        '''
        if using 4 allele model and want to extract all other non cpg mutations for a given cpg mutation, 
        this function will return all other non cpg mutations in the same trinucleotide context
        input cpg mut: trinuc->alt XCG->T'''
        BASES=['A','C','G','T']
        
        other_muts=[]
        trinuc=mut.tri
        left_flank=trinuc[0]
        right_flank=trinuc[2]
        ref=trinuc[1] #G
        alt=mut.base  #T
        for x in BASES:
            for y in BASES:
                if x==y:
                    continue
                if (x==ref) & (y==alt):
                    continue
                other_muts.append(es.mutation(left_flank+x+right_flank,y))
        print(f'Mutation: {mut}, Other Mutations: {other_muts}')
        return other_muts



    def run_pipeline(self,CpG_remove_percentage=0.0,cutoff=0.8):
        '''
        main calling of pipeline
        '''
        #self.plot_raw_rates()
        self.cutoff=cutoff
        

        if self.non_cpg_pool is None:
            self.non_cpg_pool = es.get_general_non_cpg_pool()

        #TESTING always use all 4 allele alt alleles per cpg category
        '''
        non_cpg_muts=[]
        for cpg in self.cpg_muts:
            non_cpg_muts=non_cpg_muts+self.extract_non_cpg_per_context(cpg)
        self.non_cpg_muts=[x if str(x) in self.occ_dict_raw['chr1'].index else x.get_rev_comp() for x in non_cpg_muts]
        print(f'Non CpG pool: {self.non_cpg_muts}')
        self.best_candidates=[str(x) for x in self.non_cpg_muts]
        print(f'Best candidates: {self.best_candidates}')
        self.cpg_non_cpg_dict={self.cpg_muts[i]:self.non_cpg_muts for i in range(len(self.cpg_muts))}
        '''
                        
        while CpG_remove_percentage<=0.75:
        
            self.write_logs(f'Running pipeline with CpG remove percentage: {CpG_remove_percentage}')
            directory = f'cpg_remove_percentage_{CpG_remove_percentage}/'
            self.prefix = directory
            try: os.mkdir(directory)
            except FileExistsError: pass
            #try:
            self.cpg_remove_percentage = CpG_remove_percentage
            self.write_logs('Getting best non cpg candidates')
            #choosing candidates by linear regression of each non cpg with cpg pooled
            
            #OPTIONAL smoothing of occ and muts dicts
            #self.muts_dict_raw=self.smooth_dict(self.muts_dict_raw)
            #self.occ_dict_raw=self.smooth_dict(self.occ_dict_raw)

            beam=BestNonCpGCandidatesBeam(name=self.name, directory=self.directory,
                                        best_smoothing=1,
                                        cpgs=self.cpgs, non_cpg_pool=self.non_cpg_pool,
                                        collapse=self.collapse,
                                        muts_dict_raw=self.muts_dict_raw,
                                        occ_dict_raw=self.occ_dict_raw,
                                        cpg_remove_percentage=self.cpg_remove_percentage,cutoff=self.cutoff,
                                        prefix=self.prefix, operations=self)
            
            
            
            if not self.best_candidates: 
            #choose by beam search
                self.cpg_non_cpg_dict={}
                #for cpg_mut in self.cpg_muts: #if want to regress seperatly per cpg,rather than mean across cpgs
                beam=BestNonCpGCandidatesBeam(name=self.name, directory=self.directory,
                                    best_smoothing=1,
                                    cpgs=self.cpg_muts, non_cpg_pool=self.non_cpg_pool,
                                    collapse=self.collapse,
                                    muts_dict_raw=self.muts_dict_raw,
                                    occ_dict_raw=self.occ_dict_raw,
                                    cpg_remove_percentage=self.cpg_remove_percentage,cutoff=self.cutoff,
                                    prefix=self.prefix, operations=self)
        
                non_cpg_muts = beam.get_best_candidates()  #returns list of mut objects
                #merging all and running one regression 
                self.cpg_non_cpg_dict={self.cpg_muts:non_cpg_muts}
                #if running regression per cpg-non-cpg pair and want to keep seperate
                #self.cpg_non_cpg_dict={self.cpg_muts[i]:best_candidates[i] for i in range(4)}
                
                self.write_logs(f'cpg_non_cpg_dict {self.cpg_non_cpg_dict}')
                self.non_cpg_muts=[x for x in self.cpg_non_cpg_dict.values()]
                
                self.best_candidates=[str(x) for x in self.non_cpg_muts]
                print(f'best candidates: {self.best_candidates}')

            else:
                self.non_cpg_muts=[x for x in self.non_cpg_pool if str(x) in self.best_candidates.values()] #mut objects for best candidates,used in recc vectors

            
            self.write_logs(f'cpg candidates {self.cpg_muts}')
            self.write_logs(f'Best non cpg candidates: {self.best_candidates}')
        
        
            
            self.write_logs('filtering and binning')

            self.indices=self.bin_vectors(beam,bins=100)
            #iniices is a list of lists of indicies for each bin
            #based on orginial occ dict raw indicies
            for bin in self.indices:
                print(len(bin))
            self.plot_bins()
            
            
    
            self.write_logs('running recurrence vectors')
            reccur=ReccurenceVectors(self.name, self.directory,
                                indices=self.indices,
                                non_cpgs=self.non_cpg_muts, cpgs=self.cpgs,
                                collapse=self.collapse,
                                muts_dict_raw=self.muts_dict_raw,
                                occ_dict_raw=self.occ_dict_raw,
                                cpg_remove_percentage=self.cpg_remove_percentage,
                                generations=self.generations,
                                prefix=self.prefix,
                                operations=self)

            #editted so c, nc etc are now dictionaries with lists of rates per 4 mut categories
            rates_dict, occs_dict  = reccur.get_mrkv_corrected_vctrs()

        
            self.write_logs('recurrence vectors calculated')
            self.write_logs(rates_dict)
            self.write_logs(rates_dict.keys())
            cpg_subs={key:value for key,value in rates_dict.items() if key in self.cpgs}
            non_cpg_subs={key:value for key,value in rates_dict.items() if key in self.non_cpg_muts}

            self.write_logs('running regressions to find N')
            popcalc = PopSizeCalculator(gens=self.generations, cpg_subs=cpg_subs,
                            non_cpg_subs=non_cpg_subs, cpg_subs_bckwrds=[],
                            non_cpg_subs_bckwrds=[],
                            cpg_occs=[],
                            non_cpg_occs=[], cpg_occs_bckwrds=[],
                            non_cpg_occs_bckwrds=[],
                            directory=self.prefix, operations=self)
            popcalc.cpg_non_cpg_dict=self.cpg_non_cpg_dict
            popcalc.rates_dict=rates_dict
            popcalc.calc_best_pop_regress_once()
            popcalc.plot_correction()
            self.write_logs(f'best pop:{popcalc.best_pop}, min error:{popcalc.min_error}')
            
            CpG_remove_percentage+=0.05
            self.best_candidates=None #reset for next round, to choose by beam search again
            
            '''
            except:
                self.write_logs(f'Error at CpG remove percentage: {CpG_remove_percentage}')
                CpG_remove_percentage+=0.05
                self.best_candidates=None #reset for next round, to choose by beam search again
            '''
        
