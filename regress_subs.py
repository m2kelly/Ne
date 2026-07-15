'''
script added by maria
idea, load subs + targets per bin and model using negative binomial or poisson regression
accounts for different noise levels
restrict to only non cpg mut categories with positive coefficients
do the same for backwards muts + cpg forward+backwards
extract 4 rates with cpg/non cpg
muts dict, keys= chrom, value=df ith index=mut categories, columns =bins along chrom 
'''
from sklearn.linear_model import LinearRegression
from . import essentials as es
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from .base_operations import Operations
import matplotlib.pyplot as plt
from kneed import KneeLocator

class RegressSubs(Operations):
    def __init__(self, name, directory, cpgs=None, non_cpg_pool=None, collapse=True,cutoff=0.8,
                 muts_dict_raw=None, occ_dict_raw=None, cpg_remove_percentage=0, prefix='',
                 operations=None):


        Operations.__init__(self, name=name, directory=directory, collapse=collapse,
                            muts_dict_raw=muts_dict_raw, occ_dict_raw=occ_dict_raw,
                            cpg_remove_percentage=cpg_remove_percentage, prefix=prefix,
                            operations=operations, cpgs=cpgs, non_cpg_pool=non_cpg_pool)
        
        self.cutoff=cutoff
        
    
        self.cpg_remove_percentage=cpg_remove_percentage
        if cpgs is None:
            all_cpgs=es.mutation.get_cpg_muts()
            self.cpg_muts = [x for x in all_cpgs if str(x) in self.occ_dict_raw['chr1'].index]
            print(self.cpg_muts)
        if non_cpg_pool is None:
            self.non_cpg_pool = [m for m in es.get_mut_obj_list() if str(m) in self.occ_dict_raw['chr1'].index]
            self.non_cpg_pool = [m for m in self.non_cpg_pool if m not in self.cpg_muts]

       

    def bin_and_plot(self,cpg_remove=0.15,bins=20,cpg_labels=None,non_cpg_labels=None,cutoff=0.9):
        muts_vector=es.rename_cols(self.muts_dict_raw)
        occ_vector=es.rename_cols(self.occ_dict_raw)
        

        mask = occ_vector.sum(axis=0) >= 100000
        occ_vector = occ_vector.loc[:, mask]
        muts_vector = muts_vector.loc[:, mask]
        

        
        if cpg_labels==None:
            cpg_labels=[str(x) for x in self.cpg_muts]
        if non_cpg_labels==None:
            non_cpg_labels=[str(x) for x in self.non_cpg_pool]
        print(cpg_labels)
        print(non_cpg_labels)

        #cpg pooled rate
        cpg_muts = muts_vector.loc[cpg_labels].sum()
        cpg_occs_copy = occ_vector.loc[cpg_labels].sum()
        cpg_rate = cpg_muts / cpg_occs_copy


        #non cpg pooled rate
        non_cpg_muts = muts_vector.loc[non_cpg_labels].sum()
        non_cpg_occs_copy = occ_vector.loc[non_cpg_labels].sum()
        non_cpg_rate = non_cpg_muts / non_cpg_occs_copy

        keep_ranks=self.remove_based_on_rank_and_low_mut(cpg_rate,non_cpg_rate,cutoff=cutoff,cpg_remove=0.0) #removing after
        #restrict to keep ranks
        cpg_rate=cpg_rate.loc[keep_ranks]
        non_cpg_rate=non_cpg_rate.loc[keep_ranks]

        sorted_indexes=list(self.sort_by_cpg_non_cpg_random(cpg_rate,non_cpg_rate,cpg_remove=cpg_remove))
        
        to_keep_more_mutable=sorted_indexes[int(len(sorted_indexes)*cpg_remove):]
        no_indexes=len(to_keep_more_mutable)
        print('remaining bins after low mutable remove',no_indexes)
        #remove least mutable cpg bins + bin based on order 
        bin_size=no_indexes//bins
        binned_cpg_rate=[]
        binned_non_cpg_rate=[]
        for i in range(0,no_indexes,bin_size):

            bin_indexes=to_keep_more_mutable[i:i+bin_size]
            if len(bin_indexes)<bin_size:
                continue
            print(len(cpg_muts.loc[bin_indexes]),len(cpg_occs_copy.loc[bin_indexes]),len(cpg_rate.loc[bin_indexes]))
            binned_cpg_rate.append(cpg_muts.loc[bin_indexes].sum()/cpg_occs_copy.loc[bin_indexes].sum())
            binned_non_cpg_rate.append(non_cpg_muts.loc[bin_indexes].sum()/non_cpg_occs_copy.loc[bin_indexes].sum())
        #linear regression
        x=np.array(binned_non_cpg_rate).reshape(-1,1)
        y=np.array(binned_cpg_rate)
        model=LinearRegression(fit_intercept=False).fit(x,y)
        y_pred=model.predict(x)

        plt.scatter(x,y)
        plt.plot(x, y_pred, color='red')

        plt.xlabel('non cpg sub rate')
        plt.ylabel('cpg sub rate')
        plt.title(f'cpg remove {cpg_remove}')
        



    
    def remove_based_on_rank_and_low_mut(self,cpg_vector,non_cpg_vector,cutoff=0.8,cpg_remove=0.15):
        '''
        output indices where cpg and non cpg rank are within cutoff of each other '''

        if cpg_remove>0:
            #remove least mutable bins
            sorted_indexes=list(self.sort_by_cpg_non_cpg_random(cpg_vector,non_cpg_vector))
            to_remove_cpg_index=sorted_indexes[:int(len(sorted_indexes)*cpg_remove)]
            
            #remove least mutable cpg bins
            #no_bins_remove=int(len(cpg_vector)*cpg_remove)
            #to_remove_cpg_index=cpg_vector.sort_values(ascending=True).iloc[:no_bins_remove].index
            
            cpg_vector=cpg_vector.drop(to_remove_cpg_index)
            non_cpg_vector=non_cpg_vector.drop(to_remove_cpg_index)    

        #test remove least mutable bins                                                                              
        # compute ranks
        cpg_rank = cpg_vector.rank(method='average')
        non_cpg_rank = non_cpg_vector.rank(method='average')

        # absolute rank difference
        rank_diff = np.abs(cpg_rank - non_cpg_rank)
        cut=rank_diff.quantile(cutoff)
        # keep indices within cutoff
        keep_indices = rank_diff[rank_diff <= cut].index
        #plot

        plt.hist(rank_diff, bins=50)
        plt.xlabel('absolute rank difference between cpg and non cpg rates')
        plt.ylabel('frequency')
        plt.title(f'Rank Discordance Between Cpg and Non-Cpg Rates, cpg_remove={cpg_remove}')
        plt.axvline(cut, color='red', linestyle='--', label=f'{cutoff*100:.1f}th Percentile Cutoff')
        plt.legend()
        
        return keep_indices

    
    def find_linear_breakpoint(self,x, y, min_points=10000,jumps=1000):
        """
        Find x where linear fit stops being good (first segment only).
        
        Returns:
            best_idx: index of breakpoint
            best_x: x value at breakpoint
        """

        x = np.asarray(x)
        y = np.asarray(y)

        # mask nans
        mask = np.isfinite(x)
        mask &= np.isfinite(y)
        x = x[mask]
        y = y[mask]


        # sort by y (important!)
        order = np.argsort(y)
        x = x[order]
        y = y[order]

        best_score = -np.inf
        best_idx = None

        sizes=[]
        R_sqr=[]
        for i in range(min_points, len(y) - 1,jumps):
            x_subset = x[:i].reshape(-1, 1)
            y_subset = y[:i]

            model = LinearRegression(fit_intercept=False)
            model.fit(x_subset, y_subset)

            score = model.score(x_subset, y_subset)  # R²

            sizes.append(i)
            R_sqr.append(score)


            if score > best_score:
                best_score = score
                best_idx = i

        #find knee of R² curve
        kn = KneeLocator(sizes, R_sqr, curve='concave', direction='decreasing')
        best_knee_idx = kn.knee

        plt.gcf()
        plt.plot(sizes,R_sqr)
        plt.xlabel('number of points included')
        plt.ylabel('R² of linear fit')
        plt.title('Finding linear breakpoint')
        plt.axvline(best_idx, color='red', linestyle='--', label=f'Best breakpoint at {best_idx} points')
        plt.axvline(best_knee_idx, color='blue', linestyle='--', label=f'Best knee at {best_knee_idx} points')
        plt.legend()
        plt.savefig(self.directory + 'linear_breakpoint.png')




        return best_knee_idx, y[best_knee_idx]
    
    @staticmethod
    #reindex these then remove again?
    def sort_by_cpg_non_cpg_random(cpg,non_cpg):
        print('chosing cpg, non cpg orders')
        '''
        sort cpg and non cpg vectors
        choose each positoin by randomly chossing next smallest cpg or next smappest non cpg
        avoiding sorting by only one, giving flick effect'''
        cpg_sorted = cpg.sort_values(ascending=True).index.tolist()
        non_cpg_sorted = non_cpg.sort_values(ascending=True).index.tolist()
        
        indexes=[]
        i=0
        j=0
        cpg_non_cpg_choices=[]
        for x in range(len(cpg)):
            choice = np.random.choice([0, 1])
            if choice==0:
                if i>=len(cpg_sorted): #only intems in j list left, so have to choose from there
                    choosen_index= non_cpg_sorted[j]
                    j+=1
                    cpg_sorted.remove(choosen_index) #remove index from other list to avoid choosing it again
                    cpg_non_cpg_choices.append(1)   #1 is added non cpg, 0 is added cpg
                else:
                    choosen_index=cpg_sorted[i] #chosen index is the next smallest cpg
                    i+=1   
                    non_cpg_sorted.remove(choosen_index) #remove index from other list to avoid choosing it again
                    cpg_non_cpg_choices.append(0)

            else:
                if j>=len(non_cpg_sorted): #only intems in i list left, so have to choose from there
                    choosen_index= cpg_sorted[i]
                    i+=1 
                    non_cpg_sorted.remove(choosen_index)
                    cpg_non_cpg_choices.append(0)
                else:
                    choosen_index=non_cpg_sorted[j]
                    j+=1
                    cpg_sorted.remove(choosen_index)
                    cpg_non_cpg_choices.append(1)

            indexes.append(choosen_index)
            
        #print(i,j, indexes, cpg_non_cpg_choices)
        return indexes
        


   

    def regress_to_choose(self, cpg_labels=None, non_cpg_labels=None):    

        
        #convert to vectors
        #extract cpg and no cpg and collapse? 
        #divide by non cpgs vectors

        
        muts_vector=es.rename_cols(self.muts_dict_raw)
        occ_vector=es.rename_cols(self.occ_dict_raw)
        
        #remove indices where total occ across all catgories is 
        #less than 100k (as 100k bins x 3 expected entries)
        plt.hist(occ_vector.sum(axis=0), bins=50)
        plt.xlabel('total occ across all categories per bin')
        plt.ylabel('frequency')
        
        mask = occ_vector.sum(axis=0) >= 100000
        occ_vector = occ_vector.loc[:, mask]
        muts_vector = muts_vector.loc[:, mask]
        print(f"Remaining bins after filtering low coverage: {len(occ_vector.columns)}")

        if cpg_labels==None:
            cpg_labels=[str(x) for x in self.cpg_muts]
        if non_cpg_labels==None:
            non_cpg_labels=[x for x in muts_vector.index if x not in cpg_labels] #as collapse trinucs

        #cpg pooled rate
        cpg_muts = muts_vector.loc[cpg_labels].sum()
        cpg_occs_copy = occ_vector.loc[cpg_labels].sum()
        cpg_rate = cpg_muts / cpg_occs_copy


        #non cpg pooled rate
        print(muts_vector.index)
        non_cpg_muts = muts_vector.loc[non_cpg_labels].sum()
        non_cpg_occs_copy = occ_vector.loc[non_cpg_labels].sum()
        non_cpg_rate = non_cpg_muts / non_cpg_occs_copy

        #remove cpg percentage of least mutable bins 
        #plus bins with big overall rank discordance
        keep_indices=self.remove_based_on_rank_and_low_mut(cpg_rate,non_cpg_rate,cutoff=self.cutoff,cpg_remove=self.cpg_remove_percentage)
        #then sort based on cpg and non cpg rates randomly to avoid sorting by only one and giving a flick effect
            
        non_cpg_rate=non_cpg_rate.loc[keep_indices]
        cpg_rate=cpg_rate.loc[keep_indices]


        #sorts by second parameter, here cpg rate and fits y~x
        x_idx,max_cpg =self.find_linear_breakpoint(non_cpg_rate,cpg_rate)
        print(f'linear breakpoint at cpg rate, index: {x_idx}')
        

        #optional find knee of cpg graph, when non longer linear increase with pooled non cpgs,
        #then use these bins for picking
        kn = KneeLocator(cpg_rate, non_cpg_rate, curve='concave', direction='increasing')
        pooled_knee = kn.knee
        print(f'pooled knee: {pooled_knee}')
        plt.figure()
        plt.gcf()
        plt.scatter(non_cpg_rate, cpg_rate)
        plt.xlabel('non cpg pooled rate')
        plt.ylabel('cpg pooled rate')
        plt.axvline(pooled_knee, color='red', linestyle='--', label=f'Knee at {pooled_knee:.2e}')
        plt.axhline(max_cpg, color='blue', linestyle='--', label=f'Linear breakpoint at {x_idx:.2e}')
        plt.legend()
        plt.savefig(self.directory + '/cpg_non_cpg_scatter.png')
        


        #run regression on each non cpg category 
        coefficients=[]
        R_sqrs=[]
        for non_cpg_label in non_cpg_labels:
            non_cpg_rate = muts_vector.loc[non_cpg_label]/occ_vector.loc[non_cpg_label]

            #align indices and remove non cpg when cpg above linear breakpoint (should already be done)
            common_index = cpg_rate.index.intersection(non_cpg_rate.index)
            cpg = cpg_rate.loc[common_index]
            non_cpg = non_cpg_rate.loc[common_index]

            #occs indexes not matching
            non_cpg_occs=non_cpg_occs_copy[common_index]
            cpg_occs=cpg_occs_copy[common_index]

            # mask nans
            mask = np.isfinite(cpg)
            mask &= np.isfinite(non_cpg)
            cpg = cpg[mask]
            non_cpg = non_cpg[mask]
            non_cpg_occs = non_cpg_occs[mask]
            cpg_occs = cpg_occs[mask]
            

            model=LinearRegression(fit_intercept=False)
            #fit_intercept=False
            x=np.array(non_cpg).reshape(-1,1)
            y=np.array(cpg)
            model.fit(x,y)
            y_pred=model.predict(x)
            plt.scatter(x,y,alpha=0.2)
            plt.title(f'{non_cpg_label}, coef={model.coef_[0]:.4f}, score={model.score(x,y):.4f}, {len(x)} windows')
            plt.plot(x,y_pred)
            plt.xlabel(f'non cpg rate {non_cpg_label}')
            plt.ylabel('all cpgs collapsed rate')
            
            coefficients.append(model.coef_[0])
            R_sqrs.append(model.score(x,y))

        df=pd.DataFrame({
            'non_cpg_label': non_cpg_labels,
            'coef': coefficients,
            'R_squared': R_sqrs
        }).sort_values(by='R_squared',ascending=False)
        print(df)
        df.to_csv(f'{self.directory}/regression_results.csv',index=False)
        
        return df
