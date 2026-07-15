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

class Regress_subs_playground(Operations):
    def __init__(self, name, directory, cpgs=None, non_cpg_pool=None, collapse=True,
                 muts_dict_raw=None, occ_dict_raw=None, cpg_remove_percentage=0, prefix='',
                 operations=None):


        Operations.__init__(self, name=name, directory=directory, collapse=collapse,
                            muts_dict_raw=muts_dict_raw, occ_dict_raw=occ_dict_raw,
                            cpg_remove_percentage=cpg_remove_percentage, prefix=prefix,
                            operations=operations, cpgs=cpgs, non_cpg_pool=non_cpg_pool)
        
        
        
    
        self.cpg_remove_percentage=cpg_remove_percentage
        if cpgs is None:
            all_cpgs=es.mutation.get_cpg_muts()
            self.cpg_muts = [x for x in all_cpgs if str(x) in self.occ_dict_raw['chr1'].index]
            print(self.cpg_muts)
        if non_cpg_pool is None:
            self.non_cpg_pool = [m for m in es.get_mut_obj_list() if str(m) in self.occ_dict_raw['chr1'].index]
            self.non_cpg_pool = [m for m in self.non_cpg_pool if m not in self.cpg_muts]

        '''
        if cpgs is None:
            all_cpgs = es.mutation.get_cpg_muts()
            self.cpg_muts = [x for x in all_cpgs if str(x) in self.occ_dict_CpGfiltered['chr1'].index]
        else:
            self.cpg_muts = [c for c in cpgs if str(c) in self.occ_dict_CpGfiltered['chr1'].index]

        if non_cpg_pool is None:
            self.non_cpg_muts = [m for m in es.get_mut_obj_list() if str(m) in self.occ_dict_CpGfiltered['chr1'].index]
            self.non_cpg_muts = [m for m in self.non_cpg_muts if m not in self.cpg_muts]
        else:
            self.non_cpg_muts = [m for m in non_cpg_pool if str(m) in self.occ_dict_CpGfiltered['chr1'].index]
            self.non_cpg_muts = [m for m in self.non_cpg_muts if m not in self.cpg_muts]

        '''
    @staticmethod
    def regress_df(df):
        
        #order bins by mutability, model expected muts across bins with log offset for expsore,
        #collapse categoires           
        
        #---------------------------
        # log offset
        # -----------------------------------
        df["log_exposure"] = np.log(df["exposure"])

        # -----------------------------------
        # Negative binomial regression
        # -----------------------------------
        
        #C makes data catgeorical C(mut_bins)"
        model = smf.glm(
            formula="counts ~ C(mut_bins)",
            data=df,
            family=sm.families.NegativeBinomial(),
            offset=df["log_exposure"]
        ).fit()
        
        #print(model.summary())

        return model

    def run_linear_regress():
        return

    def run_cpg_all_non_cpg(self):
        cpg_df=self.generate_collapsed_regress_df(self.cpg_muts)
        non_cpg_df=self.generate_collapsed_regress_df(self.non_cpg_pool)
        
        # -----------------------------------
        # Keep only shared bins
        # -----------------------------------
        merged = non_cpg_df.merge(
            cpg_df[['chrom', 'bin', 'rate']],
            on=['chrom', 'bin'],
            suffixes=('_noncpg', '_cpg'),
            how='inner'
        )

        #before quantiling
        plt.scatter(merged['rate_noncpg'],merged['rate_cpg'])
        plt.show()
        # -----------------------------------
        # Rank bins independently
        # -----------------------------------
        merged['rank_noncpg'] = merged['rate_noncpg'].rank(method='average')
        merged['rank_cpg'] = merged['rate_cpg'].rank(method='average')

        # -----------------------------------
        # Rank disagreement (spearman residuals)
        # -----------------------------------
        merged['rank_diff'] = np.abs(
            merged['rank_noncpg'] - merged['rank_cpg']
        )
        print(merged['rank_diff'])
        # -----------------------------------
        # Remove outliers
        # e.g. top 1% most discordant bins
        # -----------------------------------
        '''
        cutoff = merged['rank_diff'].quantile(0.8)
        #cutoff=50
        merged = merged[
            merged['rank_diff'] < cutoff
        ].copy()
        '''

        print(f"Remaining bins after filtering: {len(merged)}")

        # -----------------------------------
        # Quantile bins using non-CpG rate
        # -----------------------------------
        merged['mut_bins'] = pd.qcut(
            merged['rate_noncpg'],
            q=5,
            duplicates='drop',
            labels=False
        )

        # -----------------------------------
        # Build final dfs
        # -----------------------------------
        non_cpg_df=non_cpg_df.merge(merged[['chrom','bin','mut_bins']],on=['chrom','bin'])
        cpg_df=cpg_df.merge(merged[['chrom','bin','mut_bins']],on=['chrom','bin'])
        
        non_cpg_df.sort_values(by=['chrom','bin'],inplace=True)
        cpg_df.sort_values(by=['chrom','bin'],inplace=True)
        print(non_cpg_df)

        non_cpg_model=self.regress_df(non_cpg_df)
        print(cpg_df)
        cpg_model=self.regress_df(cpg_df)

        plt.figure()
        pred_df = pd.DataFrame({
            'mut_bins': cpg_df['mut_bins']
        })

        # predicted cpg counts
        pred_cpg_counts = cpg_model.predict(
            pred_df,
            offset=cpg_df['log_exposure']
        )

        pred_df = pd.DataFrame({
            'mut_bins': non_cpg_df['mut_bins']
        })
        # predicted non-cpg counts
        pred_non_cpg_counts = non_cpg_model.predict(
            pred_df,
            offset=non_cpg_df['log_exposure']
        )
        # predicted rates
        x = pred_non_cpg_counts / non_cpg_df['exposure']
        y = pred_cpg_counts / cpg_df['exposure']

        plt.scatter(x,y,label='regressed binned')
        plt.xlabel('non cpg sub')
        plt.ylabel('cpg subs')
        
        mean_cpg_df=cpg_df[['rate','mut_bins']].groupby('mut_bins').mean().reset_index()
        mean_non_cpg_df=non_cpg_df[['rate','mut_bins']].groupby('mut_bins').mean().reset_index()
        x = mean_non_cpg_df['rate']
        y = mean_cpg_df['rate'] 
        plt.scatter(x,y,label='raw binned')

        #plt.plot(x,x,label='y=x')
        plt.legend()
        plt.savefig('/home/dweghorngroup/regression_options/regress_subs.png')

        #without regression
        x = non_cpg_df['rate']
        y = cpg_df['rate']
        plt.scatter(x,y,label='all points')

        mean_cpg_df=cpg_df[['rate','mut_bins']].groupby('mut_bins').mean().reset_index()
        mean_non_cpg_df=non_cpg_df[['rate','mut_bins']].groupby('mut_bins').mean().reset_index()
        x = mean_non_cpg_df['rate']
        y = mean_cpg_df['rate'] 
        plt.scatter(x,y,label='binned')

        plt.xlabel('non cpg sub')
        plt.ylabel('cpg subs')
        plt.legend()
        plt.savefig('/home/dweghorngroup/regression_options/pre_regress_subs.png')


    def generate_collapsed_regress_df(self,group_labels):
             #run negative binomial regression on cpg categories
        #muts dict, keys= chrom, value=df ith index=mut categories, columns =bins along chrom
        rows=[]
        for chrom, occ_df in self.occ_dict_raw.items():
            muts_df = self.muts_dict_raw[chrom]
            # exposure/opportunities per bin
            occ_vector = occ_df.loc[[str(mut) for mut in group_labels]].sum()
            # mutation counts per bin
            mut_vector = muts_df.loc[[str(mut) for mut in group_labels]].sum()
            for bin_name in occ_vector.index:
                occ = occ_vector.loc[bin_name]
                mut = mut_vector.loc[bin_name]
                if not np.isfinite(mut):
                    continue
                if not np.isfinite(occ):
                    continue
                if occ <= 0:
                    continue
                rows.append({
                    "chrom": chrom,
                    "bin": bin_name,
                    "counts":  mut,
                    "exposure": occ
                })

        df = pd.DataFrame(rows)
        #order bins by mutability, model expected muts across bins with log offset for expsore,
        #collapse categoires           
        df['rate']=df['counts']/df['exposure']
        return df


    def regress_cpg_to_non_cpg(self):
        cpg_df=self.generate_collapsed_regress_df(self.cpg_muts)
        non_cpg_df=self.generate_collapsed_regress_df(self.non_cpg_pool)
        
        # -----------------------------------
        # Keep only shared bins
        # -----------------------------------
        merged = non_cpg_df.merge(
            cpg_df,
            on=['chrom', 'bin'],
            suffixes=('_noncpg', '_cpg'),
            how='inner'
        )

        # -----------------------------------
        # Rank bins independently
        # -----------------------------------
        merged['rank_noncpg'] = merged['rate_noncpg'].rank(method='average')
        merged['rank_cpg'] = merged['rate_cpg'].rank(method='average')

        # -----------------------------------
        # Rank disagreement (spearman residuals)
        # -----------------------------------
        merged['rank_diff'] = np.abs(
            merged['rank_noncpg'] - merged['rank_cpg']
        )
        print(merged['rank_diff'])
        # -----------------------------------
        # Remove outliers
        # e.g. top 1% most discordant bins
        # -----------------------------------
        cutoff = merged['rank_diff'].quantile(0.95)

        #cutoff=50
        merged = merged[
            merged['rank_diff'] < cutoff
        ].copy()

        print(f"Remaining bins after filtering: {len(merged)}")

        merged["log_exposure_cpg"] = np.log(merged["exposure_cpg"]+ 1e-12)
        merged["log_rate_noncpg"] = np.log(merged['rate_noncpg']+ 1e-12)
        # -----------------------------------
        # Negative binomial regression
        # -----------------------------------
        model = smf.glm(
            formula="counts_cpg ~ log_rate_noncpg",
            data=merged,
            family=sm.families.NegativeBinomial(),
            offset=merged["log_exposure_cpg"]
        ).fit()

        print(model.summary())

        model=LinearRegression(fit_intercept=False)
        x=np.array(merged['rate_noncpg']).reshape(-1,1)
        y=np.array(merged['rate_cpg'])
        model.fit(x,y)

        print(model.score(x,y))

        return model
    
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

        keep_ranks=self.remove_based_on_rank(cpg_rate,non_cpg_rate,cutoff=cutoff,cpg_remove=0.0) #removing after
        #restrict to keep ranks
        cpg_rate=cpg_rate.loc[keep_ranks]
        non_cpg_rate=non_cpg_rate.loc[keep_ranks]

        sorted_indexes=list(self.sort_by_cpg_non_cpg_random(cpg_rate,non_cpg_rate))
        
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
        plt.show()



    
    def remove_based_on_rank(self,cpg_vector,non_cpg_vector,cutoff=0.7,cpg_remove=0.15):
        '''
        output indices where cpg and non cpg rank are within cutoff of each other '''

        
        
        if cpg_remove>0:
            #remove least mutable bins
            sorted_indexes=list(self.sort_by_cpg_non_cpg_random(cpg_vector,non_cpg_vector))
            to_remove_cpg_index=sorted_indexes[:int(len(sorted_indexes)*cpg_remove)]
            
            #remove least mutable cpg bins
            no_bins_remove=int(len(cpg_vector)*cpg_remove)
            to_remove_cpg_index=cpg_vector.sort_values(ascending=True).iloc[:no_bins_remove].index
            
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
        plt.show()

        plt.scatter(cpg_vector,rank_diff,alpha=0.01)
        plt.xlabel('CpG Rate')
        plt.ylabel('Rank Difference')
        plt.title(f'Discordance Between Cpg and Non-Cpg Rates, cpg_remove={cpg_remove}')
        plt.axhline(cut, color='red', linestyle='--', label=f'{cutoff*100:.1f}th Percentile Cutoff')
        plt.legend()
        plt.show()
        
        return keep_indices

    @staticmethod
    def find_linear_breakpoint(x, y, min_points=10000,jumps=1000):
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


        plt.plot(sizes,R_sqr)
        plt.xlabel('number of points included')
        plt.ylabel('R² of linear fit')
        plt.title('Finding linear breakpoint')
        plt.axvline(best_idx, color='red', linestyle='--', label=f'Best breakpoint at {best_idx} points')
        plt.axvline(best_knee_idx, color='blue', linestyle='--', label=f'Best knee at {best_knee_idx} points')
        plt.legend()
        plt.show()



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
        


    @staticmethod
    def bin_neighbours(dict, window_size=5):
        CHROMS = [f'chr{str(i)}' for i in range(1, 23)] + ['chrX', 'chrY']

        global_df = pd.DataFrame()
        last_col = 0
        for chrom in CHROMS:
            if chrom not in dict: continue
            df = dict[chrom]
            
            #bin within colos
            #gives overlapping windows, so each bin is sum of itself and 2 bins on either side, centered on itself
            #df = df.rolling(window=windows, center=True,axis=1).sum()
            
            #collapse to non overlapping windows
            bins=len(df.columns)
            new_df=pd.DataFrame()
            for i in range(0,bins,window_size):
                #sum current column to next
                
                new_df[i//window_size]=df.iloc[:,i:i+window_size].sum(axis=1)
              
            
            
            df.columns = [i + last_col for i in range(1,len(df.columns)+1)]
            if not df.empty:
                last_col = df.columns[-1]
                global_df = pd.concat([global_df, df], axis=1)

        return global_df


    def regress_to_choose(self, cpg_labels=None, non_cpg_labels=None):    

        
        #convert to vectors
        #extract cpg and no cpg and collapse? 
        #divide by non cpgs vectors

        #convert dict with chrom: df index=mut categories, columns=windows, to concatenated df
        #concatentae+rename
        #now index =category
        muts_vector=es.rename_cols(self.muts_dict_raw)
        occ_vector=es.rename_cols(self.occ_dict_raw)
        #muts_vector=self.bin_neighbours(self.muts_dict_raw)
        #occ_vector=self.bin_neighbours(self.occ_dict_raw)


        #remove indices where total occ across all catgories is 
        #less than 100k (as 100k bins x 3 expected entries)
        plt.hist(occ_vector.sum(axis=0), bins=50)
        plt.xlabel('total occ across all categories per bin')
        plt.ylabel('frequency')
        plt.show()
        mask = occ_vector.sum(axis=0) >= 100000
        occ_vector = occ_vector.loc[:, mask]
        muts_vector = muts_vector.loc[:, mask]
        

        print(muts_vector)
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

        no_rank_mismatch=self.remove_based_on_rank(cpg_rate,non_cpg_rate,cutoff=0.8)
            


        #sorts by second parameter, here cpg rate and fits y~x
        x_idx,max_cpg =self.find_linear_breakpoint(non_cpg_rate,cpg_rate)
        print(f'linear breakpoint at cpg rate, index: {x_idx}')
        

        #optional find knee of cpg graph, when non longer linear increase with pooled non cpgs,
        #then use these bins for picking
        kn = KneeLocator(cpg_rate, non_cpg_rate, curve='concave', direction='increasing')
        pooled_knee = kn.knee
        print(f'pooled knee: {pooled_knee}')
        plt.scatter(non_cpg_rate, cpg_rate)
        plt.xlabel('non cpg pooled rate')
        plt.ylabel('cpg pooled rate')
        plt.axvline(pooled_knee, color='red', linestyle='--', label=f'Knee at {pooled_knee:.2e}')
        plt.axhline(max_cpg, color='blue', linestyle='--', label=f'Linear breakpoint at {x_idx:.2e}')
        plt.legend()
        plt.show()


        
        #max_non_cpg_mask=non_cpg_rate <= max_non_cpg
        #cpg_rate = cpg_rate[cpg_rate <= max_cpg]
        


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
            mask &= (cpg > 0)
            mask &= (non_cpg > 0)
            cpg = cpg[mask]
            non_cpg = non_cpg[mask]
            non_cpg_occs = non_cpg_occs[mask]
            cpg_occs = cpg_occs[mask]
            no_rank_mismatch=self.remove_based_on_rank(cpg,non_cpg,cutoff=0.8)
            cpg=cpg[no_rank_mismatch]
            non_cpg=non_cpg[no_rank_mismatch]
            cpg_occs=cpg_occs[no_rank_mismatch]
            non_cpg_occs=non_cpg_occs[no_rank_mismatch]

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
            plt.show()

            plt.scatter(x,np.array(cpg_occs)/np.array(non_cpg_occs),alpha=0.2)
            plt.title(f'{non_cpg_label}, coef={model.coef_[0]:.4f}, score={model.score(x,y):.4f}, {len(x)} windows')
            #plt.plot(x,y_pred)
            plt.xlabel(f'non cpg rate {non_cpg_label}')
            plt.ylabel('cpg occs/non cpg occs')
            plt.show()
            
            coefficients.append(model.coef_[0])
            R_sqrs.append(model.score(x,y))

        df=pd.DataFrame({
            'non_cpg_label': non_cpg_labels,
            'coef': coefficients,
            'R_squared': R_sqrs
        }).sort_values(by='R_squared',ascending=False)
        print(df)
        df.to_csv(f'regression_results.csv',index=False)

        return model
