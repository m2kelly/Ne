# Pipeline that estimates an effective population size based on the effect of parallel mutations

  

**This pipeline requires an Nvidia cuda enabeled gpu**

  
  

## Inputs:

1. This pipeline takes as input two files (pickeled python), one for mutation counts per bin (usually a 100kb length) and one for the corresponding target size. Both are trinucleotide based contexts.

    Each of the two pickled files is a dictionaries. Keys are chromosomes, i.e., 'chr1', 'chr2', ..'chr22'; and values are pandas dataframes where the index is the mutation type, i.e., 'AAA->C', 'AAA->G', .. 'TTT->G', and columns are the index of the bin in the genome, i.e, 1 for the first 100kb window, 2 for the second, and so on.

  

    Put both files in a directory, and name them according to the follwoing scheme:

        f'whole_genome_dict_{name}_unsmoothed_muts.pkl' for the mutation counts file
        f'whole_genome_dict_{name}_unsmoothed_occs.pkl' for the target size file

    and provide the name (the value between the curly brackets) as an input

2. The directory where the two files (Inputs number 1) are located.
3. The estimated number of generations of the studied period.

## Usage:

1. Put the package in a directory and import it

    Protip: you can put it in the python directory of liberaries and import it as any other library, e.g., like import pandas as pd

2. Create a pipeline object: 

    example: 

        p = Pipeline(name='bins_after_defense', directory='../../', generations=160000)
3. Run the pipeline:
    example:
        p.run_until_min_error(step=0.01, CpG_remove_percentage=0.0)

All in all, the pipeline could be run as:
   
    from popsize_pipeline import Pipeline
    p = Pipeline(name='name_of_the_run', directory='path/to/directory', generations=160000)
    p.run_until_min_error()

## How it works?

1. The pipeline starts by choosing the best smoothing window; it tries smoothing windows from 1 to a 100 and chooses the one that maximises the correlation between CpG transitions and other mutations. This is done through the module "BestSmoothing"

2. Then the pipeline chooses a cohort of non-CpG transitions that correlates the best with CpG transitions. This is done through the module "BestNonCpGCandidates"

3. Then it divides bins into quantiles and checks if there is a signal of parallel mutations. It also chooses the size of the first bin where methylated CpGs should be located. This is done through the module "ReccurenceVectors".

4. It estimates the substitution rate per generation from the observed subsitutions through a markov chain process in the module "MarkovInfSites". Note that this step is done on the GPU.

5. It then calculates the population size that best corrects for the effect of recurrence identified at the previous step. The population size is estimated in the module "PopSizeCalculator"

6. The pipeline removes the least mutable percentage of the bins and repeat the steps 1-5. For each cycle, the error (the bias between CpG and non-CpG quantiles) determins the goodness of the correction. The pipeline keeps repeating steps 1-5 until the error stops decreasing. The population size with the least error is selected as "The" population size.

## Output
1. A logs file:

    The logs file holds all the logs of the pipeline. For each cycle, it logs the chosen smoothing window size, the cohort of mutations selected as the cohort correlating the best with CpG transitions, the number of remaining smoothed bins after filteration (logged as len of cpg vector), the overall substitution rates divided by the number of generations and the corrected substitution rate per generation after the markov process correction, the estimated mutation rates after correcting for parallel mutations, each used population size (thorugh the search) and the erorr of each one, and the best estimated population size.

2. A directory for each cycle with a different removed percentage of the least mutable bins. The rest of the outputs are located inside these directories.

3. A heat map of the Kendall's tau correlation coffecients of CpG transition mutations with each other (with a smoothing window size of 100). Manually check that there are no outliers.

![Example heatmap](https://codeberg.org/hossam26644/NeParallel-estimator/raw/branch/master/plots_for_readme/max_corr_heatmap.png)

4. A graph of the correlation between CpG transitions and the rest of mutations for smoothing window sizes of 1-100. The window size with the maximum correlation between the two values is selected as the best smoothing window for this cycle.

![Example smoothing range](https://codeberg.org/hossam26644/NeParallel-estimator/raw/branch/master/plots_for_readme/smoothing_range_plot.png)

5. A graph of the difference between the correlation within the CpG transitions group and the correlation within the rest of mutations. **Legacy!!** it is not used further down the pipeline
of
https://codeberg.org/hossam26644/NeParallel-estimator/raw/branch/master/plots_for_readme/smoothing_range_plot.png
6. A file of the tried combinations of non-CpG transitions and their correlation with CpG transitions: best_correlations_smoothed_{run-name}.csv. The top raw in this list should have the chosen cohort of non-CpG transitions used downstream the analyis for this cycle.

7. A box plot per each tried first quantile size. The box plots are the observed substitution rates for bins making each quantile. Grouped CpG transitions in red and grouped choosen non-CpG transitions in blue. 

![Example box plot](https://codeberg.org/hossam26644/NeParallel-estimator/raw/branch/master/plots_for_readme/boxes_plot.png)

8. Remaining bias between the median of the mutation rates for each quantile for each tried population size. The population size with the least error is choosen as the best population size.

![Example errors](https://codeberg.org/hossam26644/NeParallel-estimator/raw/branch/master/plots_for_readme/error_points_log.png)


8. Medians of the observed substitution rates for each quantile and the mutation rates (after correcting with the best population size)








   
