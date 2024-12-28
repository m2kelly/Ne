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

4. It then calculates the population size that best corrects for the effect of recurrence identified at the previous step. The population size is estimated in the module "PopSizeCalculator"

5. The pipeline removes the least mutable percentage of the bins and repeat the steps 1-4. For each cycle, the error (the bias between CpG and non-CpG quantiles) determins the goodness of the correction. The pipeline keeps repeating steps 1-4 until the error stops decreasing. The population size with the least error is selected as "The" population size.

## Output
1. A logs file:

    The logs file holds all the logs of the pipeline. For




   
