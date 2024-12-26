# Pipeline that estimates an effective population size based on the effect of parallel mutations

  

**This pipeline requires an Nvidia cuda enabeled gpu**

  
  

Inputs:

1. This pipeline takes as input two files (pickeled python), one for mutation counts per bin (usually a 100kb length) and one for the corresponding target size. Both are trinucleotide based contexts.

    Each of the two pickled files is a dictionaries. Keys are chromosomes, i.e., 'chr1', 'chr2', ..'chr22'; and values are pandas dataframes where the index is the mutation type, i.e., 'AAA->C', 'AAA->G', .. 'TTT->G', and columns are the index of the bin in the genome, i.e, 1 for the first 100kb window, 2 for the second, and so on.

  

    Put both files in a directory, and name them according to the follwoing scheme:

        f'whole_genome_dict_{name}_unsmoothed_muts.pkl' for the mutation counts file
        f'whole_genome_dict_{name}_unsmoothed_occs.pkl' for the target size file

    and provide the name (the value between the curly brackets) as an input

2. The directory where the two files (Inputs number 1) are located.
3. 

   
