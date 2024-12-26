from . import essentials as es

def get_h_non_cpgs_and_l_non_cpgs(muts_dict_raw, occ_dict_raw):
    all_muts = None; all_occs = None
    for chrom in es.CHROMS:
        if chrom not in muts_dict_raw: continue
        if all_muts is None:
            all_muts = muts_dict_raw[chrom].sum(axis=1)
            all_occs = occ_dict_raw[chrom].sum(axis=1)
        else:
            all_muts += muts_dict_raw[chrom].sum(axis=1)
            all_occs += occ_dict_raw[chrom].sum(axis=1)

    cpgs = [str(m) for m in es.mutation.get_cpg_muts()]
    sorted_matrix = all_muts/all_occs
    sorted_matrix = sorted_matrix.sort_values(ascending=False)
    sorted_matrix = sorted_matrix.dropna()
    mirrored_mutations  = [str(m) for m in sorted_matrix.index if es.mutation(label=m).is_mirrored()]


    sorted_matrix = sorted_matrix[~sorted_matrix.index.isin(cpgs)]
    sorted_matrix = sorted_matrix[~sorted_matrix.index.isin(mirrored_mutations)]

    sorted_matrix = sorted_matrix.dropna()

    non_cpg_transitions = sorted_matrix[sorted_matrix.index.str.contains(r'^.[CG].*[TA]$')]
    non_cpg_transitions = non_cpg_transitions[~non_cpg_transitions.index.str.contains('CG|GC')]

    least_mutable = non_cpg_transitions.min()
    non_cpg_transitions = non_cpg_transitions.sort_values(ascending=False)[:8]

    h_non_cpgs = [es.mutation(label=m) for m in non_cpg_transitions.index]

    #remove any mutation in the sorted matrix if it is more mutable than a non_cpg_transition
    sorted_matrix = sorted_matrix[sorted_matrix < least_mutable]
    #remove the highest 20% of the sorted matrix
    sorted_matrix = sorted_matrix[sorted_matrix < sorted_matrix.quantile(0.8)]
    l_non_cpgs = [es.mutation(label=m) for m in sorted_matrix.index]
    return h_non_cpgs, l_non_cpgs

