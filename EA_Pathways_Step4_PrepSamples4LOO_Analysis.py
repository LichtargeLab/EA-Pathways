#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import pandas as pd
import numpy as np
import time

#v3 - Nested lists of groups, genes, and EA scores (includes synon SNV)
#build two sets of nested lists, one with sig genes and one without sig genes

def PrepSamples4LOO_Analysis(sample_input_genes, all_groups_genes_unique, nonsyn_errored_genes, groups_input, sample_input_gene_action_matrix, sig_single_genes_lst, txt_summary_location):

    #first part of function - prep groups with EA scores for LOO analysis
    start=time.time()
    overlapping_sample_and_group_genes = set(sample_input_genes).intersection(all_groups_genes_unique)
    groups_input_id_lst = list(groups_input[0])

    groups_with_genes_and_EA_scores_lst = []
    groups_with_noSigGenes_and_EA_scores_lst = []
    genes_in_cohort_and_groups_with_mutation = []
    nonsyn_error_genes_in_cohort_and_groups = []
    #total_variants_in_group = []

    for row_outer in range(groups_input.shape[0]):
        #omit from analysis any groups with only one gene
        if len(list(groups_input.iloc[row_outer, 2:groups_input.shape[1]].dropna())) == 1:
            pass
        else:
            group_lst_all = []
            group_lst_noSigGenes = []
            group_lst_error_genes = []
            #group_lst_variants = []

            group_name = groups_input.iloc[row_outer,0]
            group_lst_all.append(group_name)
            group_lst_noSigGenes.append(group_name)
            #group_lst_variants.append(group_name)
            #group_lst_error_genes.append(group_name)

            group_genes = list(groups_input.iloc[row_outer, 2:groups_input.shape[1]].dropna())

            for gene in group_genes:
                gene_with_EA_lst = []
                gene_with_EA_lst.append(gene)

                for row_inner in range(sample_input_gene_action_matrix.shape[0]):
                    if gene == sample_input_gene_action_matrix[row_inner,0]:
                        gene_ea_scores = sample_input_gene_action_matrix[row_inner, 1:sample_input_gene_action_matrix.shape[1]]
                        gene_ea_scores = [x for x in gene_ea_scores if x != None]
                        gene_with_EA_lst.extend(gene_ea_scores)

                        if len(gene_ea_scores) != 0:
                            genes_in_cohort_and_groups_with_mutation.append(gene)
                        else:
                            pass

                group_lst_all.append(gene_with_EA_lst)

                if gene in sig_single_genes_lst:
                    pass
                else:
                    group_lst_noSigGenes.append(gene_with_EA_lst)

                if gene in nonsyn_errored_genes:
                    group_lst_error_genes.append(gene)
                else:
                    pass

            groups_with_genes_and_EA_scores_lst.append(group_lst_all)
            groups_with_noSigGenes_and_EA_scores_lst.append(group_lst_noSigGenes)
            nonsyn_error_genes_in_cohort_and_groups.append(group_lst_error_genes)

    #second part of function - generate summary matrix to collect analysis results
    #v3 - generate summary matrix prior to LOO analysis, synon SNV included in count

    #create lists to generate summary matrix
    group_names_lst = []
    group_length_lst = []
    group_genes_with_EAscores_lst = [] #v3 count includes synon SNV
    group_number_genes_with_EAscores_lst = [] #v3 count includes synon SNV
    group_sig_single_genes_lst = []
    group_total_variants_in_cohort_lst = []

    for group in groups_with_genes_and_EA_scores_lst:
        group_name = group[0]
        group_names_lst.append(group_name)

        group_length = len(group) - 1
        group_length_lst.append(group_length)

        group_genes_with_EAscores = []
        group_sig_genes = []
        group_variants = []

        for gene in group[1:]:
            if len(gene) == 1:
                pass
            else:
                group_genes_with_EAscores.append(gene[0])
                gene_variants = gene[1:len(gene)]
                gene_variants = [x for x in gene_variants if x != 'synon']
                group_variants.append(len(gene_variants))

            if gene[0] in sig_single_genes_lst:
                group_sig_genes.append(gene[0])
            else:
                pass

        group_genes_with_EAscores_lst.append(group_genes_with_EAscores)
        group_number_genes_with_EAscores_lst.append(len(group_genes_with_EAscores))
        group_sig_single_genes_lst.append(group_sig_genes)
        group_total_variants_in_cohort_lst.append(sum(group_variants))

    summary_matrix = pd.DataFrame()
    summary_matrix['group_name'] = group_names_lst
    summary_matrix['number_group_genes'] = group_length_lst

    group_sig_single_genes_lst_lengths = [len(x) for x in group_sig_single_genes_lst]
    functional_group_size_lst = [a - b for a, b in zip(group_length_lst, group_sig_single_genes_lst_lengths)]
    #summary_matrix['functional_group_size'] = functional_group_size_lst

    group_error_genes_lst_lengths = [len(x) for x in nonsyn_error_genes_in_cohort_and_groups]
    functional_group_size_lst2 = [a - b for a, b in zip(functional_group_size_lst, group_error_genes_lst_lengths)]
    summary_matrix['functional_group_size'] = functional_group_size_lst2
    summary_matrix['group_errored_genes'] = nonsyn_error_genes_in_cohort_and_groups


    summary_matrix['number_group_genes_with_EAscores'] = group_number_genes_with_EAscores_lst
    summary_matrix['group_genes_with_EAscores'] = group_genes_with_EAscores_lst
    summary_matrix['group_sig_genes'] = group_sig_single_genes_lst
    summary_matrix['total_group_variants'] = group_total_variants_in_cohort_lst

    text_file = open(txt_summary_location, 'w')
    text_file.write('Summary of Prepping Input Samples into Biological Groups for LOO Analysis' + '\n' + '\n')
    text_file.write('Number of biological groups: ' + str(len(groups_input_id_lst)) + '\n')
    text_file.write('Number of unique genes in biological groups: ' + str(len(all_groups_genes_unique)) + '\n')
    text_file.write('Number of unique genes with at least one mutation in input samples: ' + str(len(sample_input_genes)) + '\n')
    text_file.write('Overlap of unique sample input genes and unique biological groups genes: ' +
          str(len(overlapping_sample_and_group_genes)) + '\n')
    text_file.write('Number of genes in input samples and groups with >= 1 EA annotation: ' +
          str(len(set(genes_in_cohort_and_groups_with_mutation))) + '\n' + '\n')
    text_file.write('Time to prep sample for LOO analysis and generate summary matrix: ' + str(time.time() - start) + '\n')
    text_file.close()

    return summary_matrix, groups_with_noSigGenes_and_EA_scores_lst

