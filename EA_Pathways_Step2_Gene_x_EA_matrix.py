#!/usr/bin/env python
# coding: utf-8

# In[ ]:

# sample input genes used to build matrix: removed nonsyn SNV without EA scores, kept synon SNV

import numpy as np
import pandas as pd
import time

def generate_sample_gene_by_EAmatrix(user_defined_variants, sample_input, summary_txt_location):

    start = time.time()

    relevant_variant_filter = sample_input['Variant_classification'].isin(user_defined_variants)
    sample_input_filtered = sample_input.copy()
    sample_input_filtered = sample_input_filtered[relevant_variant_filter]

    sample_input_filtered.loc[sample_input_filtered['Variant_classification'] == 'stopgain SNV','Action'] = 100
    sample_input_filtered.loc[sample_input_filtered['Variant_classification'] == 'stop loss','Action'] = 100
    sample_input_filtered.loc[sample_input_filtered['Variant_classification'] == 'start loss','Action'] = 'synon'
    sample_input_filtered.loc[sample_input_filtered['Variant_classification'] == 'fs-indel', 'Action'] = 100
    sample_input_filtered.loc[sample_input_filtered['Variant_classification'] == 'splice site', 'Action'] = 100
    sample_input_filtered.loc[sample_input_filtered['Variant_classification'] == 'synonymous SNV','Action'] = 'synon'
    sample_input_filtered.loc[sample_input_filtered['Variant_classification'] == '3_prime_UTR_variant', 'Action'] = 'synon'
    sample_input_filtered.loc[sample_input_filtered['Variant_classification'] == '5_prime_UTR_variant', 'Action'] = 'synon'
    sample_input_filtered.loc[sample_input_filtered['Variant_classification'] == 'indel', 'Action'] = 'synon'
    sample_input_filtered['Action'].fillna('no EA score', inplace = True)
    nonsyn_error_genes = list(sample_input_filtered[sample_input_filtered['Action'] == 'no EA score']['gene_ID'])

    EA_100 = [x for x in list(sample_input_filtered['Action']) if x == 100]
    syn_0 = [x for x in list(sample_input_filtered['Action']) if x == 'synon']
    nonsyn_missing_EA = [x for x in list(sample_input_filtered['Action']) if x == 'no EA score']
    nonsyn_with_EA = [x for x in list(sample_input_filtered['Action']) if type(x) != str and x < 100]

    # need to drop from sample_input_filtered df genes with 'no_EA'; these are "errored" genes in AK code
    nonsyn_missing_EA_filter = sample_input_filtered['Action'] != 'no EA score'
    sample_input_filtered_final = sample_input_filtered.copy()
    sample_input_filtered_final = sample_input_filtered_final[nonsyn_missing_EA_filter]

    # creating nested list of genes with corresponding EA scores to buid gene x EA matrix (after filtering 'no_EA' genes)
    # collecting list of all EA scores in cancer type (after filtering 'no_EA' genes)
    # building gene x EA matrix

    sample_input_genes = list(set(list(sample_input_filtered_final['gene_ID'])))

    sample_genes_with_EA_lsts = []
    sample_genes_all_EA_scores = []

    for gene in sample_input_genes:
        gene_filter = sample_input_filtered_final['gene_ID'] == gene
        gene_df = sample_input_filtered_final.copy()
        gene_df = gene_df[gene_filter]
        gene_action = list(gene_df['Action'])
        sample_genes_all_EA_scores.extend(gene_action)
        gene_action.insert(0, gene)
        sample_genes_with_EA_lsts.append(gene_action)

    length = max(map(len, sample_genes_with_EA_lsts))
    sample_input_gene_action_matrix = np.array([x+[None]*(length-len(x)) for x in sample_genes_with_EA_lsts])

    # text_file summary of gene x EA matrix generation process
    text_file = open(summary_txt_location, 'w')

    text_file.write("Summary of sample_input genes x EA matrix creation process:" + '\n' + '\n')
    text_file.write('Shape of sample_input matrix: ' + str(sample_input.shape) + '\n')
    text_file.write("Considered (relevant) SNVs: " + '\n')
    for item in user_defined_variants:
        text_file.write(item + '\n')
    text_file.write('\n')
    text_file.write('Number of stopgain SNV, fs-indels, splice site, and stop loss SNV annotated with EA = 100: '+ str(len(EA_100)) + '\n')
    text_file.write('Number of synonymous SNV, indels, 5/3_UTR, and start loss SNV annotated with EA = synon: '+ str(len(syn_0)) + '\n')
    text_file.write('Number of nonsynonymous SNV w/o EA annotated with EA = no EA score: '+ str(len(nonsyn_missing_EA))+ '\n')
    text_file.write('Number of nonsynonymous SNV with EA scores: '+ str(len(nonsyn_with_EA)) + '\n')
    text_file.write('Shape of input sample matrix (relevant SNVs only): '+ str(sample_input_filtered.shape)+ '\n')
    text_file.write('Shape of input sample matrix (relevant SNVs only + drop no_EA SNVs): '+ str(sample_input_filtered_final.shape)+ '\n')
    text_file.write('\n')
    text_file.write('Total number of unique genes with SNVs in input samples after filtering: ' + str(len(sample_input_genes))+ '\n')
    text_file.write('Shape of gene x EA matrix: ' + str(sample_input_gene_action_matrix.shape) + '\n')
    text_file.write('\n')
    text_file.write("Time to generate gene x EA matrix: " + str(time.time() - start) + '\n')

    text_file.close()
    
    return sample_input_gene_action_matrix, sample_genes_all_EA_scores, sample_input_genes, list(set(nonsyn_error_genes))

