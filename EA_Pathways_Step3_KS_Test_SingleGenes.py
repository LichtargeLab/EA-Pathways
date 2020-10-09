#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import statistics
from statsmodels.stats.multitest import fdrcorrection
from scipy import stats as statistics
import time

def KS_test_individual_sample_genes(sample_genes_all_EA_scores, sample_input_gene_action_matrix, txt_summary_location, csv_summary_location):
    
    # v3: KS test on each mutated gene in cancer type; synon SNVs included in gene x EA matrix
    start = time.time()
    sample_genes_all_float_EA_scores = [x for x in sample_genes_all_EA_scores if x != 'synon']
    sample_genes_all_float_EA_scores = [float(x) for x in sample_genes_all_float_EA_scores]

    sig_single_gene_df = pd.DataFrame()
    sig_single_gene_df['gene'] = sample_input_gene_action_matrix[:,0]

    sig_single_gene_pvalues = []

    for row in range(sample_input_gene_action_matrix.shape[0]):
        gene_ea_scores = sample_input_gene_action_matrix[row, 1:sample_input_gene_action_matrix.shape[1]]
        gene_ea_scores = [x for x in gene_ea_scores if x != None]
        gene_ea_scores = [x for x in gene_ea_scores if x != 'synon']
        gene_ea_scores = [float(x) for x in gene_ea_scores]
        if len(gene_ea_scores) == 0:
            sig_single_gene_pvalues.append('No EA scores')
        else: 
            sig_single_gene_pvalues.append(statistics.mstats.ks_twosamp(gene_ea_scores, sample_genes_all_float_EA_scores, alternative='less')[1])

    sig_single_gene_df['p_value'] = sig_single_gene_pvalues

    sig_single_gene_pvalues_dropNoEA = [x for x in sig_single_gene_pvalues if x != 'No EA scores']

    q_value_df = sig_single_gene_df.copy()
    index_noIntEA = q_value_df[q_value_df['p_value']=='No EA scores'].index
    q_value_df.drop(index_noIntEA, inplace = True)
    q_value_df_pvalue_lst = list(q_value_df['p_value'])
    sig_single_gene_qvalues = fdrcorrection(q_value_df_pvalue_lst, alpha=0.05)[1]
    q_value_df['q_value'] = sig_single_gene_qvalues

    ks_merged_df = sig_single_gene_df.merge(q_value_df, how = 'outer', left_on='gene', right_on='gene')
    ks_merged_df.sort_values(by = 'p_value_y', inplace = True)
    ks_merged_df.drop(columns = 'p_value_y', inplace = True)
    ks_merged_df.rename(columns = {'p_value_x':'p_value'}, inplace = True)

    # generate sig_single_genes_lst from KS analysis performed on each gene
    filter_sig_genes = ks_merged_df['q_value'] < 0.1
    sig_single_genes_lst = list(ks_merged_df[filter_sig_genes]['gene'])

    ks_merged_df.to_csv(csv_summary_location, index = False)

    # text file summarizing KS test of each gene mutated in cancer type
    text_file = open(txt_summary_location, 'w')
    text_file.write("Summary of single gene KS tests for input samples:" + '\n' + '\n')
    text_file.write('Total number of EA annotations in input samples: ' +  str(len(sample_genes_all_EA_scores)) + '\n')
    text_file.write('Total number of integer EA scores in input samples: ' + str(len(sample_genes_all_float_EA_scores))+ '\n')
    text_file.write('Total number of unique mutated genes assessed by KS test: '+ str(len(sig_single_gene_pvalues_dropNoEA))+ '\n')
    text_file.write('Shape of gene x EA matrix: ' + str(sample_input_gene_action_matrix.shape) + '\n')
    text_file.write('Shape of sig_single_gene_matrix: '+ str(ks_merged_df.shape) + '\n')
    text_file.write('\n')
    text_file.write("Time to perform KS test for each unique mutated gene in input samples: " + str(time.time() - start) + '\n')
    text_file.close()
    
    return sample_genes_all_float_EA_scores, sig_single_genes_lst

