#!/usr/bin/env python
# coding: utf-8

# In[ ]:

#python packages
import pandas as pd
import numpy as np
import time
from statistics import stdev

def sample_and_groups_basic_stats(sample_input, groups_input, output_text_location, variant_types):

    start = time.time()
    text_file = open(output_text_location, 'w')

    # cancer stats
    relevant_variant_lst = variant_types

    # variant stats
    total_mutations = sample_input.shape[0]
    sample_variant_summary = sample_input.groupby('Variant_classification').count()

    total_relevant_mts = 0
    sample_summary_dict = {}
    for item in relevant_variant_lst:
        item_filter = sample_input['Variant_classification'] == item
        num_item = len(list(sample_input.loc[item_filter]['gene_ID']))
        sample_summary_dict[item] = num_item
        total_relevant_mts = total_relevant_mts + num_item

    # stats on biological groupings
    all_groups_genes = []
    for row in range(groups_input.shape[0]):
        group_genes = list(groups_input.loc[row, 2:groups_input.shape[1]].dropna())
        all_groups_genes.extend(group_genes)

    all_groups_genes_unique = set(all_groups_genes)

    # Output summary of stats to text file
    # Write cancer stats to text file
    text_file.write("Summary of Mutations in Input Samples:" + '\n' + '\n')
    #text_file.write("Total number of unique input samples: " + str(unique_patients) + '\n')
    text_file.write("Total number of mutations: " + str(total_mutations) + '\n')
    #text_file.write("Avg. number mutations per sample: " + str(avg_mt_per_patient) + '\n')
    #text_file.write("Stdev. of mutations per sample: " + str(std_mt_per_patient) + '\n')
    #text_file.write("Range of mutations per sample: " + str(min(lst_mt_per_patient))
                    #+ ' - ' + str(max(lst_mt_per_patient)) + '\n')
    text_file.write("Considered (relevant) SNVs: " + '\n')
    for item in relevant_variant_lst:
        text_file.write(item + '\n')
    text_file.write("Total number of relevant SNVs in input samples: "
                    + str(total_relevant_mts) + '\n')
    text_file.write('\n')
    text_file.write("Summary of relevant SNV data for input samples: " + '\n')
    for key, values in sample_summary_dict.items():
        text_file.write(key + ': ' + str(sample_summary_dict[key]) + '\n')

    #write stats on biological groupings
    text_file.write('\n' + '\n')
    text_file.write("Summary of Input Samples Mutations in Biological Groupings:" + '\n' + '\n')
    text_file.write('Total number of biological groups: ' + str(groups_input.shape[0]) + '\n')
    text_file.write("Number of unique genes in biological groups: " + str(len(all_groups_genes_unique)) + '\n')

    cancer_variant_gene_lst = []
    for variant in relevant_variant_lst:
        variant_filter = sample_input['Variant_classification'] == variant
        genes = set(list(sample_input[variant_filter]['gene_ID']))
        text_file.write('Total number of input samples ' + variant + 's in biological groups: '+
              str(len(genes.intersection(all_groups_genes_unique))) + '\n')
        cancer_variant_gene_lst.extend(genes.intersection(all_groups_genes_unique))

    #print("Time to generate basic stats: ", str(time.time() - start))
    text_file.write('Total input samples (relevant) SNVs in biological groups: ' + str(len(cancer_variant_gene_lst)) + '\n')
    text_file.write('Percentage input samples (relevant) SNVs in biological groups: ' + str(len(cancer_variant_gene_lst)/total_relevant_mts) + '\n')
    text_file.write('\n')
    text_file.write("Time to generate basic stats: " + str(time.time() - start))
    text_file.close()

    return all_groups_genes_unique, all_groups_genes
    

