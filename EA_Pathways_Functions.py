#!/usr/bin/env python

"""
@author: Jenn Asmussen

Functions for running EA-Pathways

"""

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import fdrcorrection
from scipy import stats as statistics
from scipy.optimize import curve_fit
import time
import random
import os
import multiprocessing as mp
import sys
import ast

def createPath(currentDirectory, folderName, fileName):
    return os.path.join(currentDirectory, folderName, fileName)

def inputBasicStats(sample_input, groups_input, output_location, variant_types, verbose):

    total_mutations = sample_input.shape[0]
    total_relevant_mts = 0
    sample_summary_dict = {}
    for item in variant_types:
        item_df = sample_input[sample_input['Variant_classification']==item]
        num_item = item_df['Variant_classification'].count()
        sample_summary_dict[item] = num_item
        total_relevant_mts = total_relevant_mts + num_item

    all_groups_genes = []
    for row in range(groups_input.shape[0]):
        group_genes = list(groups_input.loc[row, 2:groups_input.shape[1]].dropna())
        all_groups_genes.extend(group_genes)

    all_groups_genes_unique = set(all_groups_genes)

    if verbose != 'N':
        with open(output_location['step1'], 'w') as text_file:
            text_file.write("Summary of Mutations in Input Samples:" + '\n' + '\n')
            text_file.write("Total number of mutations: " + str(total_mutations) + '\n')
            text_file.write("Considered (relevant) SNVs: " + '\n')
            for item in variant_types:
                text_file.write(item + '\n')
            text_file.write("Total number of relevant SNVs in input samples: "+ str(total_relevant_mts) + '\n')
            text_file.write('\n')
            text_file.write("Summary of relevant SNV data for input samples: " + '\n')
            for key, values in sample_summary_dict.items():
                text_file.write(key + ': ' + str(sample_summary_dict[key]) + '\n')

            text_file.write('\n' + '\n')
            text_file.write("Summary of Input Samples Mutations in Biological Groupings:" + '\n' + '\n')
            text_file.write('Total number of biological groups: ' + str(groups_input.shape[0]) + '\n')
            text_file.write("Number of unique genes in biological groups: " + str(len(all_groups_genes_unique)) + '\n')

            cancer_variant_gene_lst = []
            for variant in variant_types:
                try:
                    variant_df = sample_input[sample_input['Variant_classification']==variant]
                    genes = set(variant_df['gene_ID'])
                    genesInGroups = genes.intersection(all_groups_genes_unique)
                    text_file.write('Total number of input samples ' + variant + 's in biological groups: '+ str(len(genesInGroups)) + '\n')
                    cancer_variant_gene_lst.extend(genesInGroups)
                except:
                    pass

            text_file.write('Total input samples (relevant) SNVs in biological groups: ' + str(len(cancer_variant_gene_lst)) + '\n')
            text_file.write('Percentage input samples (relevant) SNVs in biological groups: ' + str(len(cancer_variant_gene_lst)/total_relevant_mts) + '\n')
            text_file.write('\n')
            text_file.close()

    return all_groups_genes_unique, all_groups_genes

def createGeneEAdictionary(user_defined_variants, sample_input, output_location, verbose):

    sample_input_filtered = sample_input.copy()
    sample_input_filtered = sample_input_filtered[sample_input_filtered['Variant_classification'].isin(user_defined_variants)]
    sample_input_filtered['Action'] = sample_input_filtered['Action'].astype('object')

    sample_input_filtered.loc[sample_input_filtered['Variant_classification'] == 'stopgain SNV', 'Action'] = 100
    sample_input_filtered.loc[sample_input_filtered['Variant_classification'] == 'stop loss', 'Action'] = 100
    sample_input_filtered.loc[sample_input_filtered['Variant_classification'] == 'start loss', 'Action'] = 100
    sample_input_filtered.loc[sample_input_filtered['Variant_classification'] == 'fs-indel', 'Action'] = 100
    sample_input_filtered.loc[sample_input_filtered['Variant_classification'] == 'splice site', 'Action'] = 100
    sample_input_filtered.loc[sample_input_filtered['Variant_classification'] == 'synonymous SNV', 'Action'] = 'synon'
    sample_input_filtered.loc[sample_input_filtered['Variant_classification'] == 'indel', 'Action'] = 'synon'
    sample_input_filtered['Action'] = sample_input_filtered['Action'].fillna('no EA score')

    try:
        nonsyn_error_genes = list(sample_input_filtered[sample_input_filtered['Action'] == 'no EA score']['gene_ID'])
    except:
        nonsyn_error_genes = []

    EA_100 = [x for x in list(sample_input_filtered['Action']) if x == 100]
    syn_0 = [x for x in list(sample_input_filtered['Action']) if x == 'synon']
    nonsyn_missing_EA = [x for x in list(sample_input_filtered['Action']) if x == 'no EA score']
    nonsyn_with_EA = [x for x in list(sample_input_filtered['Action']) if x != 'synon']
    nonsyn_with_EA = [x for x in nonsyn_with_EA if x != 'no EA score']
    nonsyn_with_EA = [float(x) for x in nonsyn_with_EA]
    nonsyn_with_EA = [x for x in nonsyn_with_EA if x < 100]

    sample_input_filtered_final = sample_input_filtered.copy()
    sample_input_filtered_final = sample_input_filtered_final[sample_input_filtered_final['Action'] != 'no EA score']

    sample_input_genes = list(set(list(sample_input_filtered_final['gene_ID'])))

    sampleGeneEAdict = {}
    sample_genes_all_EA_scores = []

    for gene in sample_input_genes:
        gene_df = sample_input_filtered_final.copy()
        gene_df = gene_df[gene_df['gene_ID'] == gene]
        gene_action = list(gene_df['Action'])
        sample_genes_all_EA_scores.extend(gene_action)
        sampleGeneEAdict[gene] = gene_action

    if verbose != 'N':
        text_file = open(output_location['step2'], 'w')
        text_file.write("Summary of sample_input genes EA dictionary creation process:" + '\n' + '\n')
        text_file.write('Shape of sample_input matrix: ' + str(sample_input.shape) + '\n')
        text_file.write("Considered (relevant) SNVs: " + '\n')
        for item in user_defined_variants:
            text_file.write(item + '\n')
        text_file.write('\n')
        text_file.write('Number of stopgain SNV, fs-indels, splice site, start loss, and stop loss SNV annotated with EA = 100: ' + str(
            len(EA_100)) + '\n')
        text_file.write('Number of synonymous SNV and indels annotated with EA = synon: ' + str(
            len(syn_0)) + '\n')
        text_file.write(
            'Number of nonsynonymous SNV w/o EA annotated with EA = no EA score: ' + str(len(nonsyn_missing_EA)) + '\n')
        text_file.write('Number of nonsynonymous SNV with EA scores: ' + str(len(nonsyn_with_EA)) + '\n')
        text_file.write('Shape of input sample matrix (relevant SNVs only): ' + str(sample_input_filtered.shape) + '\n')
        text_file.write('Shape of input sample matrix (relevant SNVs only + drop no EA score SNVs): ' + str(
            sample_input_filtered_final.shape) + '\n')
        text_file.write('\n')
        text_file.write('Total number of unique genes with SNVs in input samples after filtering: ' + str(
            len(sample_input_genes)) + '\n')
        text_file.write('Length of gene x EA dictionary: ' + str(len(sampleGeneEAdict)) + '\n')
        text_file.write('\n')
        text_file.close()

    return sampleGeneEAdict, sample_genes_all_EA_scores, sample_input_genes, list(set(nonsyn_error_genes))

def KStestIndividualGenes(sample_genes_all_EA_scores, sample_input_gene_action_dict, output_location, verbose):

    sample_genes_all_float_EA_scores = [x for x in sample_genes_all_EA_scores if x != 'synon']
    sample_genes_all_float_EA_scores = [float(x) for x in sample_genes_all_float_EA_scores]

    sig_single_gene_df = pd.DataFrame()
    sample_genes = list(sample_input_gene_action_dict.keys())
    sig_single_gene_df['gene'] = sample_genes

    sig_single_gene_pvalues = []

    for gene in sample_genes:
        gene_ea_scores = sample_input_gene_action_dict[gene]
        gene_ea_scores = [x for x in gene_ea_scores if x != 'synon']
        gene_ea_scores = [float(x) for x in gene_ea_scores]
        if len(gene_ea_scores) == 0:
            sig_single_gene_pvalues.append('No EA scores')
        else:
            sig_single_gene_pvalues.append(
                statistics.mstats.ks_twosamp(gene_ea_scores, sample_genes_all_float_EA_scores, alternative='less')[1])

    sig_single_gene_df['p_value'] = sig_single_gene_pvalues

    sig_single_gene_pvalues_dropNoEA = [x for x in sig_single_gene_pvalues if x != 'No EA scores']

    q_value_df = sig_single_gene_df.copy()
    q_value_df = q_value_df[q_value_df['p_value'] != 'No EA scores']

    q_value_df_pvalue_lst = list(q_value_df['p_value'])
    sig_single_gene_qvalues = fdrcorrection(q_value_df_pvalue_lst, alpha=0.05)[1]
    q_value_df['q_value'] = sig_single_gene_qvalues

    ks_merged_df = sig_single_gene_df.merge(q_value_df, how='outer', left_on='gene', right_on='gene')
    ks_merged_df.sort_values(by='p_value_y', inplace=True)
    ks_merged_df.drop(columns='p_value_y', inplace=True)
    ks_merged_df.rename(columns={'p_value_x': 'p_value'}, inplace=True)

    filter_sig_genes = ks_merged_df['q_value'] < 0.1
    sig_single_genes_lst = list(ks_merged_df[filter_sig_genes]['gene'])
    ks_merged_df.to_csv(output_location['step3_csv'], index=False)

    if verbose != 'N':
        text_file = open(output_location['step3_txt'], 'w')
        text_file.write("Summary of single gene KS tests for input samples:" + '\n' + '\n')
        text_file.write('Total number of EA annotations in input samples: ' + str(len(sample_genes_all_EA_scores)) + '\n')
        text_file.write(
            'Total number of integer EA scores in input samples: ' + str(len(sample_genes_all_float_EA_scores)) + '\n')
        text_file.write('Total number of unique mutated genes assessed by KS test: ' + str(
            len(sig_single_gene_pvalues_dropNoEA)) + '\n')
        text_file.write('Length of gene x EA dictionary: ' + str(len(sample_input_gene_action_dict)) + '\n')
        text_file.write('Shape of sig_single_gene_matrix: ' + str(ks_merged_df.shape) + '\n')
        text_file.write('\n')
        text_file.close()

    return sample_genes_all_float_EA_scores, sig_single_genes_lst

def PrepSamples4LOO_Analysis(sample_input_genes, all_groups_genes_unique, nonsyn_errored_genes, groups_input,
                             sample_input_gene_action_dict, sig_single_genes_lst, output_location, verbose):

    # Part1: Prep groups with EA scores for LOO analysis
    overlapping_sample_and_group_genes = set(sample_input_genes).intersection(all_groups_genes_unique)
    groups_input_id_lst = list(groups_input[0])

    groups_with_genes_and_EA_scores_lst = []
    groups_with_noSigGenes_and_EA_scores_lst = []
    genes_in_cohort_and_groups_with_mutation = []
    nonsyn_error_genes_in_cohort_and_groups = []

    for row_outer in range(groups_input.shape[0]):
        if len(list(groups_input.iloc[row_outer, 2:groups_input.shape[1]].dropna())) == 1:
            pass #Omit any single gene pathways
        else:
            group_lst_all = []
            group_lst_noSigGenes = []
            group_lst_error_genes = []

            group_name = groups_input.iloc[row_outer,0]
            group_lst_all.append(group_name)
            group_lst_noSigGenes.append(group_name)
            group_genes = list(groups_input.iloc[row_outer, 2:groups_input.shape[1]].dropna())

            for gene in group_genes:
                gene_with_EA_lst = []
                gene_with_EA_lst.append(gene)

                try:
                    gene_ea_scores = sample_input_gene_action_dict[gene]
                    gene_ea_scores = [x for x in gene_ea_scores if x != 'synon']
                    gene_ea_scores = [float(x) for x in gene_ea_scores]
                except:
                    gene_ea_scores = []
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

    # Part 2. Generate summary matrix

    group_names_lst = []
    group_length_lst = []
    group_genes_with_EAscores_lst = []
    group_number_genes_with_EAscores_lst = []
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
    group_error_genes_lst_lengths = [len(x) for x in nonsyn_error_genes_in_cohort_and_groups]
    functional_group_size_lst2 = [a - b for a, b in zip(functional_group_size_lst, group_error_genes_lst_lengths)]
    summary_matrix['functional_group_size'] = functional_group_size_lst2
    summary_matrix['group_errored_genes'] = nonsyn_error_genes_in_cohort_and_groups
    summary_matrix['number_group_genes_with_EAscores'] = group_number_genes_with_EAscores_lst
    summary_matrix['group_genes_with_EAscores'] = group_genes_with_EAscores_lst
    summary_matrix['group_sig_genes'] = group_sig_single_genes_lst
    summary_matrix['total_group_variants'] = group_total_variants_in_cohort_lst

    if verbose != 'N':
        text_file = open(output_location['step4'], 'w')
        text_file.write('Summary of Prepping Input Samples into Biological Groups for LOO Analysis' + '\n' + '\n')
        text_file.write('Number of biological groups: ' + str(len(groups_input_id_lst)) + '\n')
        text_file.write('Number of unique genes in biological groups: ' + str(len(all_groups_genes_unique)) + '\n')
        text_file.write('Number of unique genes with at least one mutation in input samples: ' + str(len(sample_input_genes)) + '\n')
        text_file.write('Overlap of unique sample input genes and unique biological groups genes: ' +
              str(len(overlapping_sample_and_group_genes)) + '\n')
        text_file.write('Number of genes in input samples and groups with >= 1 EA annotation: ' +
              str(len(set(genes_in_cohort_and_groups_with_mutation))) + '\n' + '\n')
        text_file.close()

    return summary_matrix, groups_with_noSigGenes_and_EA_scores_lst

def sims_loo_multiprocessing(arg):
    simulation = arg[0]
    background = arg[1]
    name, ini_pvalue, core_gen, cor_pvalue = group_LOO_core_gene_analysis(simulation, background)
    return name, ini_pvalue, core_gen, cor_pvalue

def pool_loo_analysis_sims_fx(all_simulations, EA_background, cores):
    args = tuple(zip(all_simulations, [EA_background] * len(all_simulations)))
    pool = mp.Pool(processes=cores)
    output = pool.map(sims_loo_multiprocessing, args)
    pool.close()
    pool.join()
    return output

def group_LOO_core_gene_analysis(biological_group_noSigGenes_EAscores_item, background_EAscores):
    background_float_EAscores = [float(x) for x in background_EAscores]

    group_name = biological_group_noSigGenes_EAscores_item[0]
    group_length = len(biological_group_noSigGenes_EAscores_item)

    group_all_EA_scores = []
    for gene in biological_group_noSigGenes_EAscores_item[1:group_length]:
        if len(gene) == 1:
            pass
        else:
            group_all_EA_scores.append(gene[1:len(gene)])

    group_all_EA_scores = [item for sublist in group_all_EA_scores for item in sublist]
    group_all_EA_scores = [x for x in group_all_EA_scores if x != 'synon']
    group_all_EA_scores = [float(x) for x in group_all_EA_scores]

    if len(group_all_EA_scores) == 0:
        initial_group_pvalue = 1
    else:
        initial_group_pvalue = \
        statistics.mstats.ks_twosamp(group_all_EA_scores, background_float_EAscores, alternative='less')[1]

    group_core_genes = []
    group_core_EA_scores = []

    if initial_group_pvalue < 1:
        for test_gene in biological_group_noSigGenes_EAscores_item[1:group_length]:
            if len(test_gene) > 1:
                test_group = biological_group_noSigGenes_EAscores_item[1:group_length].copy()
                test_group.remove(test_gene)

                loo_group_EA_scores = []
                for gene_item in test_group:
                    if len(gene_item) == 1:
                        pass
                    else:
                        loo_group_EA_scores.append(gene_item[1:len(gene_item)])

                loo_group_EA_scores = [item for sublist in loo_group_EA_scores for item in sublist]
                loo_group_EA_scores = [x for x in loo_group_EA_scores if x != 'synon']
                loo_group_EA_scores = [float(x) for x in loo_group_EA_scores]

                if len(loo_group_EA_scores) == 0:
                    group_core_genes.append(test_gene[0])
                    group_core_EA_scores.append(test_gene[1:len(test_gene)])
                else:
                    test_gene_loo_pvalue = \
                    statistics.mstats.ks_twosamp(loo_group_EA_scores, background_float_EAscores, alternative='less')[1]

                    if test_gene_loo_pvalue > initial_group_pvalue:
                        group_core_genes.append(test_gene[0])
                        group_core_EA_scores.append(test_gene[1:len(test_gene)])
                    else:
                        pass

            else:
                pass

    else:
        group_core_genes.append('No nonsyn SNV mutations in biological group')

    group_core_EA_scores = [item for sublist in group_core_EA_scores for item in sublist]
    group_core_EA_scores = [x for x in group_core_EA_scores if x != 'synon']

    if len(group_core_EA_scores) == 0:
        core_group_pvalue = 1
    else:
        core_group_pvalue = \
        statistics.mstats.ks_twosamp(group_core_EA_scores, background_float_EAscores, alternative='less')[1]

    return group_name, initial_group_pvalue, group_core_genes, core_group_pvalue


def generate_simulated_groups(total_simulations, simulation_path_size, gene_EA_dict, ALL_group_input_genes,
                              sig_single_genes_lst):
    lst_all_genes = ALL_group_input_genes
    lst_all_genes = [elem for elem in lst_all_genes if elem not in sig_single_genes_lst]

    lst_sim_pathways = []

    for sim in range(total_simulations):
        sim_genes = random.sample(lst_all_genes, k=simulation_path_size)
        if len(sim_genes) != len(set(sim_genes)):
            while len(sim_genes) != len(set(sim_genes)):
                new_sim_genes = []
                set_sim_genes = set(sim_genes)
                num_replacements = simulation_path_size - len(set_sim_genes)
                replacements = random.sample(lst_all_genes, k=num_replacements)
                new_sim_genes.extend(list(set_sim_genes))
                new_sim_genes.extend(replacements)
                sim_genes = new_sim_genes.copy()
            lst_sim_pathways.append(sim_genes)
        else:
            lst_sim_pathways.append(sim_genes)

    lst_final_sim_paths_NoSigGenes = []
    i = 1
    for sim_path in lst_sim_pathways:
        final_path = [str(simulation_path_size) + '_sim_' + str(i)]
        i += 1
        for gene in sim_path:
            if gene in sig_single_genes_lst:
                pass
            else:
                gene_name_with_EA_scores = []
                gene_name_with_EA_scores.append(gene)
                try:
                    gene_ea_scores = gene_EA_dict[gene]
                    gene_ea_scores = [x for x in gene_ea_scores if x != 'synon']
                    gene_ea_scores = [float(x) for x in gene_ea_scores]
                except:
                    gene_ea_scores = []
                gene_name_with_EA_scores.extend(gene_ea_scores)
                final_path.append(gene_name_with_EA_scores)
        lst_final_sim_paths_NoSigGenes.append(final_path)
    return lst_final_sim_paths_NoSigGenes

def build_sims_multiprocessing(arg):
    num_sims = arg[0]
    size_sims = arg[1]
    sample_gene_EA_dict = arg[2]
    group_input_genes = arg[3]
    sig_genes = arg[4]
    simulations = generate_simulated_groups(num_sims, size_sims, sample_gene_EA_dict, group_input_genes, sig_genes)
    return simulations

def pool_fx_sims(number_simulations, simulation_size_lst, gene_EA_dict, ALL_group_input_genes, sig_single_genes_lst,
                 cores):
    args = tuple(zip(np.full(len(simulation_size_lst), number_simulations).tolist(),
                     simulation_size_lst, [gene_EA_dict] * len(simulation_size_lst),
                     [ALL_group_input_genes] * len(simulation_size_lst),
                     [sig_single_genes_lst] * len(simulation_size_lst)))
    pool = mp.Pool(processes=cores)
    output = pool.map(build_sims_multiprocessing, args)
    pool.close()
    pool.join()
    return output

def collect_sim_core_pvalues_and_percentiles(list_all_group_sizes, LOO_KS_sim_output):
    all_core_pvalues_lst = []
    all_core_percentiles_lst = []
    for sim_size in list_all_group_sizes:
        sim_name = "simulations_size_" + str(sim_size)
        sim_core_pvalues = []
        sim_core_pvalues.append(sim_name)
        for item in LOO_KS_sim_output:
            loo_simulation_output_name = item[0]
            result = loo_simulation_output_name.startswith(str(sim_size) + '_')
            if result == False:
                pass
            else:
                test_core_pvalue = item[3]
                sim_core_pvalues.append(test_core_pvalue)
        all_core_pvalues_lst.append(sim_core_pvalues)

        # prep list for conversion to percentile
        sim_core_pvalues_copy = sim_core_pvalues.copy()
        sim_core_pvalues_copy.remove(sim_name)
        sim_core_percentile = []
        sim_core_percentile.append(sim_name)
        for perc in range(100):
            sim_core_percentile.append(np.percentile(sim_core_pvalues_copy, perc))
        all_core_percentiles_lst.append(sim_core_percentile)

    return all_core_pvalues_lst, all_core_percentiles_lst

def func(x, a, b):
    return a * np.exp(-b * x)

def rsquared(x, y):
    slope, intercept, r_value, p_value, stderr = statistics.linregress(x, y)
    return r_value ** 2

def qvalue_filter(row):
    if row['fdr_q_value_core_pathway'] < 0.05:
        return 1
    else:
        return 0

def threshold(row, a, b, threshold_dictionary):
    if int(row['functional_group_size']) <= 15 and int(
            row['functional_group_size']) >= 5:  # modified for removing smaller pathway simulations
        thresh = threshold_dictionary[int(row['functional_group_size'])]
        return thresh
    else:
        thresh = func(float(row['functional_group_size']), a, b)
        return thresh

def foldbetter(row):
    fold_better = row['5th_percentile_threshold'] / row['core_group_pvalue']
    return fold_better

def collect_core_genes_in_sig_groups(summary_matrix):
    significant_core_genes = []
    for row in range(summary_matrix.shape[0]):
        row_q_value_filter = summary_matrix.at[row, 'passed_q_value_filter']
        row_foldbetter = summary_matrix.at[row, 'fold_better']
        row_total_variants = summary_matrix.at[row, 'total_group_variants']

        if row_q_value_filter == 1 and row_foldbetter > 1 and row_total_variants >= 20:
            row_core_genes = [summary_matrix.at[row, 'core_genes']]
            significant_core_genes.extend(row_core_genes)

    significant_core_genes = [item for sublist in significant_core_genes for item in sublist]
    significant_core_genes_set = set(significant_core_genes)

    return significant_core_genes_set

def compare_to_simulations(small_and_large_sims_lst, sim_percentile_matrix, summary_matrix):

    xvalues = small_and_large_sims_lst
    yvalues = list(sim_percentile_matrix[5])
    xvalues = [float(x) for x in xvalues]
    yvalues = [float(x) for x in yvalues]

    # For simulation sizes <= 15
    threshold_dictionary = dict(zip(xvalues, yvalues))

    x = np.array(xvalues)
    y = np.array(yvalues)
    popt, pcov = curve_fit(func, x, y, [0.2, 0.2])
    print('threshold=' + str(popt[0]) + '*EXP(-' + str(popt[1]) + '*OriginalGroupSize)')
    a = popt[0]
    b = popt[1]
    print('r-squared for this equation is: ' + str(rsquared(x, y)))

    # fdr correction of pathways
    summary_matrix_fdr = summary_matrix.copy()
    summary_matrix_fdr = summary_matrix_fdr[
        ~summary_matrix_fdr['core_genes'].apply(lambda x: 'No nonsyn SNV mutations in biological group' in x)]
    sample_core_group_pvalues = tuple(summary_matrix_fdr['core_group_pvalue'])
    fdr_core_group_pvalues = fdrcorrection(sample_core_group_pvalues, alpha=0.05)[1]
    summary_matrix_fdr['fdr_q_value_core_pathway'] = fdr_core_group_pvalues
    summary_matrix = summary_matrix.merge(summary_matrix_fdr[['group_name', 'fdr_q_value_core_pathway']],
                                          on="group_name", how="outer")

    # Compare pathways to simulated pathways
    summary_matrix['passed_q_value_filter'] = summary_matrix.apply(lambda row: qvalue_filter(row), axis=1)
    summary_matrix['5th_percentile_threshold'] = summary_matrix.apply(
        lambda row: threshold(row, a, b, threshold_dictionary), axis=1)
    summary_matrix['fold_better'] = summary_matrix.apply(lambda row: foldbetter(row), axis=1)
    summary_matrix.sort_values(by=['passed_q_value_filter', 'fold_better'], ascending=False, inplace=True)

    return summary_matrix

def filterPathOutput(df):
    df = df[(df['total_group_variants']>= 20) & (df['passed_q_value_filter'] == 1) & (df['fold_better']>=1)]
    return df

def getSigPathsMatrix(pathwaysSummary_dictionary):
    fullCohortPathways = {}
    fullCohortPathways_CoreGenes = {}
    fullCohortPathways_FullGenes = {}
    fullCohortPathways_startKS = {}
    fullCohortPathways_endKS = {}
    fullCohortPathways_fdrKS = {}
    fullCohortPathways_fb = {}

    for i, out_df in pathwaysSummary_dictionary.items():
        df = out_df
        df = filterPathOutput(df)
        df_paths = df['group_name'].tolist()
        df_paths_dict = dict(zip(df['group_name'], df['core_genes']))
        df_paths_full_dict = dict(zip(df['group_name'], df['group_genes_with_EAscores']))
        df_paths_ks_start_dict = dict(zip(df['group_name'], df['original_group_pvalue']))
        df_paths_ks_end_dict = dict(zip(df['group_name'], df['core_group_pvalue']))
        df_paths_ks_fdr_end_dict = dict(zip(df['group_name'], df['fdr_q_value_core_pathway']))
        df_paths_fb_dict = dict(zip(df['group_name'], df['fold_better']))

        fullCohortPathways[i] = df_paths

        for k, v in df_paths_dict.items():
            fullCohortPathways_CoreGenes[k] = v

        for k, v in df_paths_full_dict.items():
            fullCohortPathways_FullGenes[k] = v

        for k, v in df_paths_ks_start_dict.items():
            fullCohortPathways_startKS[k] = v

        for k, v in df_paths_ks_end_dict.items():
            fullCohortPathways_endKS[k] = v

        for k, v in df_paths_ks_fdr_end_dict.items():
            fullCohortPathways_fdrKS[k] = v

        for k, v in df_paths_fb_dict.items():
            if k in fullCohortPathways_fb:
                fullCohortPathways_fb[k].append(v)
            else:
                fullCohortPathways_fb[k] = [v]

    fullCohortPathways_counts = {}
    for k, v in fullCohortPathways.items():
        for p in v:
            if p in fullCohortPathways_counts:
                fullCohortPathways_counts[p] += 1
            else:
                fullCohortPathways_counts[p] = 1

    fullCohortPaths_df = pd.DataFrame.from_dict(fullCohortPathways_counts, orient='index', columns=['ExpCount'])
    fullCohortPaths_df.sort_values(by='ExpCount', ascending=False, inplace=True)
    fullCohortPaths_df['CoreGenes'] = fullCohortPaths_df.index.map(fullCohortPathways_CoreGenes)
    fullCohortPaths_df['PathGenesWithEAscores'] = fullCohortPaths_df.index.map(fullCohortPathways_FullGenes)
    fullCohortPaths_df['StartKS'] = fullCohortPaths_df.index.map(fullCohortPathways_startKS)
    fullCohortPaths_df['EndKS'] = fullCohortPaths_df.index.map(fullCohortPathways_endKS)
    fullCohortPaths_df['FdrKS'] = fullCohortPaths_df.index.map(fullCohortPathways_fdrKS)
    fullCohortPaths_df['FoldBetter'] = fullCohortPaths_df.index.map(fullCohortPathways_fb)
    fullCohortPaths_df['Avg_FoldBetter'] = fullCohortPaths_df['FoldBetter'].apply(lambda x: np.mean(x))
    fullCohortPaths_df['Std_FoldBetter'] = fullCohortPaths_df['FoldBetter'].apply(lambda x: np.std(x))

    return fullCohortPaths_df

def getCoreGenes(matrix, recurrence):
    df = matrix[matrix['ExpCount']==recurrence]
    coreGenes = df['CoreGenes'].tolist()
    coreGenes = [x for Xs in coreGenes for x in Xs]
    return list(set(coreGenes))

def combineSummaryMatrices(pathwaySummaryDictionary):

    sigPaths_df = getSigPathsMatrix(pathwaySummaryDictionary)
    sigCoreGenes = getCoreGenes(sigPaths_df, 10)

    return sigPaths_df, sigCoreGenes