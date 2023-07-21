#!/usr/bin/env python
# coding: utf-8

# In[11]:

# python packages and libraries
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
from EA_Pathways_Functions import *
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Parse inputs to EA-Pathways")
    parser.add_argument('--AnalysisName', nargs='?', required=True, help = 'Name of the analysis, prefix added to all output files in analysis')
    parser.add_argument('--Variants', nargs='?', required=True, help = 'location of file containing variants of input cohort')
    parser.add_argument('--BiologicalGroups', nargs='?', required=True, help = 'location of file containing biological groups of interest')
    parser.add_argument('--Output', nargs='?', default=None, help = 'location for output files')
    parser.add_argument('--Cores', nargs='?', default=1, type=int, help = 'number of cores for parallelization of analysis')
    arguments = parser.parse_args()
    arguments = vars(arguments)
    return arguments

if __name__ == '__main__':
    args = parse_args()
# collect system arguments
    sample_name = args['AnalysisName']
    sample_file = args['Variants']
    groups_file = args['BiologicalGroups']
    output_directory = os.getcwd() if not args['Output'] else args['Output']
    number_cores = args['cores']

    # creating platform independent file paths
    cwd = os.getcwd()
    outputFolder = output_directory

    # output file names
    LogFile_txt_summary = sample_name + '_Time4EachStep.txt'
    step1_txt_summary = sample_name + '_Step1_BasicStats.txt'
    step2_txt_summary = sample_name + '_Step2_GeneEA_MatrixSummary.txt'
    step3_txt_summary = sample_name + '_Step3_KS_TestSingleGeneSummary.txt'
    step3_csv_summary = sample_name + '_Step3_KS_TestSingleGeneSummary.csv'
    step4_txt_summary = sample_name + '_Step4_SamplePrep4LOO_AnalysisSummary.txt'
    step7_csv_RawSim_summary = sample_name + '_Step7_RawSimulation.csv'
    step7_csv_SimCorePvalues_summary = sample_name + '_Step7_SimCorePvalues.csv'
    step7_csv_SimCorePvaluePerc_summary = sample_name + '_Step7_SimCorePvaluePercentiles.csv'
    step8_csv_final_summary = sample_name + '_Step8_FinalAnalysisSummary.csv'
    step8_txt_SigCoreGenes = sample_name + '_Step8_SigCoreGenesAbove5thPercentile.txt'

    # output file locations
    LogFile_summary_txt_location = createPath(cwd, outputFolder, LogFile_txt_summary)
    step1_summary_txt_location = createPath(cwd, outputFolder, step1_txt_summary)
    step2_summary_txt_location = createPath(cwd, outputFolder, step2_txt_summary)
    step3_summary_txt_location = createPath(cwd, outputFolder, step3_txt_summary)
    step3_summary_csv_location = createPath(cwd, outputFolder, step3_csv_summary)
    step4_summary_txt_location = createPath(cwd, outputFolder, step4_txt_summary)
    step7_RawSim_summary_csv_location = createPath(cwd, outputFolder, step7_csv_RawSim_summary)
    step7_SimCorePvalues_csv_location = createPath(cwd, outputFolder, step7_csv_SimCorePvalues_summary)
    step7_SimCorePvaluePerc_csv_location = createPath(cwd, outputFolder, step7_csv_SimCorePvaluePerc_summary)
    step8_FinalSummary_location = createPath(cwd, outputFolder, step8_csv_final_summary)
    step8_SigCoreGenes_location = createPath(cwd, outputFolder, step8_txt_SigCoreGenes)

    # matrix of gene vs EA scores for debugging VEP annotations; can remove later
    step5_csv_test = sample_name + '_Step5_SummaryMatrixTest.csv'
    step5_TestMatrix_location = createPath(cwd, outputFolder, step5_csv_test)

    # initiate start time for full analysis
    start_all = time.time()

    # load sample_file and groups_file into pd dataframes
    sample_input_df = pd.read_csv(sample_file, header=0)
    groups_input_df = pd.read_csv(groups_file, header=None)

    # variants considered in analysis
    #'3_prime_UTR_variant','5_prime_UTR_variant' may be considered in future analyses
    relevant_variants = ['nonsynonymous SNV', 'stopgain SNV', 'synonymous SNV','stop loss','start loss','indel','fs-indel',
                         'splice site']

    # open log file to collect time for each step
    LogFile = open(LogFile_summary_txt_location, 'w')

    # Step1: Generate basic stats on samples and biological groups
    step1_start = time.time()
    print('Now Performing Step1: Generating basic statistics on input samples and biological groups')
    all_unique_group_genes, all_group_genes_with_duplicates = inputBasicStats(sample_input_df, groups_input_df,
                                                                              step1_summary_txt_location, relevant_variants)
    LogFile.write('Time to perform Step1: ' + str(time.time() - step1_start) + '\n')

    # Step2: Generate sample genes x EA matrix
    step2_start = time.time()
    print('Now Performing Step2: Creating genes from input samples X EA scores dictionary')
    sample_gene_EA_dict, all_sample_genes_EAscores, all_unique_sample_genes, errored_genes = createGeneEAdictionary(relevant_variants,
                                                                                                                 sample_input_df,
                                                                                                                 step2_summary_txt_location)
    LogFile.write('Time to perform Step2: ' + str(time.time() - step2_start) + '\n')

    # Step3: Perform KS test on each sample gene to identify individually significant sample genes
    step3_start = time.time()
    print(
        'Now Performing Step3: Performing KS test on each gene in input samples to identify genes with significantly biased EA distributions')
    all_sample_genes_floatEAscores, sig_single_sample_genes_lst = KStestIndividualGenes(all_sample_genes_EAscores,
                                                                                        sample_gene_EA_dict,
                                                                                        step3_summary_txt_location,
                                                                                        step3_summary_csv_location)
    LogFile.write('Time to perform Step3: ' + str(time.time() - step3_start) + '\n')

    # Step4: Prep samples genes in biological groups with EA scores for LOO analysis and generate summary matrix to collect results
    step4_start = time.time()
    print('Now Performing Step4: Prepping input samples and biological groups for LOO analysis')
    final_summary_matrix, prepped_groups_noSigGenes_EAscores_lst = PrepSamples4LOO_Analysis(all_unique_sample_genes,
                                                                                            all_unique_group_genes,
                                                                                            errored_genes,
                                                                                            groups_input_df,
                                                                                            sample_gene_EA_dict,
                                                                                            sig_single_sample_genes_lst,
                                                                                            step4_summary_txt_location)
    #final_summary_matrix.to_csv(step4_TestMatrix_location, index = False)
    LogFile.write('Time to perform Step4: ' + str(time.time() - step4_start) + '\n')

    # Step5: Perform LOO analysis on input samples and biological groups
    print('Now Performing Step5: Performing LOO analysis on input samples and biological groups')
    step5_start = time.time()

    loo_input_samples_output = pool_loo_analysis_sims_fx(prepped_groups_noSigGenes_EAscores_lst,
                                                      all_sample_genes_floatEAscores,
                                                      number_cores)

    loo_input_samples_df = pd.DataFrame(loo_input_samples_output,
                                 columns=['group_name', 'original_group_pvalue', 'core_genes', 'core_group_pvalue'])

    final_summary_matrix = final_summary_matrix.merge(loo_input_samples_df, on = "group_name", how = "outer")
    #final_summary_matrix.to_csv(step5_TestMatrix_location, index = False)

    LogFile.write('Time to perform Step5: ' + str(time.time() - step5_start) + '\n')

    # Step6: Generate simulated biological groups with multiprocessing
    step6_start = time.time()
    print('Now Performing Step6: Generating simulated biological groups')
    lst_small_sims = np.arange(5, 15, 1).tolist()
    lst_large_sims = np.arange(15, 51, 5).tolist()
    small_and_large_sims = lst_small_sims + lst_large_sims

    # build simulations with parallelization
    all_small_sims = pool_fx_sims(1000, lst_small_sims, sample_gene_EA_dict, all_group_genes_with_duplicates,
                                  sig_single_sample_genes_lst, errored_genes, number_cores)

    start6_large_start = time.time()
    all_large_sims = pool_fx_sims(1000, lst_large_sims, sample_gene_EA_dict, all_group_genes_with_duplicates,
                                      sig_single_sample_genes_lst, errored_genes, number_cores)

    all_small_sims_flat = [item for sublist in all_small_sims for item in sublist]
    all_large_sims_flat = [item for sublist in all_large_sims for item in sublist]
    all_sims_combined_flat = all_small_sims_flat + all_large_sims_flat

    LogFile.write('Time to perform Step6: ' + str(time.time() - step6_start) + '\n')

    # Step7: Perform LOO analysis on simulated groups
    step7_start = time.time()
    print('Now Performing Step7: LOO analysis on simulated biological groups')
    # LOO analysis with multiprocessing
    loo_simulation_output = pool_loo_analysis_sims_fx(all_sims_combined_flat, all_sample_genes_floatEAscores, number_cores)
    # save simulation LOO and KS analysis output to dataframe for QC purposes
    simulation_df = pd.DataFrame(loo_simulation_output,
                                 columns=['sim_number', 'sim_initial_pvalue', 'sim_core_genes', 'sim_core_pvalue'])
    simulation_df.to_csv(step7_RawSim_summary_csv_location, index=False)
    # collecting core pvalues and core pvalue percentiles
    sim_loo_core_pvalues, sim_loo_core_pvalue_perc = collect_sim_core_pvalues_and_percentiles(small_and_large_sims,
                                                                                              loo_simulation_output)
    # save simulation core pvalues to df and export to csv
    sim_core_pvalue_df = pd.DataFrame(sim_loo_core_pvalues)
    sim_core_pvalue_df.to_csv(step7_SimCorePvalues_csv_location, index=False)
    # save simulation core pvalue percentiles to df and export to csv
    sim_percentile_df = pd.DataFrame(sim_loo_core_pvalue_perc)
    sim_percentile_df.to_csv(step7_SimCorePvaluePerc_csv_location, index=False)
    LogFile.write('Time to perform Step7: ' + str(time.time() - step7_start) + '\n')

    # Step8: Compare sample input/group data to simulated biological groups
    step8_start = time.time()
    print('Now Performing Step8: Comparison of sample input data to simulated biological groups')
    final_summary_matrix_updated = compare_to_simulations(small_and_large_sims, sim_percentile_df, final_summary_matrix)
    final_summary_matrix_updated.to_csv(step8_FinalSummary_location, index=False)
    summary_sig_core_genes = collect_core_genes_in_sig_groups(final_summary_matrix_updated)
    text_file = open(step8_SigCoreGenes_location, 'w')
    text_file.write('Core Genes in Pathways Above 5th Percentile: ' + '\n')
    text_file.write('Total genes: ' + str(len(summary_sig_core_genes)) + '\n')
    for gene in summary_sig_core_genes:
        text_file.write(gene + '\n')
    text_file.close()
    LogFile.write('Time to perform Step8: ' + str(time.time() - step8_start) + '\n')

    # calculate time to complete full analysis
    print("Time to complete full analysis: ", time.time() - start_all)
    LogFile.write('Time to perform full analysis - ' + str(time.time() - start_all)  + '\n')
    LogFile.close()

    # In[ ]:


    # In[ ]:
