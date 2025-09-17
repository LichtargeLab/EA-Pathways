#!/usr/bin/env python
"""
@author: Jenn Asmussen

EA-Pathways Pipeline

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
from EA_Pathways_Functions import *
from EA_Pathways_VCF_VEPparser import *
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Parse inputs to EA-Pathways")
    parser.add_argument('--AnalysisName', nargs='?', required=True,
                        help = 'Name of the analysis, prefix added to all output files in analysis')
    parser.add_argument('--Variants', nargs='?', required=False,
                        help = 'location of file containing variants of input cohort')
    parser.add_argument('--BiologicalGroups', nargs='?', required=True,
                        help = 'location of file containing biological groups of interest')
    parser.add_argument('--Output', nargs='?', default=True, help = 'location for output files')
    parser.add_argument('--Cores', nargs='?', default=1, type=int,
                        help = 'number of cores for parallelization of analysis')
    parser.add_argument('--VCF', nargs='?', required=False, default=None,
                        help = 'Location of VCF with cohort variants')
    parser.add_argument('--Samples',nargs='?', required=False, default=None,
                        help = 'Single column txt file with cohort sample IDs')
    parser.add_argument('--RefPopAC', nargs='?', required=False, default = None,
                        help = 'Variant AC values from reference population')
    parser.add_argument('--MinAC', nargs='?',default=1, required=False, type=int,
                        help = 'Min AC count for variant inclusion')
    parser.add_argument('--MaxAC', nargs='?',default=5, type=int, required=False,
                        help='Max AC count for variant inclusion')
    parser.add_argument('--CollectGT', nargs='?', default=None, required = False, type=int,
                        help='Set to 1 to collect singelton genotypes from cohort')
    parser.add_argument('--Verbose', nargs='?', default='N', required = False,
                        help='Flag for capturing intermediate summary files')
    arguments = parser.parse_args()
    arguments = vars(arguments)
    return arguments

def verbose_output(sample_name, cwd_path, out_dir):

    step1_txt_summary = sample_name + '_BasicStats.txt'
    step2_txt_summary = sample_name + '_GeneEA_MatrixSummary.txt'
    step3_txt_summary = sample_name + '_SingleGene_KStest_TxtSummary.txt'
    step3_csv_summary = sample_name + '_SingleGene_KStest_Summary.csv'
    step4_txt_summary = sample_name + '_SamplePrep4LOO_AnalysisSummary.txt'
    final_combined_csv = sample_name + '_CombinedPathwayAnalysisSummary.csv'
    final_combined_txt = sample_name + '_CombinedSignificantCoreGenes.txt'

    step1_summary_txt_location = createPath(cwd_path, out_dir, step1_txt_summary)
    step2_summary_txt_location = createPath(cwd_path, out_dir, step2_txt_summary)
    step3_summary_txt_location = createPath(cwd_path, out_dir, step3_txt_summary)
    step3_summary_csv_location = createPath(cwd_path, out_dir, step3_csv_summary)
    step4_summary_txt_location = createPath(cwd_path, out_dir, step4_txt_summary)
    final_combined_csv_location = createPath(cwd_path, out_dir, final_combined_csv)
    final_combined_txt_location = createPath(cwd_path, out_dir, final_combined_txt)


    return dict(zip(['step1','step2','step3_txt','step3_csv','step4','final_csv','final_txt'],
                    [step1_summary_txt_location, step2_summary_txt_location, step3_summary_txt_location,
                     step3_summary_csv_location, step4_summary_txt_location, final_combined_csv_location,
                     final_combined_txt_location]))

if __name__ == '__main__':
    args = parse_args()

    sample_name = args['AnalysisName']
    sample_file = args['Variants']
    groups_file = args['BiologicalGroups']
    output_directory = os.getcwd() if not args['Output'] else args['Output']
    number_cores = args['Cores']
    vcfFile = args['VCF']
    patientFile = args['Samples']
    refPopVariantFile = args['RefPopAC']
    minRefAC = args['MinAC']
    maxRefAC = args['MaxAC']
    gt_status = args['CollectGT']
    verbose_status = args['Verbose']

    cwd = os.getcwd()
    output_dictionary = verbose_output(sample_name, cwd, output_directory)

    start_all = time.time()
    LogFile_summary_txt_location = createPath(cwd, output_directory, sample_name + '_EA-Pathways_Analysis_Time.txt')
    LogFile = open(LogFile_summary_txt_location, 'w')

    # Part 1: Parse, filter, and format variants from VCF
    if vcfFile != None and sample_file == None:
        part1_start = time.time()
        cohort_variants_df = collectVCFvariants(vcfFile)
        ## In the following function --> if refPopVariantFile == None, run with cohort singletons
        cohort_variants_df, refAC_dict = filterVCFvariants(cohort_variants_df, refPopVariantFile, maxRefAC, minRefAC)

        if gt_status == 1:
            cohort_variants_df = getFilteredVCFvariantsGT(cohort_variants_df, patientFile, number_cores, vcfFile,
                                                          refAC_dict, maxRefAC, minRefAC)

            cohort_variants_df.to_csv(output_directory + sample_name + '_cohort_variants_GT.csv', index=False)
        else:
            cohort_variants_df = cohort_variants_df[['gene_ID', 'Variant_classification','AAchange','Action','refAC']]
            cohort_variants_df.to_csv(output_directory + sample_name + '_cohort_variants.csv', index=False)

        LogFile.write('Time to parse, filter, format cohort variants: ' + str(time.time() - part1_start) + '\n')

    elif sample_file != None and vcfFile == None:
        cohort_variants_df = pd.read_csv(sample_file)
    else:
        print('Warning: Conflicting variant input arguments. Do you want to run from VCF or pre-generated variant file?')
        sys.exit(1)

    # Part 2: Perform EA-Pathways
    sample_input_df = cohort_variants_df
    groups_input_df = pd.read_csv(groups_file, header=None)
    relevant_variants = ['nonsynonymous SNV', 'stopgain SNV', 'synonymous SNV','stop loss','start loss','indel','fs-indel',
                         'splice site']

    # A. Generate basic stats on samples and biological groups
    step1_start = time.time()
    print('A. Generating basic statistics on input samples and biological groups')
    all_unique_group_genes, all_group_genes_with_duplicates = inputBasicStats(sample_input_df, groups_input_df,
                                                                              output_dictionary, relevant_variants,
                                                                              verbose_status)
    LogFile.write('Time to generate cohort variant statistics: ' + str(time.time() - step1_start) + '\n')

    # B. Generate sample genes x EA matrix
    step2_start = time.time()
    print('B. Collecting EA scores per gene')
    sample_gene_EA_dict, all_sample_genes_EAscores, all_unique_sample_genes, errored_genes = createGeneEAdictionary(relevant_variants,
                                                                                                                 sample_input_df,
                                                                                                                 output_dictionary,
                                                                                                                    verbose_status)
    LogFile.write('Time to collect EA scores per gene: ' + str(time.time() - step2_start) + '\n')

    # C. Perform gene-level KS test to identify genes with EA distribution biases
    step3_start = time.time()
    print('C. Performing gene-level KS tests to identify genes with biased EA distributions')
    all_sample_genes_floatEAscores, sig_single_sample_genes_lst = KStestIndividualGenes(all_sample_genes_EAscores,
                                                                                        sample_gene_EA_dict,
                                                                                        output_dictionary,
                                                                                        verbose_status)
    LogFile.write('Time to perform gene-level KS tests: ' + str(time.time() - step3_start) + '\n')

    # D. Prep biological groups with EA scores for LOO analysis
    step4_start = time.time()
    print('D. Prepping pathways for LOO optimization')
    final_summary_matrix, prepped_groups_noSigGenes_EAscores_lst = PrepSamples4LOO_Analysis(all_unique_sample_genes,
                                                                                            all_unique_group_genes,
                                                                                            errored_genes,
                                                                                            groups_input_df,
                                                                                            sample_gene_EA_dict,
                                                                                            sig_single_sample_genes_lst,
                                                                                            output_dictionary,
                                                                                            verbose_status)
    LogFile.write('Time to prep pathways for LOO optimization: ' + str(time.time() - step4_start) + '\n')

    # E. Perform LOO optimization on pathways
    print('E. Performing LOO optimization on pathways')
    step5_start = time.time()

    loo_input_samples_output = pool_loo_analysis_sims_fx(prepped_groups_noSigGenes_EAscores_lst,
                                                      all_sample_genes_floatEAscores,
                                                      number_cores)

    loo_input_samples_df = pd.DataFrame(loo_input_samples_output,
                                 columns=['group_name', 'original_group_pvalue', 'core_genes', 'core_group_pvalue'])

    final_summary_matrix = final_summary_matrix.merge(loo_input_samples_df, on = "group_name", how = "outer")
    LogFile.write('Time to perform LOO optimization on pathways: ' + str(time.time() - step5_start) + '\n')

    # The following subsections are repeated 10x to identify pathways that are significant across 10x independent
    # pathway simulations
    optimized_pathway_output_dict = {}
    for i in np.arange(1, 11, 1):

        output_directory_i = createPath(cwd, output_directory, sample_name + '_simulations_'+str(i))
        if not os.path.exists(output_directory_i):
            os.makedirs(output_directory_i)
        else:
            pass

        print('Generating iteration ', str(i), ' of pathway simulations --')

        # F. Generate simulated pathways
        step6_start = time.time()
        print('F. Generating simulated pathways')
        lst_small_sims = np.arange(5, 15, 1).tolist()
        lst_large_sims = np.arange(15, 51, 5).tolist()
        small_and_large_sims = lst_small_sims + lst_large_sims

        all_small_sims = pool_fx_sims(1000, lst_small_sims, sample_gene_EA_dict, all_group_genes_with_duplicates,
                                      sig_single_sample_genes_lst, number_cores)
        all_large_sims = pool_fx_sims(1000, lst_large_sims, sample_gene_EA_dict, all_group_genes_with_duplicates,
                                          sig_single_sample_genes_lst, number_cores)

        all_small_sims_flat = [item for sublist in all_small_sims for item in sublist]
        all_large_sims_flat = [item for sublist in all_large_sims for item in sublist]
        all_sims_combined_flat = all_small_sims_flat + all_large_sims_flat
        LogFile.write('Time to generate simulated pathways for iteration '+str(i)+': ' + str(time.time() - step6_start) + '\n')

        # G. Perform LOO optimization on simulated groups
        step7_start = time.time()
        print('G. Performing LOO optimization on simulated pathways')
        loo_simulation_output = pool_loo_analysis_sims_fx(all_sims_combined_flat, all_sample_genes_floatEAscores, number_cores)
        simulation_df = pd.DataFrame(loo_simulation_output,
                                     columns=['sim_number', 'sim_initial_pvalue', 'sim_core_genes', 'sim_core_pvalue'])

        sim_loo_core_pvalues, sim_loo_core_pvalue_perc = collect_sim_core_pvalues_and_percentiles(small_and_large_sims,
                                                                                                  loo_simulation_output)
        sim_core_pvalue_df = pd.DataFrame(sim_loo_core_pvalues)
        sim_percentile_df = pd.DataFrame(sim_loo_core_pvalue_perc)
        if verbose_status != 'N':
            simulation_df.to_csv(output_directory_i + '/' + sample_name + '_RawSimulations.csv', index=False)
            sim_core_pvalue_df.to_csv(output_directory_i + '/' + sample_name + '_Simulations_CorePvalues.csv', index=False)
            sim_percentile_df.to_csv(output_directory_i + '/' + sample_name + '_Simulations_CorePvaluePercentiles.csv', index=False)
        LogFile.write('Time to perform LOO optimization on simulated pathways for iteration '+str(i)+': ' + str(time.time() - step7_start) + '\n')

        # H. Compare optimized pathways to optimized simulated pathways
        step8_start = time.time()
        print('H. Comparing optimized pathways to optimized simulated pathways')
        final_summary_matrix_updated = compare_to_simulations(small_and_large_sims, sim_percentile_df, final_summary_matrix)
        final_summary_matrix_updated.to_csv(output_directory_i + '/' + sample_name + '_PathwayAnalysisSummary.csv', index=False)
        optimized_pathway_output_dict[i] = final_summary_matrix_updated

        summary_sig_core_genes = collect_core_genes_in_sig_groups(final_summary_matrix_updated)
        text_file = open(output_directory_i + '/' + sample_name + '_SignificantPathwayCoreGenes.txt', 'w')
        text_file.write('Core Genes in Pathways Above 5th Percentile: ' + '\n')
        text_file.write('Total genes: ' + str(len(summary_sig_core_genes)) + '\n')
        for gene in summary_sig_core_genes:
            text_file.write(gene + '\n')
        text_file.close()
        LogFile.write('Time to compare optimized pathways to simulated pathways for iteration '+str(i)+': ' + str(time.time() - step8_start) + '\n')

    # I. Combine 10x pathway simulation output files to collect final set of significant pathways
    sigPaths_Output_df, sigCoreGenes_Output = combineSummaryMatrices(optimized_pathway_output_dict)
    sigPaths_Output_df.to_csv(output_dictionary['final_csv'])
    text_file = open(output_dictionary['final_txt'], 'w')
    for gene in summary_sig_core_genes:
        text_file.write(gene + '\n')
    text_file.close()

    print("Time to complete full analysis: ", time.time() - start_all)
    LogFile.write('Time to perform full analysis - ' + str(time.time() - start_all)  + '\n')
    LogFile.close()


