#!/usr/bin/env python
# coding: utf-8

# In[11]:

# command line format for executing script:
# python jenn_wrapper.py sample_name ./location/samples_with_EA_file_name.csv
# #./location/groups_file_name.csv ./outputdirectory ./histogramdirectory
# maybe specifying variants can be text file argument; currently, only option is "all"

# python packages and libraries
import numpy as np
import pandas as pd
#from statistics import stdev
from statsmodels.stats.multitest import fdrcorrection
from scipy import stats as statistics
from scipy.optimize import curve_fit
import time
import random
import os
import multiprocessing as mp
import sys
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, AutoMinorLocator

# user defined functions
from EA_Pathways_Step1_Stats import sample_and_groups_basic_stats
from EA_Pathways_Step2_Gene_x_EA_matrix import generate_sample_gene_by_EAmatrix
from EA_Pathways_Step3_KS_Test_SingleGenes import KS_test_individual_sample_genes
from EA_Pathways_Step4_PrepSamples4LOO_Analysis import PrepSamples4LOO_Analysis
from EA_Pathways_Step5_Sample_LOO_Analysis import group_LOO_core_gene_analysis, input_samples_loo_multiprocessing, pool_loo_analysis_input_samples_fx
from EA_Pathways_Step6_GenerateGroupSimulations import generate_simulated_groups, \
    build_sims_multiprocessing, pool_fx_sims
from EA_Pathways_Step7_LOO_AnalysisSimGroups import sims_loo_multiprocessing, pool_loo_analysis_sims_fx, \
    collect_sim_core_pvalues_and_percentiles
from EA_Pathways_Step8_CompareSamplesToSims import func, rsquared, qvalue_filter, threshold, foldbetter, \
    collect_core_genes_in_sig_groups, compare_to_simulations
from EA_Pathways_Step9_MakeHistograms import full_biological_group_historgrams, core_biological_group_historgrams

# collect system arguments
sample_name = sys.argv[1]
sample_file = sys.argv[2]
groups_file = sys.argv[3]
output_directory = sys.argv[4]#[2:]
hist_directory = sys.argv[5]#[2:]
number_cores = int(sys.argv[6])

# creating platform independent file paths
cwd = os.getcwd()
inputFolder = "InputData"
outputFolder = output_directory
histFolder = hist_directory

# File path creation function
def createPath(currentDirectory, folderName, fileName):
    return os.path.join(currentDirectory, folderName, fileName)

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
step9_hist_plots = sample_name + '_'
#step4_csv_test = sample_name + '_Step4_SummaryMatrixTest.csv'
#step5_csv_test = sample_name + '_Step5_ParallelTestMatrix.csv'

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
step9_HistCoreGenes_location = createPath(cwd, histFolder, step9_hist_plots)
#step4_TestMatrix_location = createPath(cwd, outputFolder, step4_csv_test)
#step5_TestMatrix_location = createPath(cwd, outputFolder, step5_csv_test)

# initiate start time for full analysis
start_all = time.time()

# load sample_file and groups_file into pd dataframes
sample_input_df = pd.read_csv(sample_file, header=0)
groups_input_df = pd.read_csv(groups_file, header=None)

# variants considered in analysis
relevant_variants = ['nonsynonymous SNV', 'stopgain SNV', 'synonymous SNV','stop loss','start loss','indel','fs-indel']

# open log file to collect time for each step
LogFile = open(LogFile_summary_txt_location, 'w')

# Step1: Generate basic stats on samples and biological groups
step1_start = time.time()
print('Now Performing Step1: Generating basic statistics on input samples and biological groups')
all_unique_group_genes, all_group_genes_with_duplicates = sample_and_groups_basic_stats(sample_input_df,
                                                                                        groups_input_df,
                                                                                        step1_summary_txt_location,
                                                                                        relevant_variants)
LogFile.write('Time to perform Step1: ' + str(time.time() - step1_start) + '\n')

# Step2: Generate sample genes x EA matrix
step2_start = time.time()
print('Now Performing Step2: Creating genes from input samples X EA scores matrix')
sample_gene_EA_matrix, all_sample_genes_EAscores, all_unique_sample_genes, errored_genes = generate_sample_gene_by_EAmatrix(relevant_variants,
                                                                                                             sample_input_df,
                                                                                                             step2_summary_txt_location)

LogFile.write('Time to perform Step2: ' + str(time.time() - step2_start) + '\n')

# Step3: Perform KS test on each sample gene to identify individually significant sample genes
step3_start = time.time()
print(
    'Now Performing Step3: Performing KS test on each gene in input samples to identify genes with significantly biased EA distributions')
all_sample_genes_floatEAscores, sig_single_sample_genes_lst = KS_test_individual_sample_genes(all_sample_genes_EAscores,
                                                                                              sample_gene_EA_matrix,
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
                                                                                        sample_gene_EA_matrix,
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
lst_small_sims = np.arange(2, 16, 1).tolist()
lst_large_sims = np.arange(15, 51, 5).tolist()
lst_large_sims.remove(15)
small_and_large_sims = lst_small_sims + lst_large_sims

# build simulations with parallelization
all_small_sims = pool_fx_sims(1000, lst_small_sims, sample_gene_EA_matrix, all_group_genes_with_duplicates,
                              sig_single_sample_genes_lst, errored_genes, number_cores)

start6_large_start = time.time()
all_large_sims_100 = pool_fx_sims(1000, lst_large_sims, sample_gene_EA_matrix, all_group_genes_with_duplicates,
                                  sig_single_sample_genes_lst, errored_genes, number_cores)

all_small_sims_flat = [item for sublist in all_small_sims for item in sublist]
all_large_sims_100_flat = [item for sublist in all_large_sims_100 for item in sublist]
all_sims_combined_flat = all_small_sims_flat + all_large_sims_100_flat
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

#Step9: Generate historgrams of significant biological groups
step9_start = time.time()
print('Now Performing Step9: Generating histograms of significant biological groups and core genes')
significant_groups_df = final_summary_matrix_updated.query("(passed_q_value_filter == 1) and (fold_better >= 1)")
print("Number significant pathways in final analysis: ", str(significant_groups_df.shape[0]))
significant_groups = list(significant_groups_df['group_name'])
for sig_group in significant_groups:
    full_biological_group_historgrams(sig_group, groups_input_df, sig_single_sample_genes_lst, sample_gene_EA_matrix, step9_HistCoreGenes_location)
    core_biological_group_historgrams(sig_group, significant_groups_df, sample_gene_EA_matrix, step9_HistCoreGenes_location)

LogFile.write('Time to perform Step9: ' + str(time.time() - step9_start) + '\n')

# calculate time to complete full analysis
print("Time to complete full analysis: ", time.time() - start_all)
LogFile.write('Time to perform full analysis - ' + str(time.time() - start_all)  + '\n')
LogFile.close()

# In[ ]:


# In[ ]:
