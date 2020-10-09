#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# v3 - version corrects for synon SNV
# function will be distributed to different cores with list item
# currently written to take one item from nested list and compare to background EA scores
# output group name, original group KS test p-value, core genes, core genes KS test p-value

import numpy as np
import pandas as pd
import statistics
import time
from scipy import stats as statistics
import multiprocessing as mp

def input_samples_loo_multiprocessing(arg):
    simulation = arg[0]
    background = arg[1]
    name, ini_pvalue, core_gen, cor_pvalue = group_LOO_core_gene_analysis(simulation, background)
    return name, ini_pvalue, core_gen, cor_pvalue

def pool_loo_analysis_input_samples_fx(all_simulations, EA_background, cores):
    args = tuple(zip(all_simulations, [EA_background] * len(all_simulations)))
    pool = mp.Pool(processes=cores)
    output = pool.map(input_samples_loo_multiprocessing, args)
    pool.close()
    pool.join()
    return output


def group_LOO_core_gene_analysis(biological_group_noSigGenes_EAscores_item, background_EAscores):
    
    background_float_EAscores = [float(x) for x in background_EAscores]
    
    group_name = biological_group_noSigGenes_EAscores_item[0]
    # uncomment the following line if you want to see progress during LOO analysis
    # print(group_name)
    group_length = len(biological_group_noSigGenes_EAscores_item)
    
    # collect group EA scores
    group_all_EA_scores = []
    for gene in biological_group_noSigGenes_EAscores_item[1:group_length]:
        if len(gene) == 1:
            pass
        else:
            group_all_EA_scores.append(gene[1:len(gene)])
    
    group_all_EA_scores = [item for sublist in group_all_EA_scores for item in sublist]
    group_all_EA_scores = [x for x in group_all_EA_scores if x != 'synon']
    group_all_EA_scores = [float(x) for x in group_all_EA_scores]
    
    # perform KS test on group EA scores to generate original p-value
    if len(group_all_EA_scores) == 0:
        initial_group_pvalue = 1
    else:
        initial_group_pvalue = statistics.mstats.ks_twosamp(group_all_EA_scores, background_float_EAscores, alternative='less')[1]
    
    # perform LOO-KS analysis if initial_group_pvalue < 1, else pass
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
                
                # perform KS test on group subset to generate new p-value
                if len(loo_group_EA_scores) == 0:
                    group_core_genes.append(test_gene[0])
                    group_core_EA_scores.append(test_gene[1:len(test_gene)])
                else:
                    test_gene_loo_pvalue = statistics.mstats.ks_twosamp(loo_group_EA_scores, background_float_EAscores, alternative='less')[1]
                    
                    if test_gene_loo_pvalue > initial_group_pvalue:
                        group_core_genes.append(test_gene[0])
                        group_core_EA_scores.append(test_gene[1:len(test_gene)])
                    else:
                        pass   
            
            else:
                pass
            
    else:
        # generate placeholders for whatever is generated in previous if statement
        # append "no core genes" to list where core genes are being collected
        group_core_genes.append('No nonsyn SNV mutations in biological group')
    
    group_core_EA_scores = [item for sublist in group_core_EA_scores for item in sublist]
    group_core_EA_scores = [x for x in group_core_EA_scores if x != 'synon']
    
    # perform KS test on core gene EA scores to generate final p-value
    if len(group_core_EA_scores) == 0:
        core_group_pvalue = 1
    else:
        core_group_pvalue = statistics.mstats.ks_twosamp(group_core_EA_scores, background_float_EAscores, alternative='less')[1]

    return group_name, initial_group_pvalue, group_core_genes, core_group_pvalue    

