#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import numpy as np
import pandas as pd
import multiprocessing as mp
import time
from EA_Pathways_Step5_Sample_LOO_Analysis import group_LOO_core_gene_analysis

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

def collect_sim_core_pvalues_and_percentiles(list_all_group_sizes, LOO_KS_sim_output):
    all_core_pvalues_lst = []
    all_core_percentiles_lst = []
    for sim_size in list_all_group_sizes:
        sim_name = "simulations_size_"+str(sim_size)
        sim_core_pvalues = []
        sim_core_pvalues.append(sim_name)
        for item in LOO_KS_sim_output:
            loo_simulation_output_name = item[0]
            result = loo_simulation_output_name.startswith(str(sim_size)+'_')
            if result == False:
                pass
            else:
                test_core_pvalue = item[3]
                sim_core_pvalues.append(test_core_pvalue)
        all_core_pvalues_lst.append(sim_core_pvalues)

        #prep list for conversion to percentile
        sim_core_pvalues_copy = sim_core_pvalues.copy()
        sim_core_pvalues_copy.remove(sim_name)
        sim_core_percentile = []
        sim_core_percentile.append(sim_name)
        for perc in range(100):
            sim_core_percentile.append(np.percentile(sim_core_pvalues_copy, perc))
        all_core_percentiles_lst.append(sim_core_percentile)
    
    return all_core_pvalues_lst, all_core_percentiles_lst