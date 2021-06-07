#!/usr/bin/env python
# coding: utf-8

# In[ ]:

# function includes sig_single_genes when generating sim paths
# function uses all genes in groups (includes duplicates) to generate simulated pathways
# note - AK did not include sig_single_genes when building sim paths
# this code includes sig_single_genes when making simulations, but removes them for LOO analysis
# including sig_single_genes should make this more like real groups

# function generates nested list of simulated pathways with EA scores
# function input: num sims, gene x EA matrix, ALL group input genes, sig_single_genes

import numpy as np
import pandas as pd
import random
import multiprocessing as mp

def generate_simulated_groups(total_simulations, simulation_path_size, gene_EA_matrix, ALL_group_input_genes, sig_single_genes_lst, errors):
    
    lst_all_genes = ALL_group_input_genes
    # the following line removes sig_single_genes from list of group input genes prior to building simulations
    lst_all_genes = [elem for elem in lst_all_genes if elem not in sig_single_genes_lst]
    # the following line removes errored genes from list of gorup input genes prior to building simulations
    lst_all_genes = [elem for elem in lst_all_genes if elem not in errors]

    lst_sim_pathways = []
    
    for sim in range(total_simulations):
        sim_genes = random.sample(lst_all_genes, k = simulation_path_size)
        if len(sim_genes) != len(set(sim_genes)):
            while len(sim_genes) != len(set(sim_genes)):
                new_sim_genes = []
                set_sim_genes = set(sim_genes)
                num_replacements = simulation_path_size - len(set_sim_genes)
                replacements = random.sample(lst_all_genes, k = num_replacements)
                new_sim_genes.extend(list(set_sim_genes))
                new_sim_genes.extend(replacements)
                sim_genes = new_sim_genes.copy()
            lst_sim_pathways.append(sim_genes)
        else:
            lst_sim_pathways.append(sim_genes)
    
    lst_final_sim_paths_NoSigGenes = []
    i = 1
    for sim_path in lst_sim_pathways:
        final_path = [str(simulation_path_size)+'_sim_'+str(i)]
        i += 1
        for gene in sim_path:
            if gene in sig_single_genes_lst:
                pass
            else:
                gene_name_with_EA_scores = []
                gene_name_with_EA_scores.append(gene)
                for row in range(gene_EA_matrix.shape[0]):
                    if gene_EA_matrix[row,0] == gene:
                        gene_ea_scores = gene_EA_matrix[row, 1:gene_EA_matrix.shape[1]]
                        gene_ea_scores = [x for x in gene_ea_scores if x != None]
                        gene_ea_scores = [x for x in gene_ea_scores if x != 'synon']
                        gene_ea_scores = [float(x) for x in gene_ea_scores]
                        gene_name_with_EA_scores.extend(gene_ea_scores)
                    else:
                        pass
                final_path.append(gene_name_with_EA_scores)
        lst_final_sim_paths_NoSigGenes.append(final_path)
    return lst_final_sim_paths_NoSigGenes

def build_sims_multiprocessing(arg):
    num_sims = arg[0]
    size_sims = arg[1]
    sample_gene_EA_matrix = arg[2]
    group_input_genes = arg[3]
    sig_genes = arg[4]
    error_genes = arg[5]
    simulations = generate_simulated_groups(num_sims, size_sims, sample_gene_EA_matrix, group_input_genes, sig_genes, error_genes)
    return simulations

def pool_fx_sims(number_simulations, simulation_size_lst, gene_EA_matrix, ALL_group_input_genes, sig_single_genes_lst, error_gene_lst, cores):
    args = tuple(zip(np.full(len(simulation_size_lst), number_simulations).tolist(),
                    simulation_size_lst, [gene_EA_matrix] * len(simulation_size_lst), 
                    [ALL_group_input_genes] * len(simulation_size_lst),
                     [sig_single_genes_lst] * len(simulation_size_lst),
                     [error_gene_lst] * len(simulation_size_lst)))
    pool = mp.Pool(processes=cores)
    output = pool.map(build_sims_multiprocessing, args)
    pool.close()
    pool.join()
    
    return output