# python packages and libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, AutoMinorLocator

def full_biological_group_historgrams(group_name, group_input_df, sig_genes, sample_gene_EA_matrix, output_directory):

    for row in range(group_input_df.shape[0]):
        if group_input_df.iloc[row,0] == group_name:
            full_group_genes = list(group_input_df.iloc[row, 2:group_input_df.shape[1]].dropna())
        else:
            pass

    full_group_genes = [x for x in full_group_genes if x not in sig_genes]

    full_group_genes_for_plot = []
    full_group_genes_EA_scores = []
    for gene in full_group_genes:
        for row in range(sample_gene_EA_matrix.shape[0]):
            if sample_gene_EA_matrix[row, 0] == gene:
                #print(gene)
                full_group_genes_for_plot.append(gene)
                gene_EA_scores = sample_gene_EA_matrix[row, 1:sample_gene_EA_matrix.shape[1]]
                #print(gene_EA_scores)
                gene_EA_scores = [x for x in gene_EA_scores if x != None]
                gene_EA_scores = ['0' if x == 'synon' else x for x in gene_EA_scores]
                gene_EA_scores = [float(x) for x in gene_EA_scores]
                full_group_genes_EA_scores.append(gene_EA_scores)

    n_bins = 10
    #plt.switch_backend('agg')
    fig, ax = plt.subplots()
    ax.hist(full_group_genes_EA_scores, n_bins, range = (0.0, 100.0), histtype = 'barstacked', stacked = True, label = full_group_genes_for_plot)
    ax.set_xlim(0,100)
    ax.set_ylim(0,30)
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax.yaxis.set_major_locator(MultipleLocator(5))
    ax.yaxis.set_minor_locator(MultipleLocator(1))
    plt.legend(loc = 'upper right')
    plt.title(str(group_name))
    plt.savefig(output_directory + str(group_name) + "_FullGroupHist_NoSigGenes.png", transparent = True)
    plt.close()

def core_biological_group_historgrams(group_name, significant_groups_df, sample_gene_EA_matrix, output_directory):
    for row in range(significant_groups_df.shape[0]):
        if significant_groups_df.iloc[row, 0] == group_name:
            group_name_info = significant_groups_df.iloc[row].copy()
            group_core_genes = group_name_info['core_genes']
            #print(group_core_genes)
            #print(len(group_core_genes))
            #print(type(group_core_genes))

    group_core_genes_EA_scores = []
    for gene in group_core_genes:
        for row in range(sample_gene_EA_matrix.shape[0]):
            if sample_gene_EA_matrix[row, 0] == gene:
                gene_EA_scores = sample_gene_EA_matrix[row, 1:sample_gene_EA_matrix.shape[1]]
                gene_EA_scores = [x for x in gene_EA_scores if x != None]
                gene_EA_scores = ['0' if x == 'synon' else x for x in gene_EA_scores]
                gene_EA_scores = [float(x) for x in gene_EA_scores]
                group_core_genes_EA_scores.append(gene_EA_scores)

    n_bins = 10
    #plt.switch_backend('agg')
    fig, ax = plt.subplots()
    ax.hist(group_core_genes_EA_scores, n_bins, range = (0.0, 100.0), histtype = 'barstacked', stacked = True, label = group_core_genes)
    ax.set_xlim(0,100)
    ax.set_ylim(0,30)
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax.yaxis.set_major_locator(MultipleLocator(5))
    ax.yaxis.set_minor_locator(MultipleLocator(1))
    plt.legend(loc = 'upper right')
    plt.title(str(group_name))
    plt.savefig(output_directory + str(group_name) + "_CoreGenesHist_NoSigGenes.png", transparent = True)
    plt.close()
