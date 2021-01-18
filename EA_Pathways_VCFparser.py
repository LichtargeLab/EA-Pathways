#!/usr/bin/env python3

"""
@author: Jenn Asmussen

Pilot script to extract variant information from VCF for running EA-Pathways

"""

import pandas as pd 
import numpy as np 
from pysam import VariantFile
import sys
import csv
import time

analysisName = sys.argv[1]
vcfFile = sys.argv[2]
patientFile = sys.argv[3]
refPopVariantFile = sys.argv[4]
refPopVariantThreshold = sys.argv[5]
outputPath = sys.argv[6]


def getCaseControlIds(patientLabelFile):
    patientLabel_df = pd.read_csv(patientLabelFile, header = None)
    caseIDs = list(patientLabel_df.loc[patientLabel_df[1] == 1][0])
    controlIDs = list(patientLabel_df.loc[patientLabel_df[1] == 0][0])
    return caseIDs, controlIDs

def convert_UKB_AF(x):
    try:
        item_new = float(x)
    except ValueError:
        item_new = 1.1
    return item_new

def getRefPopVariants(refPopVariantInfoFile):
    col_names = ['chr', 'pos', 'ref', 'alt', 'ref_AC', 'ref_AF']
    col_type = {'chr': str, 'pos': str, 'ref': str, 'alt': str, 'ref_AC': int, 'ref_AF': str}
    refVariant_df = pd.read_csv(refPopVariantInfoFile, sep='\t', names=col_names, dtype=col_type)

    refVariant_df['identifier'] = refVariant_df['chr'] + '-' + refVariant_df['pos'] + '-' + \
                                         refVariant_df['ref'] + '-' + refVariant_df['alt']
    refVariant_df['ref_AF_updated'] = refVariant_df['ref_AF'].apply(convert_UKB_AF)
    refVariant_df.drop_duplicates('identifier', keep=False, inplace=True)
    return refVariant_df

def getRefVariantsUnderThreshold(refVariant_df):
    refVariantThreshold_df = refVariant_df.loc[(refVariant_df['ref_AC'] <= int(refPopVariantThreshold)) & (refVariant_df['ref_AC'] != 0)]
    refVariantThreshold_dict = dict(zip(refVariantThreshold_df.identifier, refVariantThreshold_df.ref_AC))
    return refVariantThreshold_dict

def selectTranscriptSubEA(transcript, sub, EA, gene):
    if type(gene) == tuple:
        final_gene = gene[0]
    else:
        final_gene = gene
    final_EA_sub_transcript = [EA[0], sub[0], transcript[0], final_gene]
    return final_EA_sub_transcript

def variant_class(final_EA):
    if final_EA == 'STOP':
        variant_class = 'stopgain SNV'
    elif final_EA == 'fs-indel':
        variant_class = 'fs-indel'
    elif final_EA == 'indel':
        variant_class = 'indel'
    elif final_EA == 'no_STOP':
        variant_class = 'stop loss'
    elif final_EA == 'silent':
        variant_class = 'synonymous SNV'
    elif final_EA == 'START_loss':
        variant_class = 'start loss'
    else:
        final_EA_float = float(final_EA)
        variant_class = 'nonsynonymous SNV'
    return variant_class

def createFinalVariantMatrix(parsedVCFVariantsMatrix):
    snvs_to_filter = ['.', 'UNKNOWN']
    EA_variants_to_drop = ['no_trace', '.', 'no_gene', 'no_action']
    EA_Reactome_dict = {'silent': '', 'STOP': '', 'fs-indel': '', 'indel': '', 'no_STOP': '', 'START_loss': ''}

    parsedVCFVariantsMatrixCleaned = parsedVCFVariantsMatrix.copy()
    parsedVCFVariantsMatrixCleaned = parsedVCFVariantsMatrixCleaned.loc[~parsedVCFVariantsMatrixCleaned.gene.isin(snvs_to_filter)]
    parsedVCFVariantsMatrixCleaned['final_EA_sub_transcript'] = parsedVCFVariantsMatrixCleaned.apply(lambda x: selectTranscriptSubEA(x['NM'], x['sub'], x['EA'], x['gene']), axis=1)
    parse_cols = parsedVCFVariantsMatrixCleaned['final_EA_sub_transcript'].apply(pd.Series)
    parse_cols = parse_cols.rename(columns=lambda x: 'value_' + str(x))
    parsedVCFVariantsMatrixCleaned_final = pd.concat([parsedVCFVariantsMatrixCleaned[:], parse_cols[:]], axis=1)
    parsedVCFVariantsMatrixCleaned_final.rename(columns={'value_0': 'Final_EA', 'value_1': 'Final_Sub', 'value_2': 'Final_Transcript', 'value_3':'Final_Gene'}, inplace=True)
    parsedVCFVariantsMatrixCleaned_final = parsedVCFVariantsMatrixCleaned_final.loc[~parsedVCFVariantsMatrixCleaned_final['Final_EA'].isin(EA_variants_to_drop)]
    parsedVCFVariantsMatrixCleaned_final['Variant_classification'] = parsedVCFVariantsMatrixCleaned_final['Final_EA'].apply(variant_class)
    parsedVCFVariantsMatrixCleaned_final.rename(columns={'Final_Gene': 'gene_ID', 'Final_Sub': 'AAchange', 'Final_EA': 'Action'}, inplace=True)
    parsedVCFVariantsMatrixCleaned_final.replace({'Action': EA_Reactome_dict}, inplace=True)

    return parsedVCFVariantsMatrixCleaned_final


#code for parsting VCF into EA-Pathways input file
cases, controls = getCaseControlIds(patientFile)
allPatients = cases + controls
print('Number cases, controls:', len(cases),',' ,len(controls))

refPopVariant_df = getRefPopVariants(refPopVariantFile)
print('Number of variants in ref population:', refPopVariant_df.shape[0])

refPopVariantThreshold_dict = getRefVariantsUnderThreshold(refPopVariant_df)
print('Number of variants with AC <=',str(refPopVariantThreshold),':',len(refPopVariantThreshold_dict))

vcf = VariantFile(vcfFile)

start = time.time()
rows = []

for var in vcf:
    vcfVariantID = str(var.chrom) + '-' +  str(var.pos) + '-' + str(var.ref) + '-' + str(var.alts[0])
    if vcfVariantID in refPopVariantThreshold_dict:
        for sample in allPatients:
            gt = var.samples[str(sample)]['GT']
            if 1 in gt:
                rows.append([var.chrom, var.pos, var.ref, var.alts[0], sample, var.info['gene'],
                            var.info['NM'], var.info['sub'], var.info['EA'], var.samples[str(sample)]['GT']])
            else:
                pass

cols = ['chr','pos','ref','alt','sample','gene','NM','sub','EA','GT']
parsedVCFforReactomes_df = pd.DataFrame(rows, columns = cols)

parsedVCFforReactomes_final_df = createFinalVariantMatrix(parsedVCFforReactomes_df)

parsedVCFforReactomes_final_cases_df = parsedVCFforReactomes_final_df.loc[parsedVCFforReactomes_final_df['sample'].isin(cases)]
parsedVCFforReactomes_final_controls_df = parsedVCFforReactomes_final_df.loc[parsedVCFforReactomes_final_df['sample'].isin(controls)]

parsedVCFforReactomes_final_cases_df[['gene_ID','Variant_classification','AAchange','Action','sample']].to_csv(outputPath + analysisName +'_Cases_ReactomeInput.csv', index = False)
parsedVCFforReactomes_final_controls_df[['gene_ID','Variant_classification','AAchange','Action','sample']].to_csv(outputPath + analysisName +'_Controls_ReactomeInput.csv', index = False)

print('Time to parse and prep Reactome variants:', time.time() - start)


