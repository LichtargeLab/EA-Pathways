#!/usr/bin/env python3

"""
@author: Jenn Asmussen

Pilot script to extract variant information from VEP-EA annotated VCF for running EA-Pathways

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
minRefPopVariantThreshold = sys.argv[5]
maxRefPopVariantThreshold = sys.argv[6]
outputPath = sys.argv[7]


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
    #refVariant_df['ref_AF_updated'] = refVariant_df['ref_AF'].apply(convert_UKB_AF)
    refVariant_df.drop_duplicates('identifier', keep=False, inplace=True)
    return refVariant_df

def getRefVariantsUnderThreshold(refVariant_df):
    refVariantThreshold_df = refVariant_df.loc[(refVariant_df['ref_AC'] <= int(maxRefPopVariantThreshold)) & (refVariant_df['ref_AC'] != 0)]
    refVariantThreshold_dict = dict(zip(refVariantThreshold_df.identifier, refVariantThreshold_df.ref_AC))
    return refVariantThreshold_dict

def selectTranscriptSubEA(ensp, EA, ensemblProteinid):
    if type(EA) == tuple:
        try: #if EA score is None value for that transcript, this returns input EA tuple
            indexVal = ensemblProteinid.index(ensp)
            final_ea = EA[indexVal]
            final_ea = round(final_ea,2)
        except:
            final_ea = EA
    else:
        final_ea = EA
    return final_ea

def variant_class(csq):
    if "frameshift" in csq:
        variant_class = 'fs-indel'
    elif "splice_acceptor_variant" in csq:
        variant_class = 'splice site'
    elif "splice_donor_variant" in csq:
        variant_class = 'splice site'
    elif "stop_gained" in csq:
        variant_class = 'stopgain SNV'
    elif "start_lost" in csq:
        variant_class = 'start loss'
    elif "stop_lost" in csq:
        variant_class = 'stop loss'
    elif "missense" in csq:
        variant_class = 'nonsynonymous SNV'
    elif "inframe_deletion" in csq:
        variant_class = 'indel'
    elif "inframe_insertion" in csq:
        variant_class = 'indel'
    elif "5_prime_UTR_variant" in csq:
        variant_class = '5_prime_UTR_variant'
    elif "3_prime_UTR_variant" in csq:
        variant_class = '3_prime_UTR_variant'

    return variant_class

def getFinalEAFormat(ea, variant_class):
    emptyEA = ['fs-indel','splice site','stopgain SNV','start loss','stop loss','indel','5_prime_UTR_variant',
               '3_prime_UTR_variant']
    if variant_class in emptyEA:
        finalEAFormat = ''
    elif variant_class == 'nonsynonymous SNV':
        try:
            finalEAFormat = float(ea)
        except:
            finalEAFormat = 'no EA score'
    return finalEAFormat

def createFinalVariantMatrix(parsedVCFVariantsMatrix):
    ensp_to_filter = ['.']
    parsedVCFVariantsMatrixCleaned = parsedVCFVariantsMatrix.copy()
    parsedVCFVariantsMatrixCleaned = parsedVCFVariantsMatrixCleaned.loc[~parsedVCFVariantsMatrixCleaned.ENSP.isin(ensp_to_filter)]
    parsedVCFVariantsMatrixCleaned['final_EA'] = parsedVCFVariantsMatrixCleaned.apply(lambda x: selectTranscriptSubEA(x['ENSP'], x['EA'], x['Ensembl_proteinid']), axis=1)
    csq_lst = ['frameshift_variant','missense_variant','inframe_insertion','inframe_deletion','splice_acceptor_variant',
               'splice_donor_variant','start_lost','stop_gained','stop_lost','5_prime_UTR_variant','3_prime_UTR_variant']
    conseqMask = parsedVCFVariantsMatrixCleaned['Consequence'].apply(lambda x: any([c in x for c in csq_lst]))
    parsedVCFVariantsMatrixCleaned = parsedVCFVariantsMatrixCleaned[conseqMask]
    parsedVCFVariantsMatrixCleaned['Variant_classification'] = parsedVCFVariantsMatrixCleaned['Consequence'].apply(variant_class)
    parsedVCFVariantsMatrixCleaned['final_EA_Format'] = parsedVCFVariantsMatrixCleaned.apply(lambda x: getFinalEAFormat(x['final_EA'], x['Variant_classification']), axis=1)

    parsedVCFVariantsMatrixCleaned['identifier'] = parsedVCFVariantsMatrixCleaned['chr'] + '-' + \
                                                   parsedVCFVariantsMatrixCleaned['pos'] + '-' + \
                                                   parsedVCFVariantsMatrixCleaned['ref'] + '-' + \
                                                   parsedVCFVariantsMatrixCleaned['alt']
    parsedVCFVariantsMatrixCleaned['refPop_AC'] = parsedVCFVariantsMatrixCleaned['identifier'].map(refPopVariantThreshold_dict)
    parsedVCFVariantsMatrixCleaned.rename(columns={'SYMBOL': 'gene_ID', 'HGVSp': 'AAchange', 'final_EA_Format': 'Action'}, inplace=True)

    return parsedVCFVariantsMatrixCleaned

def createACoutputFiles(final_variant_can_df):

    for ac in range(int(refPopVariantThreshold) + 1):
        if ac == 0:
            pass
        else:
            final_variant_can_df_AC = final_variant_can_df.copy()
            final_variant_can_df_AC = final_variant_can_df_AC.loc[
                final_variant_can_df_AC['refPop_AC'] <= int(ac)]

            final_case_df = final_variant_can_df_AC.loc[final_variant_can_df_AC['sample'].isin(cases)]
            final_control_df = final_variant_can_df_AC.loc[final_variant_can_df_AC['sample'].isin(controls)]

            final_case_df[['gene_ID', 'Variant_classification', 'AAchange', 'Action', 'sample', 'refPop_AC']].to_csv(
                outputPath + analysisName + '_Cases_PathwaysInput_AC' + str(ac) + '.csv',
                index=False)
            final_control_df[['gene_ID', 'Variant_classification', 'AAchange', 'Action', 'sample', 'refPop_AC']].to_csv(
                outputPath + analysisName + '_Controls_PathwaysInput_AC' + str(ac) + '.csv',
                index=False)

def createACoutputFiles2(final_variant_can_df):

    ac_lst = np.arange(int(minRefPopVariantThreshold), int(maxRefPopVariantThreshold) + 1, 1).tolist()
    for ac in ac_lst:
        final_variant_can_df_AC = final_variant_can_df.copy()
        final_variant_can_df_AC = final_variant_can_df_AC.loc[final_variant_can_df_AC['refPop_AC'] <= int(ac)]

        final_case_df = final_variant_can_df_AC.loc[final_variant_can_df_AC['sample'].isin(cases)]
        final_control_df = final_variant_can_df_AC.loc[final_variant_can_df_AC['sample'].isin(controls)]

        final_case_df[['gene_ID', 'Variant_classification', 'AAchange', 'Action', 'sample', 'refPop_AC']].to_csv(
            outputPath + analysisName + '_Cases_PathwaysInput_AC' + str(ac) + '.csv',
            index=False)
        final_control_df[['gene_ID', 'Variant_classification', 'AAchange', 'Action', 'sample', 'refPop_AC']].to_csv(
            outputPath + analysisName + '_Controls_PathwaysInput_AC' + str(ac) + '.csv',
            index=False)


#code for parsing VCF into EA-Pathways input file
cases, controls = getCaseControlIds(patientFile)
allPatients = cases + controls
print('Number cases, controls:', len(cases),',' ,len(controls))

refPopVariant_df = getRefPopVariants(refPopVariantFile)
print('Number of variants in ref population:', refPopVariant_df.shape[0])

refPopVariantThreshold_dict = getRefVariantsUnderThreshold(refPopVariant_df)
print('Number of variants with AC <=',str(maxRefPopVariantThreshold),':',len(refPopVariantThreshold_dict))

vcf = VariantFile(vcfFile)

start = time.time()
rows = []

for var in vcf:
    vcfVariantID = str(var.chrom) + '-' +  str(var.pos) + '-' + str(var.ref) + '-' + str(var.alts[0])
    if vcfVariantID in refPopVariantThreshold_dict:
        if 'synonymous_variant' in var.info['Consequence'][0]:
            pass
        elif 'missense_variant' in var.info['Consequence'][0]:
            for sample in allPatients:
                gt = var.samples[str(sample)]['GT']
                if 1 in gt and '.' not in gt:
                    rows.append([var.chrom, var.pos, var.ref, var.alts[0], sample, var.info['SYMBOL'][0],
                                 var.info['ENSP'][0], var.info['HGVSp'][0],
                                 var.samples[str(sample)]['GT'],
                                 var.info['Consequence'][0], var.info['EA'], var.info['Ensembl_proteinid']])
                else:
                    pass
        else:
            for sample in allPatients:
                gt = var.samples[str(sample)]['GT']
                if 1 in gt and '.' not in gt:
                    rows.append([var.chrom, var.pos, var.ref, var.alts[0], sample, var.info['SYMBOL'][0],
                                 var.info['ENSP'][0], var.info['HGVSp'][0],
                                 var.samples[str(sample)]['GT'],
                                 var.info['Consequence'][0], '.', '.'])
                else:
                    pass

cols = ['chr','pos','ref','alt','sample','SYMBOL','ENSP','HGVSp','GT','Consequence','EA','Ensembl_proteinid']
col_type = {'chr': str, 'pos': str, 'ref': str, 'alt': str}
parsedVCFforReactomes_df = pd.DataFrame(rows, columns = cols)
parsedVCFforReactomes_df = parsedVCFforReactomes_df.astype(col_type)

parsedVCFforReactomes_final_df = createFinalVariantMatrix(parsedVCFforReactomes_df)
parsedVCFforReactomes_final_df.to_csv(outputPath + 'VEP_parsing_pilot.csv', index = False)

createACoutputFiles2(parsedVCFforReactomes_final_df)

print('Time to parse and prep Reactome variants:', time.time() - start)


