#!/usr/bin/env python3

"""
@author: Jenn Asmussen

Functions for extracting variant information from VEP-EA annotated VCF

"""

import pandas as pd 
import numpy as np 
import pysam
import sys
import time
import subprocess
import multiprocessing as mp

def getRefPopVariants(refPopVariantInfoFile):
    """Imports and processes reference variant file
    Args:
        refPopVariantInfoFile (path): Path to reference variant file with external AC, AF values
    Returns:
        refVariant_df (df): Reference variants in dataframe format with unique identifier
    """
    col_names = ['chr', 'pos', 'ref', 'alt', 'ref_AC', 'ref_AF']
    col_type = {'chr': str, 'pos': str, 'ref': str, 'alt': str, 'ref_AC': int}
    refVariant_df = pd.read_csv(refPopVariantInfoFile, sep='\t', names=col_names, dtype=col_type)

    refVariant_df['identifier'] = refVariant_df['chr'] + '-' + refVariant_df['pos'] + '-' + \
                                         refVariant_df['ref'] + '-' + refVariant_df['alt']
    start_count = refVariant_df.shape[0]
    refVariant_df.drop_duplicates('identifier', keep=False, inplace=True)
    end_count = refVariant_df.shape[0]
    if start_count != end_count:
        print('Warning: Duplicate variants exist in ref population file')
    else:
        pass
    return refVariant_df

def collectVCFvariants(input_vcf):
    """Extracts required VEP and EA annotations from input VCF
    Args:
        input_vcf (vcf file): VCF file with cohort variants annotated with VEP and EA scores
    Returns:
        DataFrame (df): Required VEP and EA annotations for each variant in input_vcf in df format
    """
    vcf_in = pysam.VariantFile(input_vcf, 'r')
    records = []
    for record in vcf_in.fetch():
        info = record.info

        try:
            ea_score = info['EA']
        except:
            ea_score = 'No_EA_Score'

        try:
            ensembl_proID = info['Ensembl_proteinid']
        except:
            ensembl_proID = 'No_Ensembl_ProID'


        records.append([record.contig, record.pos, record.ref, record.alts[0], info['Consequence'][0], info['SYMBOL'][0],
                        info['ENSP'][0], info['HGVSp'][0], ea_score, ensembl_proID, info['AC'][0]])


    vcf_in.close()

    cols = ['chr', 'pos', 'ref', 'alt', 'Consequence', 'SYMBOL', 'ENSP', 'HGVSp', 'EA', 'Ensembl_proteinid','Cohort_AC']
    col_type = {'chr': str, 'pos': str, 'ref': str, 'alt': str}

    record_df = pd.DataFrame(records, columns=cols)
    record_df = record_df.astype(col_type)

    return record_df

def selectTranscriptSubEA(ensp, EA, ensemblProteinid):
    """Extracts EA score from canonical ENSP transcript
    Args:
        ensp (str): VEP annotated canonical transcript ID
        EA (tuple): tuple of EA scores from transcripts assigned to variant
        ensemblProteinid (tuple): tuple of transcripts assigned to variant
    Returns:
        final_EA (str): EA score on canconical transcript
    """
    if type(EA) == tuple:
        try:
            indexVal = ensemblProteinid.index(ensp)
            final_ea = EA[indexVal]
        except:
            final_ea = 'no EA score'
    else:
        final_ea = EA
    return final_ea

def variant_class(csq):
    """Updates csq nomenclature to MAF format
    Args:
        csq (str): VEP consequence type of variant
    Returns:
        variant_class (str): Updated MAF consequence type of variant
    """
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
    """Converts missense EA scores to floats
    Args:
        ea (str): EA score extracted from canonical ENSP transcript
        variant_class (str): Updated MAF consequence type of variant
    Returns:
        finalEAFormat (str/float): Float version of EA score if not "no EA score" or LoF variant
    """
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
    """Processes variants parsed from input_vcf for canonical transcript, associated EA score, and filters for consequence type
    Args:
        parsedVCFVariantsMatrix (df): Dataframe of min/max AC filtered variants from cohort input_vcf
    Returns:
        parsedVCFVariantsMatrixCleaned (df): Processed variants with canonical transcript, AA change, csq, and EA scores
    """
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

    parsedVCFVariantsMatrixCleaned.rename(columns={'SYMBOL': 'gene_ID', 'HGVSp': 'AAchange', 'final_EA_Format': 'Action'}, inplace=True)

    return parsedVCFVariantsMatrixCleaned

def filterVCFvariants(var_df, refPopVariantFile, maxAC_threshold, minAC_threshold):
    """Filters cohort variants extracted from input_vcf based on user arguments
    Args:
        var_df (df): Required VEP and EA annotations for each variant in input_vcf in df format
        refPopVariantFile (path): If included, path to tab sep text file with reference allele counts for cohort variants
        maxAC_threshold (int): Max AC count of variant that will be included in analysis
        minAC_threshold (int): Min AC count of variant that will be included in analysis

    Returns:
        var_df (df): DataFrame of filtered and processed cohort variants
        refVariant_dict (dict): Dictionary of unique variant identifier and external AC (if passed as input arg)

    """
    var_df['identifier'] = var_df['chr'] + '-' + var_df['pos'] + '-' + var_df['ref'] + '-' + var_df['alt']
    print('Number of cohort variants pre-filtering:', var_df.shape[0])

    try:
        ## Filter cohort variants using ref population AC values
        refPopVariant_df = getRefPopVariants(refPopVariantFile)
        print('Number of variants in ref population:', refPopVariant_df.shape[0])
        refVariant_dict = dict(zip(refPopVariant_df.identifier, refPopVariant_df.ref_AC))
        var_df['refAC'] = var_df['identifier'].map(refVariant_dict)
        var_df = var_df.loc[(var_df['refAC'] >= minAC_threshold) & (var_df['refAC'] <= maxAC_threshold)]
        var_df = var_df[var_df['Cohort_AC'] != 0]
    except:
        ## Filter cohort variants using cohort AC values
        if refPopVariantFile == None:
            print('No ref population AC values provided. Using cohort AC for filtering.')
            var_df = var_df.loc[(var_df['Cohort_AC'] >= minAC_threshold) & (var_df['Cohort_AC'] <= maxAC_threshold)]
            refVariant_dict = None
        else:
            print('Ref population AC values provided, but not importing correctly.')
            print('Using cohort AC for filtering.')
            var_df = var_df.loc[(var_df['Cohort_AC'] >= minAC_threshold) & (var_df['Cohort_AC'] <= maxAC_threshold)]
            refVariant_dict = None

    ## Clean variant annotations from VEP/EA annotations
    var_df = createFinalVariantMatrix(var_df)
    print('Number of cohort variants post-filtering:', var_df.shape[0])

    return var_df, refVariant_dict

def pool_parseGT_fx(chunked_records, c, vcf_path):
    """Retrieves per sample GT of cohort variants extracted from input_vcf based on user arguments
    Args:
        chunked_records (list): list of variant chunks to be sent to each core
        c (int): Number of cores
        vcf_path (path): Path to input_vcf of cohort
    Returns:
        output (df): Chunked variant sites with associated GT per sample
    """
    args = tuple(zip(chunked_records, [vcf_path] * len(chunked_records)))
    pool = mp.Pool(processes=c)
    output = pool.map(parseGT_fx_stdin, args)
    pool.close()
    pool.join()
    return output

def parse_carriers(bcftools_output):
    """
    Parse bcftools query output to keep only samples carrying alt alleles.
    Removes 0/0, 0/., ./., ./0
    """
    carriers = []

    for line in bcftools_output:

        if not line:
            continue
        chrom, pos, ref, alt, *samples = line.split("\t")
        samples = [x.split(';') for x in samples]
        samples = [item for sublist in samples for item in sublist]

        # Filter samples by genotype
        alt_samples = []
        for s in samples:
            if "=" not in s:  # skip malformed
                continue
            sample, gt = s.split("=")
            if gt not in ("0/0", "0/.", "./.", "./0"):  # keep carriers
                alt_samples.append(sample)

        carriers.append([f"{chrom}-{pos}-{ref}-{alt}", alt_samples])

    return carriers

def parseGT_fx_stdin(args):
    "Parsed GT from chunked variant sites using bcftools commands"
    variant_sites, vcf = args

    updated_variant_sites = []
    for variant in variant_sites:
        chrom, pos, ref, alt = variant.split("-")
        updated_variant_sites.append(f"{chrom}\t{pos}\t{ref}\t{alt}")

    updated_variant_sites_str = "\n".join(updated_variant_sites)

    cmd = ["bcftools", "view", "-T", "/dev/stdin", vcf, "-Ov"]
    query_cmd = ["bcftools", "query", "-f", "%CHROM\t%POS\t%REF\t%ALT\t[%SAMPLE=%GT;]\n"]

    # Create the pipeline
    p1 = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # Send the sites string into bcftools view
    out1, err1 = p1.communicate(input=updated_variant_sites_str)

    # Now pass its output into bcftools query
    p2 = subprocess.Popen(query_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err2 = p2.communicate(input=out1)

    out = out.strip().split("\n") if out else []
    sample_dict = parse_carriers(out)
    return sample_dict

def chunk_list_gen(lst, n):
    '''Updates variant list chunk'''
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def getFilteredVCFvariantsGT(var_df, samples_path, ncores, vcf, refAC_dictionary,
                             maxACthreshold, minACthreshold):
    """Retrieves per sample GT of cohort variants extracted from input_vcf based on user arguments
    Args:
        var_df (df): DataFrame of filtered and processed cohort variants
        samples_path (path): Path to single column text file of sample IDs
        ncores (int): Number of cores for parallelization
        vcf (path): Path to input_vcf of cohort variants
        refAC_dictionary (dict): Dictionary of reference cohort AC values
        maxAC_threshold (int): Max AC count of variant that will be included in analysis
        minAC_threshold (int): Min AC count of variant that will be included in analysis

    Returns:
        var_df_expanded (df): DataFrame of cohort variants with cohort frequency representation and sample IDs
    """

    ## Collect sample IDs
    with open(samples_path,'r') as f:
        samples = f.readlines()
    samples = [x.strip('\n') for x in samples]
    print('Number of samples in analysis:', len(samples))

    ## Collect and chunk variants for extracting genotypes
    variant_sites = var_df['identifier'].tolist()
    var_per_chunk = int((1*10**9)/(len(samples) * (0.25*10**3)))

    split_variant_sites = []
    for chunk in chunk_list_gen(variant_sites, var_per_chunk):
        split_variant_sites.append(chunk)

    ## Parse genotypes
    variant_sites_gt = pool_parseGT_fx(split_variant_sites, int(ncores), vcf)
    variant_sites_gt = [item for sublist in variant_sites_gt for item in sublist]

    variant_sites_gt_sampleFiltered = []
    for i in variant_sites_gt:
        i_samples = [x for x in i[1] if x in samples]
        if len(i_samples) == 0:
            pass
        else:
            variant_sites_gt_sampleFiltered.append([i[0], i_samples])

    variant_sites_gt_df = pd.DataFrame(variant_sites_gt_sampleFiltered)
    variant_sites_gt_df.rename(columns={0:'identifier',1:'samples'}, inplace=True)

    if refAC_dictionary != None:
        variant_sites_gt_df['refAC'] = variant_sites_gt_df['identifier'].map(refAC_dictionary)
        variant_sites_gt_df = variant_sites_gt_df.loc[(variant_sites_gt_df['refAC']<=maxACthreshold)&
                                                    (variant_sites_gt_df['refAC']>=minACthreshold)]
    else:
        ## Grab cohort_AC_dictionary as reference AC for filtering if no ref AC provided
        cohortAC_dictionary = dict(zip(var_df['identifier'], var_df['Cohort_AC']))
        variant_sites_gt_df['Cohort_AC'] = variant_sites_gt_df['identifier'].map(cohortAC_dictionary)
        variant_sites_gt_df = variant_sites_gt_df.loc[(variant_sites_gt_df['Cohort_AC'] <= maxACthreshold) &
                                                      (variant_sites_gt_df['Cohort_AC'] >= minACthreshold)]

    variant_sites_gt_df.dropna(axis=1, how='all', inplace = True)
    variant_sites_gt_df_dict = variant_sites_gt_df.set_index('identifier')['samples'].to_dict()

    var_df = var_df.copy()
    var_df['samples'] = var_df['identifier'].map(variant_sites_gt_df_dict)
    var_df['samples'].fillna(0, inplace = True)
    var_df = var_df[var_df['samples']!=0]
    var_df['dup_count'] = var_df['samples'].apply(lambda x: len(x)) ## Duplicate rows to match sample representation in cohort
    var_df_expanded = var_df.loc[np.repeat(var_df.index, var_df['dup_count'])].reset_index(drop=True)
    try:
        var_df_expanded = var_df_expanded[['gene_ID', 'Variant_classification', 'AAchange', 'Action', 'refAC',
                                           'samples','Cohort_AC']]
    except:
        var_df_expanded = var_df_expanded[['gene_ID', 'Variant_classification', 'AAchange', 'Action', 'samples',
                                           'Cohort_AC']]
    return var_df_expanded