import os
import sys
sys.path.append(os.path.dirname(__file__))

from distributions.CloneTreeGenotypes import CloneTreeGenotypes
from genotypes.genotypes import list_all_genotypes

import re
import argparse
import numpy as np
import pandas as pd
import pymc3 as pm
import seaborn as sns
import scipy as sc
import theano
import theano.tensor as tt
import pickle
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_cpn(integercpn_selected, variables, outfname):
    integercpn_selected[['region'] + variables]

    integercpn_melted = pd.melt(
        integercpn_selected, id_vars='region', value_vars=variables)

    length_fig, length_ax = plt.subplots()
    sns.boxplot(
        x='region',
        y='value',
        hue='variable',
        data=integercpn_melted,
        linewidth=2.5,
        ax=length_ax)

    length_fig.savefig(outfname)


def ctg_transition_mask(all_genotypes):
    cn_genotype_matrix = all_genotypes.as_matrix()
    num_genotypes = cn_genotype_matrix.shape[0]

    valid_transitions = np.zeros(shape=(num_genotypes, num_genotypes))

    ## i is from, j is to
    for i in range(num_genotypes):
        geno_i = cn_genotype_matrix[i]
        loha_i = (geno_i[0] == 0)
        lohb_i = (geno_i[0] == geno_i[1])
        for j in range(num_genotypes):
            geno_j = cn_genotype_matrix[j]
            loha_j = (geno_j[0] == 0)
            lohb_j = (geno_j[0] == geno_j[1])
            if not ((loha_i & ~loha_j) | (lohb_i & ~lohb_j)):
                valid_transitions[i, j] = 1
            else:
                valid_transitions[i, j] = 0

    np.fill_diagonal(valid_transitions, val=0.)
    num_transitions = np.sum(valid_transitions, axis=1)

    outputs = {
        'valid_transitions': valid_transitions,
        'num_transitions': num_transitions,
        'num_genotypes': num_genotypes,
        'cn_genotype_matrix': cn_genotype_matrix
    }
    return outputs


def process_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--clonal_prevalence_file",
        dest="clonal_prevalence_file",
        help="Clonal prevalence file",
        metavar="FILE")
    parser.add_argument(
        "--flagstat_file",
        dest="flagstat_file",
        help="Total unique read count file",
        metavar="FILE")
    parser.add_argument(
        "--hlaloss_file",
        dest="hlaloss_file",
        help="HLAloss table (from LOHHLA)",
        metavar="FILE")
    parser.add_argument(
        "--integercpn_file",
        dest="integercpn_file",
        help="IntegerCPN table (from LOHHLA)",
        metavar="FILE")
    parser.add_argument(
        "--tree_edges_file",
        dest="tree_edges_file",
        help="Tree edges file)",
        metavar="FILE")
    parser.add_argument(
        "--ploidy_cellularity_file",
        dest="ploidy_cellularity_file",
        help="Ploidy and cellularity information (i.e. identical input to LOHHLA)",
        metavar="FILE")
    parser.add_argument(
        "--anchor_model_type",
        dest="anchor_model_type",
        help="Anchor model type",
        default="nb",
        choices=['nb', 'mult_factor'],
        metavar="character")
    parser.add_argument(
        "--anchor_mode",
        dest="anchor_mode",
        help="Anchoring type for model",
        default="snvcn",
        choices=['snvcn', 'binmedian'],
        metavar="character")
    parser.add_argument(
        "-o", "--outdir", dest="outdir", help="Output directory", metavar="DIR")

    args = parser.parse_args()
    
    return args



def write_traces(traces, outfname):
    with open(outfname, 'wb') as buff:
        pickle.dump({'traces': traces}, buff)

def lohhla_clone_model(sample_ids,
                       tree_edges,
                       clonal_prevalence_mat,
                       cellularity,
                       ploidy_values,
                       tumour_sample_reads,
                       normal_sample_reads,
                       integercpn_info,
                       all_genotypes,
                       transition_inputs,
                       stayrate_alpha=0.9,
                       stayrate_beta=0.1,
                       sd=0.5,
                       nb_alpha=0.5,
                       iter_count=20000,
                       tune_iters=20000,
                       anchor_type='nb',
                       anchor_mode='snvcn',
                       nchains = 2,
                       njobs = 2):
    '''
    stayrate_alpha: Beta prior alpha-parameter on stayrate in clone tree Markov chain
    stayrate_beta: Beta prior beta-parameter on stayrate in clone tree Markov chain
    all_genotypes: Dataframe of genotypes, 0-indexed
    '''
    num_nodes = clonal_prevalence_mat.shape[1]
    
    valid_transitions = transition_inputs['valid_transitions']
    num_transitions = transition_inputs['num_transitions']
    num_genotypes = transition_inputs['num_genotypes']
    cn_genotype_matrix = transition_inputs['cn_genotype_matrix']

    ## Beta-binomial dispersion (higher = less dispersed)
    dispersion = 200.
    
    ## Tree edges
    edges = tree_edges.as_matrix().astype(int) - 1

    with pm.Model() as model:
        BoundedNormal = pm.Bound(pm.Normal, lower = 0., upper = 1.)
        stay_rate = BoundedNormal('stayrate', mu = 0.75, sd = 0.4)

        P = np.zeros(shape=(num_genotypes, num_genotypes))
        P = P + tt.eye(num_genotypes) * stay_rate

        fill_values = tt.as_tensor((1. - stay_rate) / num_transitions)
        fill_values = tt.set_subtensor(fill_values[0], 0)

        P = P + valid_transitions * fill_values[:, np.newaxis]
        P = tt.set_subtensor(P[0, 0], 1.)

        A = tt.dmatrix('A')

        PA = tt.ones(shape=(num_genotypes)) / num_genotypes

        states = CloneTreeGenotypes(
            'genotypes',
            PA=PA,
            P=P,
            edges=edges,
            k=num_genotypes,
            shape=(num_nodes))

        total_cns = theano.shared(np.array(all_genotypes['total_cn'].values))
        alt_cns = theano.shared(np.array(all_genotypes['alt_cn'].values))

        total_cn = pm.Deterministic('total_cn', total_cns[states])
        alt_cn = pm.Deterministic('alt_cn', alt_cns[states])

        sample_alt_copies = tt.dot(clonal_prevalence_mat, alt_cn
                                   ) * cellularity + (1. - cellularity) * 1.

        vafs = sample_alt_copies / (
            tt.dot(clonal_prevalence_mat, total_cn) * cellularity +
            (1. - cellularity) * 2.)
        pm.Deterministic('vafs', vafs)

        alphas = vafs * dispersion
        betas = (1 - vafs) * dispersion

        ## Copy number of tumour cells (aggregated over clones, but not including normal contamination)
        tutotalcn = pm.Deterministic('tutotalcn',
                                     tt.dot(clonal_prevalence_mat, total_cn))

        ## Can't be vectorized further
        for j in range(len(sample_ids)):
            current_sample = sample_ids[j]
            total_counts = integercpn_info['TumorCov_type1'][current_sample].values + integercpn_info['TumorCov_type2'][current_sample].values
            alt_counts = integercpn_info['TumorCov_type2'][
                current_sample].values
            alpha_sel = alphas[j]
            beta_sel = betas[j]

            ## Draw alternative allele counts for HLA locus for each polymorphic site
            alt_reads = pm.BetaBinomial(
                'x_' + str(j),
                alpha=alpha_sel,
                beta=beta_sel,
                n=total_counts,
                observed=alt_counts)

            mult_factor_mean = (tumour_sample_reads[current_sample]
                                / normal_sample_reads)

            ploidy = ploidy_values[j]
            ploidy_ratio = (tutotalcn[j] * cellularity[j] + (1 - cellularity[j]) * 2) / (cellularity[j] * ploidy + (1 - cellularity[j]) * 2)
            if anchor_mode == 'snvcn':
                mult_factor_computed = pm.Deterministic(
                    'mult_factor_computed_' + str(j), 1. / ploidy_ratio *
                    (integercpn_info['Total_TumorCov'][current_sample].values /
                     integercpn_info['Total_NormalCov'][current_sample].values
                     ))
                nloci = len(integercpn_info['Total_TumorCov'][current_sample].values)
                
                tumour_reads_observed = integercpn_info['Total_TumorCov'][
                    current_sample].values
                normal_reads_observed = integercpn_info['Total_NormalCov'][
                    current_sample].values
            elif anchor_mode == 'binmedian':
                binvar_tumour = 'combinedBinTumor'
                binvar_normal = 'combinedBinNormal'
                ## All within a bin are the same, so this is OK
                duplicated_entries = integercpn_info['binNum'][
                    current_sample].duplicated(keep='first')
                nloci = len(integercpn_info[binvar_tumour][current_sample][
                    ~duplicated_entries].values)

                mult_factor_computed = pm.Deterministic(
                    'mult_factor_computed_' + str(j),
                    (1. / ploidy_ratio *
                     (integercpn_info[binvar_tumour][current_sample]
                      [~duplicated_entries].values /
                      integercpn_info[binvar_normal][current_sample]
                      [~duplicated_entries].values)))
                
                tumour_reads_observed = integercpn_info[binvar_tumour][
                    current_sample][~duplicated_entries].values
                normal_reads_observed = integercpn_info[binvar_normal][
                    current_sample][~duplicated_entries].values
            else:
                raise Exception("Invalid option specified.")

            ## Draw ploidy-corrected tumour/normal locus coverage ratio for each polymorphic site

            if anchor_type == 'mult_factor':
                mult_factor = pm.Lognormal(
                    'mult_factor_' + str(j),
                    mu=np.log(mult_factor_mean),
                    sd=sd,
                    observed=mult_factor_computed,
                    shape=(nloci))
            elif anchor_type == 'nb':
                tc_nc_ratio = pm.Deterministic(
                    'tc_nc_ratio_' + str(j), (tutotalcn[j] * cellularity[j] +
                                              (1 - cellularity[j]) * 2) /
                    (ploidy * cellularity[j] + (1 - cellularity[j]) * 2))

                tumoursamplecn = pm.Deterministic('tumoursamplecn_' + str(j),
                                                  (tutotalcn[j] * cellularity[j] +
                                                   (1 - cellularity[j]) * 2))

                tumour_reads_mean = pm.Deterministic(
                    'tumour_reads_mean_' + str(j),
                    tc_nc_ratio * mult_factor_mean * normal_reads_observed)

                tumour_reads = pm.NegativeBinomial(
                    'tumour_reads_' + str(j),
                    mu=tumour_reads_mean,
                    alpha=nb_alpha,
                    observed=tumour_reads_observed)
            else:
                raise Exception('Must specify a valid model type.')

        pm.Deterministic('log_prob', model.logpt)

        step1 = pm.CategoricalGibbsMetropolis(vars=[states])
        step2 = pm.Metropolis(vars=[stay_rate])

        trace = pm.sample(iter_count, tune=tune_iters, step=[step1, step2], njobs = njobs, chains = nchains)

        return trace

def main(args):
    if (args.anchor_mode == 'binmedian' and args.anchor_model_type == 'nb'):
        warnings.warn('Please use SNV-level estimates for anchor_mode', DeprecationWarning)
    
    ## Read inputs
    clonal_prevalences = pd.read_csv(args.clonal_prevalence_file, sep = '\t')
    read_stats = pd.read_csv(args.flagstat_file, sep = '\t')
    hlaloss_table = pd.read_csv(args.hlaloss_file, sep = '\t')
    integercpn_table = pd.read_csv(args.integercpn_file, sep = '\t')
    tree_edges = pd.read_csv(args.tree_edges_file, sep = '\t')
    ploidy_cellularity = pd.read_csv(args.ploidy_cellularity_file, sep = '\t')
    
    sample_sets = [clonal_prevalences['sample'], read_stats['sample'], 
                   hlaloss_table['region'], integercpn_table['region']]
    
    ## Tumour sample IDs to consider
    common_tumour_samples = list(set(sample_sets[0]).intersection(*sample_sets))
    common_tumour_samples.sort()
    
    ## Clonal prevalence matrix
    clonal_prevalence_matrix = clonal_prevalences.pivot_table(index = 'sample', columns = ['node'])
    clonal_prevalence_matrix = clonal_prevalence_matrix.loc[common_tumour_samples,:].as_matrix()
    
    ## List of HLA loci
    hla_loci = np.unique(integercpn_table['HLA_gene'])
    
    traces = {}
    
    if not os.path.isdir(args.outdir):
        os.makedirs(args.outdir)
    
    for hla_locus in hla_loci:
        print("Running HLA locus " + hla_locus)
        
        ## Output directory
        locus_outdir = os.path.join(args.outdir, hla_locus)
        if not os.path.isdir(locus_outdir):
            os.makedirs(locus_outdir)
        
        ## Prepare data for clonal HLA LOH
        hlaloss_subset = hlaloss_table[hlaloss_table['HLA_A_type1'].
                                       str.match(hla_locus)]
        integercpn_subset = integercpn_table[integercpn_table['HLA_gene'] ==
                                             hla_locus]
        
        integercpn_subset.loc[:,
                                'Total_NormalCov'] = integercpn_subset['NormalCov_type1'] + integercpn_subset['NormalCov_type2']
        integercpn_subset.loc[:,
                                'Total_TumorCov'] = integercpn_subset['TumorCov_type1'] + integercpn_subset['TumorCov_type2']

        
        ## Plot CN data
        allele_coverage_outfname = os.path.join(locus_outdir,
                                                "allele_coverage.png")
        locus_coverage_outfname = os.path.join(locus_outdir, "locus_coverage.png")

        plot_cpn(
            integercpn_subset,
            variables=['TumorCov_type1', 'TumorCov_type2'],
            outfname=allele_coverage_outfname)

        plot_cpn(
            integercpn_subset,
            variables=['Total_TumorCov', 'Total_NormalCov'],
            outfname=locus_coverage_outfname)
        
        
        
        ## Genotype states
        max_sample_cn = np.max(hlaloss_subset['HLA_type2copyNum_withBAFBin'] + 
                               hlaloss_subset['HLA_type1copyNum_withBAFBin'])
        max_sample_cn = int(np.min([np.max([np.ceil(max_sample_cn), 4.]), 5.]))
        all_genotypes = list_all_genotypes(max_cn=max_sample_cn, allow_zero=True)
        transition_inputs = ctg_transition_mask(all_genotypes)
        
        ## Run clonal HLA LOH model
        results = lohhla_clone_model(common_tumour_samples,
                                     tree_edges,
                                     clonal_prevalence_mat=clonal_prevalence_matrix,
                                     cellularity=ploidy_cellularity.set_index('sample')['Cellularity'][common_tumour_samples],
                                     ploidy_values=ploidy_cellularity.set_index('sample')['psi'][common_tumour_samples],
                                     tumour_sample_reads=read_stats.set_index('sample')['n_unique_reads'][common_tumour_samples],
                                     normal_sample_reads=read_stats.set_index('sample')['n_unique_reads']['normal'],
                                     integercpn_info=integercpn_subset.set_index('region'),
                                     all_genotypes=all_genotypes,
                                     transition_inputs=transition_inputs,
                                     stayrate_alpha=0.9,
                                     stayrate_beta=0.1,
                                     sd=0.5,
                                     nb_alpha=0.5,
                                     iter_count=20000,
                                     tune_iters=20000,
                                     anchor_type=args.anchor_model_type,
                                     anchor_mode=args.anchor_mode)
        
        traces.update({hla_locus: results})
        
        ## Plot trace
        traceplot_outfname = os.path.join(locus_outdir, "traceplot.png")
        tp = pm.traceplot(results)
        fig, axarr = plt.subplots(tp.shape[0], tp.shape[1], figsize=(20, 20))
        tp = pm.traceplot(results, ax=axarr)
        fig.savefig(traceplot_outfname)
        plt.clf()
        
    print("Saving traces.")
    
    trace_outfname = os.path.join(args.outdir, "traces.pkl")
    write_traces(traces, trace_outfname)

    print("Completed.")


if __name__ == '__main__':
    args = process_args()
    
    main(args)
    
    