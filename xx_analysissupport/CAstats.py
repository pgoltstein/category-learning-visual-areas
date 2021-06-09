#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 4, 2019

@author: pgoltstein
"""

########################################################################
### Imports
########################################################################

import numpy as np
import scipy.stats as scistats
import statsmodels.api as statsmodels

def mean_sem( datamat, axis=0 ):
    mean = np.nanmean(datamat,axis=axis)
    n = np.sum( ~np.isnan( datamat ), axis=axis )
    sem = np.nanstd( datamat, axis=axis ) / np.sqrt( n )
    return mean,sem,n

def report_mean(sample1, sample2):
    print("  Group 1, Mean (SEM) = {} ({}) n={}".format(*mean_sem(sample1.ravel())))
    print("  Group 2, Mean (SEM) = {} ({}) n={}".format(*mean_sem(sample2.ravel())))

########################################################################
### Functions for reporting statistical tests
########################################################################

def report_chisquare_test( sample1, sample2, n_indents=2, alpha=0.05, bonferroni=1  ):
    p,t,df,n = chisquared_test( sample1, sample2 )
    p_b = p*bonferroni
    print('{}one-sided Chisquare test, X^2({:0.0f},N={:0.0f})={:0.3f}, p={}{}'.format( " "*n_indents, df, n, t, p, "  >> sig." if p<(alpha/bonferroni) else "." ))
    return p_b

def report_wmpsr_test( sample1, sample2, n_indents=2, alpha=0.05, alternative="two-sided", bonferroni=1, preceding_text=""):
    p,Z,n = wilcoxon_matched_pairs_signed_rank_test( sample1, sample2, alternative=alternative )
    if alternative=="two-sided":
        preceding_text += "two-sided "
    else:
        preceding_text="one-sided "
    if bonferroni>1:
        p_b = p*bonferroni
        if p_b < 0.001:
            print('{}{}WMPSR test, W={:0.0f}, p_bonf={:4E}, n={:0.0f}{}'.format( " "*n_indents, preceding_text, Z, p_b, n, "  >> sig." if p<(alpha/bonferroni) else "." ))
        else:
            print('{}{}WMPSR test, W={:0.0f}, p_bonf={:0.4f}, n={:0.0f}{}'.format( " "*n_indents, preceding_text, Z, p_b, n, "  >> sig." if p<(alpha/bonferroni) else "." ))
        return p_b
    else:
        if p < 0.001:
            print('{}{}WMPSR test, W={:0.0f}, p={:4E}, n={:0.0f}{}'.format( " "*n_indents, preceding_text, Z, p, n, "  >> sig." if p<(alpha/bonferroni) else "." ))
        else:
            print('{}{}WMPSR test, W={:0.0f}, p={:0.4f}, n={:0.0f}{}'.format( " "*n_indents, preceding_text, Z, p, n, "  >> sig." if p<(alpha/bonferroni) else "." ))
        return p

def report_mannwhitneyu_test( sample1, sample2, n_indents=2, alpha=0.05, bonferroni=1 ):
    p,U,r,n1,n2 = mann_whitney_u_test( sample1, sample2 )
    if bonferroni>1:
        p_b = p*bonferroni
        if p_b < 0.001:
            print('{}two-sided Mann-Whitney U test, U={:0.0f}, p_bonf={:4E}, r={:0.3f}, n1={:0.0f}, n2={:0.0f}{}'.format( " "*n_indents, U, p_b, r, n1, n2, "  >> sig." if p<(alpha/bonferroni) else "." ))
        else:
            print('{}two-sided Mann-Whitney U test, U={:0.0f}, p_bonf={:0.4f}, r={:0.3f}, n1={:0.0f}, n2={:0.0f}{}'.format( " "*n_indents, U, p_b, r, n1, n2, "  >> sig." if p<(alpha/bonferroni) else "." ))
        return p_b
    else:
        if p < 0.001:
            print('{}two-sided Mann-Whitney U test, U={:0.0f}, p={:4E}, r={:0.3f}, n1={:0.0f}, n2={:0.0f}{}'.format( " "*n_indents, U, p, r, n1, n2, "  >> sig." if p<(alpha/bonferroni) else "." ))
        else:
            print('{}two-sided Mann-Whitney U test, U={:0.0f}, p={:0.4f}, r={:0.3f}, n1={:0.0f}, n2={:0.0f}{}'.format( " "*n_indents, U, p, r, n1, n2, "  >> sig." if p<(alpha/bonferroni) else "." ))
        return p

def report_kruskalwallis( samplelist, n_indents=2, alpha=0.05 ):
    p,H,DFbetween,DFwithin,n = kruskalwallis( samplelist )
    if p < 0.001:
        print("{}two-sided Kruskal-Wallis test, H({:0.0f}) = {:0.3f}, p = {:4E}, n={:0.0f}".format( " "*n_indents, DFbetween, H, p, n, "  >> sig." if p<alpha else "." ))
    else:
        print("{}two-sided Kruskal-Wallis test, H({:0.0f}) = {:0.3f}, p = {:0.4f}, n={:0.0f}".format( " "*n_indents, DFbetween, H, p, n, "  >> sig." if p<alpha else "." ))


########################################################################
### Functions for performing statistical tests
########################################################################
def chisquared_test( sample1, sample2 ):
    if len(np.unique(sample1)) > 2 or len(np.unique(sample2)) > 2:
        print("Only two samples, and boolean data allowed for chi square test")
        return 1.0,np.NaN,np.NaN,0
    sample1 = sample1[~np.isnan(sample1)].ravel()
    sample2 = sample2[~np.isnan(sample2)].ravel()
    n_categories = 2
    n_groups = 2
    df = (n_categories-1)*(n_groups-1)
    n1 = len(sample1)
    n2 = len(sample2)
    n = n1 + n2
    frequency_samples = np.array([np.sum(sample1==1),np.sum(sample2==1)])
    n_samples = np.array([n1,n2])
    chisq,p,(f_real,f_expected) = statsmodels.stats.proportions_chisquare( count=frequency_samples, nobs=n_samples )
    return p,chisq,df,n

def wilcoxon_matched_pairs_signed_rank_test( sample1, sample2, alternative="two-sided" ):
    sample1 = sample1[~np.isnan(sample1)].ravel()
    sample2 = sample2[~np.isnan(sample2)].ravel()
    if np.count_nonzero(sample1)==0 and np.count_nonzero(sample2)==0:
        return 1.0,np.NaN,np.NaN
    else:
        Z,p = scistats.wilcoxon(sample1, sample2, alternative=alternative)
        n = len(sample1)
        return p,Z,n

def mann_whitney_u_test( sample1, sample2 ):
    sample1 = sample1[~np.isnan(sample1)].ravel()
    sample2 = sample2[~np.isnan(sample2)].ravel()
    U,p = scistats.mannwhitneyu(sample1, sample2)
    n1 = len(sample1)
    n2 = len(sample2)
    r = U / np.sqrt(n1+n2)
    return p,U,r,n1,n2

def kruskalwallis( samplelist ):
    # Clean up sample list and calculate N
    N = 0
    no_nan_samplelist = []
    for b in range(len(samplelist)):
        no_nan_samples = samplelist[b][~np.isnan(samplelist[b])]
        if len(no_nan_samples) > 0:
            no_nan_samplelist.append(no_nan_samples)
            N += len(no_nan_samples)

    # Calculate degrees of freedom
    k = len(samplelist)
    DFbetween = k - 1
    DFwithin = N - k
    DFtotal = N - 1
    H,p = scistats.kruskal( *no_nan_samplelist )
    return p,H,DFbetween,DFwithin,N
