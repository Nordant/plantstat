"""
Created on Thu Apr 5 2021
@author: Nordant
"""

# Necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from scipy.stats import *
import statsmodels.api as sm

class Variable_Analyzer:
    '''
    Main class with all necessary functions.
    Takes an array of variables and a list of classes.
    '''
    def __init__(self, array, labels):
        self.array = array
        self.labels = labels
    
    def var_len(self):
        '''
        Function for calculating all variables' lengths.
        '''
        lengths = []
        for array in self.array:
            lengths.append(len(array))
        return lengths
    
    def stat(self, func):
        '''
        Function for calculation basic statistics.
        As input takes any statistic function from numpy
        or scipy packages.
        Output - a list of values (one variable - one value).
        '''
        result = []
        for array in self.array:
            result.append(func(array))
        return result
    
    def outliers(self, central_measure = np.median, iqr_n = 1.5):
        '''
        Function for finding outliers.
        As input takes central measure parameter (default median)
        and iqr_n value (default 1.5).
        Output - a dictionary of values (one variable - all outliers).
        '''
        out = dict()
        for array, label in zip(self.array, self.labels):
            upper = np.array(array)[central_measure(array) + 
                                    iqr_n * iqr(array) < array]
            lower = np.array(array)[central_measure(array) - 
                                    iqr_n * iqr(array) > array]
            out[label] = np.concatenate([lower, upper])
        return out
    
    def boxplot(self):
        '''
        Function for boxplot visualization.
        Output - boxplots of all variables.
        '''
        for idx, array in enumerate(self.array):
            fig, ax = plt.subplots(figsize = (4, 4), dpi = 100)
            plt.title('Boxplot ({})'.format(self.labels[idx]))
            sns.boxplot(y = array, color = 'gray', linewidth = 1.5)
            sns.swarmplot(y = array, color = 'blue', edgecolor = 'black', 
                          alpha = .9)
            plt.show()
        
    def basic_stats(self):
        '''
        Function for various basic statistics.
        Output - a data frame with mean, trim_mean, median, std, min and max,
        iqr values, and p-values of shapiro test and normaltest.
        '''
        stat_df = pd.DataFrame()
        for idx, array in enumerate(self.array):
            stat_df.loc[idx, 'variable'] = self.labels[idx]
            stat_df.loc[idx, 'mean'] = round(np.mean(array), 3)
            stat_df.loc[idx, 'trim_mean'] = round(trim_mean(array, 0.1), 3)
            stat_df.loc[idx, 'median'] = round(np.median(array), 3)
            stat_df.loc[idx, 'std'] = round(np.std(array), 3)
            stat_df.loc[idx, 'min'] = min(array)
            stat_df.loc[idx, 'max'] = max(array)
            stat_df.loc[idx, 'iqr'] = round(iqr(array), 3)
            
            # The Shapiro-Wilk test tests the null hypothesis that the data 
            # was drawn from a normal distribution.
            # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.shapiro.html#scipy.stats.shapiro
            stat_df.loc[idx, 'shapiro_p'] = round(shapiro(array)[1], 3)
            
            # This function tests the null hypothesis that a sample comes 
            # from a normal distribution.
            # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.normaltest.html#scipy.stats.normaltest
            stat_df.loc[idx, 'normtest_p'] = round(normaltest(array)[1], 3)
        return stat_df
        
    def var_compare(self):
        '''
        Function for variables comparisons.
        Takes all variables and finds statistical tests for all pairs.
        Output - a data frame with p-values of: bartlet test,
        T-test, Wilcoxon test, Kruskal-Wallis H-test, Mann-Whitney rank test.
        '''
        comp_df = pd.DataFrame()
        pairs = list(combinations(range(len(self.labels)), 2))
        
        for idx in range(len(pairs)):
            first = self.array[pairs[idx][0]]
            second = self.array[pairs[idx][1]]
            
            comp_df.loc[idx, 'pair'] = self.labels[pairs[idx][0]] + ' ~ ' + self.labels[pairs[idx][1]]
            comp_df.loc[idx, 'shape'] = str(len(first)) + ' ~ ' + str(len(second))
            
            # Bartlett’s test tests the null hypothesis 
            # that all input samples are from populations with equal variances.
            # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bartlett.html#scipy.stats.bartlett
            comp_df.loc[idx, 'bartlet_p'] = round(bartlett(first, second)[1], 3)
            
            # Calculate the T-test for the means of two independent samples.
            # The null hyp: 2 independent samples have identical average.
            # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html#scipy.stats.ttest_ind
            comp_df.loc[idx, 'ttest_p'] = round(ttest_ind(first, second)[1], 3)
            
            # The Wilcoxon signed-rank test tests the null hypothesis that 
            # two related paired samples come from the same distribution. 
            # It is a non-parametric version of the paired T-test.
            # 
            if len(first) == len(second):
                comp_df.loc[idx, 'wilcoxon_p'] = round(wilcoxon(first, second)[1], 3)
            else:
                comp_df.loc[idx, 'wilcoxon_p'] = np.NaN
            
            # the Kruskal-Wallis H-test for independent samples.
            # the null hypothesis that the population median of all 
            # of the groups are equal. 
            # It is a non-parametric version of ANOVA.
            # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kruskal.html#scipy.stats.kruskal
            comp_df.loc[idx, 'kruskal_p'] = round(kruskal(first, second)[1], 3)
            
            # the Mann-Whitney rank test on samples x and y
            # Use only when the number of observation in each sample is > 20 
            # and you have 2 independent samples of ranks.
            # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html#scipy.stats.mannwhitneyu
            if (len(first) > 20) and (len(second) > 20):
                comp_df.loc[idx, 'mannwhitneyu_p'] = round(mannwhitneyu(first, second)[1], 3)
            else:
                comp_df.loc[idx, 'mannwhitneyu_p'] = np.NaN
            
        return comp_df
    
    def get_pairs(self, indices = False):
        '''
        Return a list of all pairs of variables.
        Can return indices or labels' names.
        '''
        if indices == False:
            p = list(combinations(self.labels, 2))
        else:
            p = list(combinations(range(len(self.labels)), 2))
        return p
            
    def corrs(self, method = 'pearson', heatmap = False):
        '''
        Function for creating a corr matrix.
        Takes all variables and calculates correlations.
        Output - a data frame with corr coefficients.
        Also returns a corr heatmap if 'heatmap = True'.
        '''
        df = pd.DataFrame.from_records(self.array).transpose()
        df.columns = self.labels
        corrs = df.corr(method = method)
        
        if heatmap == True:
            plt.figure(figsize = (10, 6))
            heatmap = sns.heatmap(corrs, vmin = -1, vmax = 1, annot = True,
                                  cmap = "coolwarm")
            heatmap.set_title('Correlation Heatmap ({})'.format(method));
            plt.show()
        else:
            pass
        
        return corrs
    
    def QQplot(self):
        '''
        Function for Q-Q plot visualization.
        Returns Q-Q plots for each variable.
        '''
        for idx, array in enumerate(self.array):
            fig, ax = plt.subplots(figsize = (7, 5))
            plt.title('Q-Q plot ({})'.format(self.labels[idx]))
            sm.qqplot(np.array(array), line = '45', fit = True, ax = ax)
            plt.show()
    
    def pair_plot(self):
        '''
        Function for pairplot visualization.
        Returns pairplots for all variables.
        '''
        df = pd.DataFrame.from_records(self.array).transpose()
        df.columns = self.labels
        sns.pairplot(df, corner = True)
        plt.show()

