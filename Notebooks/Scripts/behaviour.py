"""
# -*- coding: utf-8 -*-
Created on Thu May  5 22:10:56 2022
@author: Brandon
"""
import scipy.io as sio
import scipy
import numpy as np
import pandas as pd
import psignifit as ps
import matplotlib.pyplot as plt
import os
import itertools
from matplotlib import cm
from collections import defaultdict
import random
import warnings


def combineBehaviour(path, all_key, subj_key, polarity_key, exp_key):
    """ Generic function to combine dataframes from the Free Choice experiment based on subject key, polarity (Anodal or Cathodal stim)
        and/or position in the experiment (pre/stim/post)

    Args:
        path (str): path where .csv files generated from eegloading.combineSubjects are stored 
        subj_key (str): string with initials for corresponding participant. Pass '' to get data from all participants
        polarity_key (str): string with key for polarity of the stimulation ran that day. 'AN' for anodal. 'CA' for cathodal. Pass '' to get data from all stim types
        exp_key (_type_): string with key for the condition. 'PR' for pre-stim. 'ST' for stim. 'PO' for post-stim. Pass '' to get data from all stim conditions

    Returns:
        df: dataframe with trials organized as shown below in the empty dataframe, concatenated across all .csv files with matching input keys in filename
    """
    directory = os.listdir(path)
    df =pd.DataFrame(
        {
            'Block Number': [],
            'Trial Number': [],
            'Good/Bad Trial': [] ,
            'Reaction Time: First Target': [] ,
            'Choice': [],
            'TOA': [],
            'First Target Onset': [],
            'Second Target Onset': [],
            'Relative RT': [],
            'Rep Number': [],
            'Alt Number': [],
            'Saccade Endpoint': []

        }
    )
    if not all_key:
        for fname in directory:
            if os.path.isfile(path + os.sep + fname):
                if subj_key in fname:
                    if polarity_key in fname:
                        if exp_key in fname:
                            df = pd.concat([df, pd.read_csv(path + '/' + fname)])
    else:
        for fname in directory:
            if os.path.isfile(path + os.sep + fname):
                df = pd.concat([df, pd.read_csv(path + '/' + fname)])
    return df

def groupDataByCondition(cond1, cond2, path):
    """Returns two dataframes matching each condition passed as an argument

    Args:
        cond1 (str): condition to pass (ex PR)
        cond2 (str): "" ""
        path (str): path where behavioural files are stored

    Returns:
        df1, df2: dataframes matching cond1 and cond2
    """
    directory = os.listdir(path)
    dates = list(set([e[10:] for e in directory]))
    df1 = 0
    df2 = 0
    for date in dates:
        for file in directory:
            if date in file:
                if cond1 in file:
                    ## Import .csv, append to current df if it exists
                    if isinstance(df1, pd.DataFrame):
                        df1 = pd.concat([df1, pd.read_csv(path + "/" + file, index_col = False)])
                    else:
                        df1 = pd.read_csv(path + "/" + file, index_col = False)
                elif cond2 in file:
                    if isinstance(df2, pd.DataFrame):
                    ## Append to current df if it exists
                        df2 = pd.concat([df2, pd.read_csv(path + "/" + file, index_col = False)])
                    else:
                        df2 = pd.read_csv(path + "/" + file, index_col = False)

    return df1, df2

def removeNaN(array):
    return array[~np.isnan(array)]

def computePsycParam(fit, param):
    """Computes mean and variance of parameter estimate for psychometric fit given param

    Args:
        fit (ps): psignifit object
        param (str): 'thresh' for threshold, 'width' for width

    Returns:
        mean_param: average from bayesian fitting procedure
        var_param = variance from bayesian fitting procedure
    """
    warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 

    # Set boundaries to compute parameter densities over
    if param == 'thresh':
        low_bound = -.05
        high_bound = .05
    else:
        low_bound = 0
        high_bound = .3
    
    # Set constant param space to compute over
    x_bounded = np.arange(low_bound, high_bound, 0.001)

    x_all, marg_all, CI = getMarginalPsycParams(fit, param=param)
    x_all = np.insert(x_all, [0, -1], [-.1, .1])  # Add 0 bound on to beginning and end just for plotting
    marg_all = np.insert(marg_all, [0, -1], [0, 0])

    # Sample from model and compute variance
    f = scipy.interpolate.interp1d(x_all, marg_all)  # interpolate over constant param space
    y_bounded = f(x_bounded)
    y_normalize = [float(i) / sum(y_bounded) for i in y_bounded]
    mean_param = np.mean(random.choices(x_bounded, weights=y_normalize, k=10000))
    var_param = np.std(random.choices(x_bounded, weights=y_normalize, k=10000))

    return mean_param, var_param

def getMarginalPsycParams(fit, param):
    """ Returns the marginal density for a given parameter of the fitted psychometric function object

    Args:
        fit (dict): fitted psychometric function dictionary 
        param (str): 'thresh' for the threshold/midpoint of the psychometric function
                     'width' for the width/slope of the psychometric function

    Returns:
        x: x-axis over which marginal density is computed
        marginal: density 
    """
    if param == 'thresh': dim = 0
    elif param == 'width': dim = 1
    x = fit['marginalsX'][dim]
    marginal = fit['marginals'][dim]
    CI = fit['conf_Intervals'][dim]

    return x, marginal, CI

def loadSingleCSV(path, filename):
    """ Quick function to load a single behavioural csv file
    """
    return pd.read_csv(path + '/' + filename)

def _check_keys(dict):
    """checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    """
    for key in dict:
        if isinstance(dict[key], sio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

def _todict(matobj):
    """
    A recursive function which constructs from matobjects nested dictionaries
    """
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, sio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict

def separate_paths(p, s = '', c = None):
   d = defaultdict(list)
   for a, *b in p:
      d[a].extend(b if not b else [b])
   if c is None or len(d) == 1:
      yield from [j for a, b in d.items() for j in separate_paths(b, s=s+a+'_', c=c if c is not None else len(b))]
   else:
      yield from [s]*c

def loadmat(filename):
    """
    this function should be called instead of direct scipy.io .loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    """
    data = sio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

def ecdf(x):
    xs = np.sort(x)
    ys = np.arange(1, len(xs)+1)/float(len(xs))
    return xs, ys

def makeDataFrame(data):
    df = pd.DataFrame(
        {
            'Block Number': data[:,1],
            'Trial Number': data[:,2],
            'Subject Number': data[:,45],
            'Good/Bad Trial': data[:,3],
            'Reaction Time: First Target': data[:,4],
            'Reaction Time: Chosen Target': data[:,44],
            'Choice': data[:,8],
            'TOA': data[:,25],
            'First Target Onset': data[:,28],
            'Second Target Onset': data[:,29],
            'Relative RT': data[:,4] + data[:,28],
            'Saccade Amplitude': data[:,34],
            'Win-Stay/Lose-Switch': data[:,38],
            'Previous Win-Lose': data[:,39],
            'Stay-Switch': data[:,40],           
        }
    )
    return df
       
def getChoiceSequence(df):
    df['Rep Number'] = np.zeros(len(df))
    df['Alt Number'] = np.zeros(len(df))
    
    for i in range(len(df)):
        if df['Good/Bad Trial'][i] == 0:
            if df['Trial Number'][i] == df['Trial Number'].shift(periods=1)[i] + 1:
                if df['Choice'][i] == df['Choice'].shift(periods=1)[i]:
                    df['Rep Number'][i] = df['Rep Number'][i-1] + 1
            if df['Trial Number'][i] == df['Trial Number'].shift(periods=1)[i] + 1:
                if df['Choice'][i] != df['Choice'].shift(periods=1)[i]:
                    df['Alt Number'][i] = df['Alt Number'][i-1] + 1
    return df
                    
def getPsycArray(df):
    psycArray = np.ndarray(shape=(9,3))
    psycArray[:,0] = [-.1, -.067, -.033, -.017, 0, .017, .033, .067, .1] # TOAs used in experiment
    
    for i in range(len(psycArray)):
        psycArray[i,1] = np.size(df.loc[df['TOA'] == psycArray[i,0]]['Choice']) - np.sum(df.loc[df['TOA'] == psycArray[i,0]]['Choice'])
        psycArray[i,2] = np.size(df.loc[df['TOA'] == psycArray[i,0]]['Choice'])
    return psycArray

def getPsycFunction(df):
    options = dict()
    options = {
            'sigmoidName': 'norm',
            'expType': 'YesNo',
            'fixedPars': np.array([np.nan, np.nan, .01, .01, np.nan]) 
    }
    psycArray = getPsycArray(df)
    fit = ps.psignifit(psycArray, options)
    ps.psigniplot.plotPsych(fit, showImediate = False)
    return fit

def plotPsycHistory(df, nback, condition):
    options = dict()
    options = {
            'sigmoidName': 'norm',
            'expType': 'YesNo',
            'fixedPars': np.array([np.nan, np.nan, .01, .01, np.nan]) 
    }
    
    for i in range(nback):
        psychArray = getPsycArray(df.loc[df[condition]== i])
        fit = ps.psignifit(psychArray, options)
        if i == nback:
            ps.psigniplot.plotPsych(fit, showImediate = True)
            # ps.psigniplot.plotMarginal(fit, dim = 0)
        else:
            ps.psigniplot.plotPsych(fit, showImediate = False)
            # ps.psigniplot.plotMarginal(fit, dim = 0)
    
    plt.show()
    
def plotPsycDelays(df, numBins):
    delayedData = binDelaydata(df, 750, 1100, numBins)
    delays= np.linspace(750, 1100, numBins)

    for i in range(numBins):
        getPsycFunction(delayedData[str(delays[i])])

def plotRTHistory(df, nback, condition):
    plt.figure()
    for i in range(0, nback):
        xs, ys = ecdf(df.loc[df[condition]==i]['Relative RT'])
        plt.plot(xs, ys, label= "%s back" % i)
        
    plt.title(label= "Relative RT %s" % condition)
    plt.savefig(rf"C:\Users\Brandon\Desktop\PhD\Baseline Dynamics\Baseline-Dynamics\Figures\cdf_rt_{condition}_relative", format='svg')

    plt.figure()
    for i in range(0, nback):
        xs, ys = ecdf(df.loc[df[condition]==i]['Reaction Time: First Target'])
        plt.plot(xs, ys, label= "%s back" %i)
        
    plt.title(label= "Stim-locked RT %s" % condition)    
    plt.xlim([-250,400])
    plt.savefig(rf"C:\Users\Brandon\Desktop\PhD\Baseline Dynamics\Baseline-Dynamics\Figures\cdf_rt_{condition}", format='svg')

    plt.show()
    
def plotPsychometricFunction(df, axisHandle, plot, showImediate):
    """I didn't spell it

    Args:
        df (_type_): behavioural dataframe to pass through getPsycArray
        axisHandle (_type_): figure to plot on
        showImediate (_type_): True if last sigmoid generated per graph
    """
    options = dict()
    options = {
            'sigmoidName': 'norm',
            'expType': 'YesNo',
            'fixedPars': np.array([np.nan, np.nan, .01, .01, np.nan]),
    }
    psycArray = getPsycArray(df)
    fit = ps.psignifit(psycArray, options)
    if plot == True:
        ps.psigniplot.plotPsych(fit, showImediate = showImediate, axisHandle = axisHandle)
    return fit

def plotRTDelays(data, minDelay, maxDelay, numBins):
    """ Plot pdfs and cdfs of reaction times from Free Choice Saccade Experiment aligned to stimulus onset (t = 0) and delay time onset.

    Args:
        data (df): free choice dataframe
        minDelay (int): minimum delay time for setting bins
        maxDelay (int): max delay time for setting bins
        numBins (int): number of bins 
    """
    # Return delay-binned data
    delayidx = binDelaydata(data, minDelay = minDelay, maxDelay = maxDelay, numBins = numBins)            
    delays= np.linspace(minDelay, maxDelay, numBins)

    # Set figure properties
    _ , axs = plt.subplots(2, 2, figsize=(16,9))
    cmap = cm.get_cmap('Blues')

    # Plot Reaction Time Distributions with alignment
    for i in range(numBins):
        axs[0,0].hist(delayidx[str(delays[i])]['Reaction Time: First Target'] + delayidx[str(delays[i])]['First Target Onset'],
                       density = True, bins = 50, color = cmap(i/5))     
        axs[0,0].set_xlim([750, 1750])
        axs[0,0].set_xlabel('Reaction Time (ms)')
        axs[0,0].set_ylabel('Count')
    
    # Plot cumulative distributions with alignment
        xs, ys = ecdf(delayidx[str(delays[i])]['Reaction Time: First Target'] + delays[i])
        axs[1,0].plot(xs, ys, color = cmap(i/5))
        axs[1,0].set_xlabel('Reaction Time (ms)')
        axs[1,0].set_ylabel('Count')

    # Plot Reaction Time Distributions aligned to 0
        axs[0,1].hist(delayidx[str(delays[i])]['Reaction Time: First Target'], bins = 50, color = cmap(i/5), density=True)
        axs[0,1].set_xlim([0, 400])
        axs[0,1].set_xlabel('Reaction Time (ms)')
        axs[0,1].set_ylabel('Count')

    # Plot Reaction Time Distributions aligned to 0
        xs, ys = ecdf(delayidx[str(delays[i])]['Reaction Time: First Target'])
        axs[1,1].plot(xs, ys, color = cmap(i/5))
        axs[1,1].set_xlim([0, 400])
        axs[1,1].set_xlabel('Reaction Time (ms)')
        axs[1,1].set_ylabel('Count')
    plt.savefig(rf"C:\Users\Brandon\Desktop\PhD\Baseline Dynamics\Baseline-Dynamics\Figures\rt_delays", format='svg')

def binDelaydata(data, minDelay, maxDelay, numBins):
    """Split up free choice data into a dictionary by equally spaced delay time bins

    Args:
        data (df): free choice dataframe 
        minDelay (int): minimum delay time for setting bins
        maxDelay (int): max delay time for setting bins
        numBins (int): number of bins 

    Returns:
        delayidx: free choice dataframes separated into a dictionary, where the index for each is the delay bin (ex delayidx['750.0'] == all data within delay bin 1 )
    """
    delays= np.linspace(minDelay, maxDelay, numBins)
    delayidx = dict()
    # Get Indices for delay bins
    for i in range(numBins):
        if i == numBins-1:
            delayidx[str(delays[i])] = data[(data['First Target Onset'] >= delays[i])] 
        else:
            delayidx[str(delays[i])] = data[(data['First Target Onset'] >= delays[i]) & (data['First Target Onset'] < delays[i+1])]
    return delayidx
  