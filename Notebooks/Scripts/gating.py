import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib.pyplot import cm
from scipy.stats import entropy
import pandas as pd
np.random.seed(0)

class baselineDynamicIntegrator:
    """
    Class for a sensory integrator with within-trial baseline dynamics. 
    Initializes with trial and dynamics parameters.
    """
    def __init__(self, name, time,
                 xO, theta, kappa,
                 sigma, muR, sR, k, scale, maxT):
        self.name = name
        self.time = time
        self.xO = xO # Initial condition for OU Baseline
        self.theta = theta # mean-reverting parameter
        self.kappa = kappa # rate parameter
        self.sigma = sigma # variance parameter
        self.muR = muR # sensory integration mean rate
        self.sR = sR # sensory integration variance
        self.maxT = maxT # maximum time for each probability distribution
        self.k = k # mixing parameter for anticipatory and sensory-guided probability distributions
        self.scale = scale # scaling parameter for OU to eLATER

    def setDelayTimes(self, delays):
        """ Adds delay times for given trial set to the class (this is kinda stupid, remove function and do it in one line somewhere else.... well I guess this depends on if the delays are simulated or 
        provided from data. Just leave it for now and figure that out as it comes)

        Args:
            delays (ndarray): distribution of delay times to sample from
        """
        self.delayTimes = delays.astype(int)

    def extended_later(self, muR, muS, sR, sigma):
        """ Extended LATER model from Nakahara et al (2006) https://pubmed.ncbi.nlm.nih.gov/16971090/
        Takes class input for fixed time parameters, plus additional arguments for the mean and variance of the baseline and sensory integration

        Args:
            muR (float): mean rate of rise for sensory integration
            sR (float): variance of the rate of rise 

        Returns:
            pT (ndarray): array of 1xt where each point is the probability of a reaction time equalling that time point (sums to 1)
        """
        t = np.arange(1, self.maxT) #Need max time to normalize with other distributions
        a = muR / muS
        b = sR / muS
        c = sigma / sR
        
        e1 = (1/t - a) ** 2
        e2 = t ** 2 / (t ** 2 + c **2)
        e3 = -1 / (2 * b **2)
        pT = np.multiply(np.multiply(t + a * (c ** 2) / (t **2 + c ** 2) **(3/2), 1/((2 * math.pi)**(1/2)*b)), 
                        np.exp(e1 * e2 * e3))
        pT = pT / np.sum(pT)
        return pT
    
    def ou_analytical(self, X0, theta, kappa, sigma):
        """ Analytical solution for mean and variance of an Ornstein Uhlenbeck Process

        Args:
            t (ndarray): time array
            X0 (float): initial conditions for OU process
            theta (float): mean reversion parameter
            kappa (float): rate parameter
            sigma (float): variance parameter

        Returns:
            ouMean (ndarray): 1xt array giving analytical mean of OU process
            ouVar (ndarray): 1xt array giving analytical variance
        """
        t = np.linspace(0.001,2,1999)

        ouMean = theta + np.exp(-kappa*t) * (X0-theta) 
        ouVar = np.sqrt( sigma**2 /(2*kappa) * (1-np.exp(-2*kappa*t)))
        return ouMean, ouVar
    
    def ou_first_passage(self, X0, theta, kappa, sigma):
        """First Passage density solution (github:Cantaro86 for implementation
        & https://arxiv.org/pdf/1810.02390.pdf for original solution)

        Args:
            t (ndarray): time vector
            X0 (float): initial condition
            theta (float): mean-reversion
            kappa (float): rate
            sigma (float): variance
            
        Returns:
            pdf (ndarray): of first hitting time
        """        
        t = np.linspace(0.001,2,1999)

        C = (X0 - theta)*np.sqrt(2*kappa)/sigma     # new starting point
        pdf = np.sqrt(2/np.pi) * np.abs(C)*np.exp(-t) \
                / (1-np.exp(-2*t))**(3/2) * np.exp(-( (C**2)*np.exp(-2*t)) / (2*(1-np.exp(-2*t))) )
        pdf = kappa * pdf
        return pdf/np.sum(pdf) 

    def combine_anticipatory_sensory(pTrialOnset, pAntic):
        """Combines and renormalizes anticipatory first passage solution and sensory-guided responses

        Args:
            pTrialOnset (ndarray): sensory responses probability distribution
            pAntic (ndarray): anticipatory responses probability distribution

        Returns:
            pT (ndarray): renormalized reaction time probability distribution
        """
        pT = np.stack((normalize_rt_pdfs(pTrialOnset), pAntic), axis = 1)
        return pT / sum(pT)

    def BDI(self):
        """ Do The Work
        """
        
        pTrialOnset, pTargetOnset, pAnticipatory = [np.zeros((self.time.size, self.delayTimes.size)) for _ in range(3)] 
        self.ouMean, self.ouVar = self.ou_analytical(self.xO, self.theta, self.kappa, self.sigma)
        pAnticipatory = self.ou_first_passage(self.xO, self.theta, self.kappa, self.sigma)
        pAnticipatory[self.delayTimes[0]+150:] = 0 #Trying cutting off the anticipatory distribution to the last delay time

        if hasattr(self, 'delayTimes') == True:
            for i in range(self.delayTimes.size):
                pTargetOnset[0:self.maxT-1,i] = self.extended_later(self.muR, self.ouMean[self.delayTimes[i]] * self.scale, self.sR, self.sigma * self.scale) #change self.sigma to self.ouVar[self.delayTimes[i]] for changing cov
                pTrialOnset[self.delayTimes[i]:self.delayTimes[i]+self.maxT,i] = pTargetOnset[0:self.maxT,i]
        else:  
            print("No delay distribution found. Please set via setDelayTimes first")    

    
        self.combined = combine_anticipatory_sensory(pTrialOnset, pAnticipatory, k = self.k) 
        self.pTargetOnset = pTargetOnset
        self.pAnticipatory = pAnticipatory
        self.pTrialOnset = pTrialOnset

class eLATER:
    """
    Class for fitting eLATER model only to delay distribution 
    """
    def __init__(self, name, time,
                 muR, sigR, muS, sigS, maxT):
        self.name = name
        self.time = time
        self.muR = muR # sensory integration mean rate
        self.sigR = sigR # sensory integration variance
        self.muS = muS
        self.sigS = sigS
        self.maxT = maxT # maximum time for each probability distribution

    def setDelayTimes(self, delays):
        """ Adds delay times for given trial set to the class (this is kinda stupid, remove function and do it in one line somewhere else.... well I guess this depends on if the delays are simulated or 
        provided from data. Just leave it for now and figure that out as it comes)

        Args:
            delays (ndarray): distribution of delay times to sample from
        """
        self.delayTimes = delays.astype(int)

    def extended_later(self, muR, muS, sR, sigma):
        """ Extended LATER model from Nakahara et al (2006) https://pubmed.ncbi.nlm.nih.gov/16971090/
        
        
        Takes class input for fixed time parameters, plus additional arguments for the mean and variance of the baseline and sensory integration

        Args:
            muR (float): mean rate of rise for sensory integration
            sR (float): variance of the rate of rise 

        Returns:
            pT (ndarray): array of 1xt where each point is the probability of a reaction time equalling that time point (sums to 1)
        """
        t = np.arange(1, self.maxT) #Need max time to normalize with other distributions
        a = muR / muS
        b = sR / muS
        c = sigma / sR
        
        e1 = (1/t - a) ** 2
        e2 = t ** 2 / (t ** 2 + c **2)
        e3 = -1 / (2 * b **2)
        pT = np.multiply(np.multiply(t + a * (c ** 2) / (t **2 + c ** 2) **(3/2), 1/((2 * math.pi)**(1/2)*b)), 
                        np.exp(e1 * e2 * e3))
        pT = pT / np.sum(pT)
        return pT
    
    def computeModel(self):
        pTrialOnset, pTargetOnset = [np.zeros((self.time.size, self.delayTimes.size)) for _ in range(2)] 
        if hasattr(self, 'delayTimes') == True:
            for i in range(self.delayTimes.size):
                pTargetOnset[0:self.maxT-1,i] = self.extended_later(self.muR, self.muS, self.sigR, self.sigS) #change self.sigma to self.ouVar[self.delayTimes[i]] for changing cov
                pTrialOnset[self.delayTimes[i]:self.delayTimes[i]+self.maxT,i] = pTargetOnset[0:self.maxT,i]
        else:  
            print("No delay distribution found. Please set via setDelayTimes first")   
        self.combined = pTrialOnset 

def fiteLATERModel(params, data, delayBins):
    """Fit extended LATER model fro Nakahara et al (2006) https://dl.acm.org/doi/10.1016/j.neunet.2006.07.001

    Args:
        params[0]: muR 
        params[1]: muS
        params[2]: sigR
        params[3]: sigS

    Returns:
        model_error: entropy between model and data to be passed into minimizer
    """
    # Set delay time bins to fit data to
    minDelay = 750
    maxDelay = 1100
    delays = np.linspace(minDelay, maxDelay, delayBins)

    # Set constants for model fitting
    shift = 0.5
    scale = 100
    trialBins = 50

    # Get Indices for delay bins
    delayData = dict()
    for i in range(delayBins):
        if i == delayBins-1:
            delayData[str(delays[i])] = data[(data['First Target Onset'] >= delays[i])] 
        else:
            delayData[str(delays[i])] = data[(data['First Target Onset'] >= delays[i]) & (data['First Target Onset'] < delays[i+1])]

    # Initialize model error
    model_error = 0 

    # Fit the model
    try:
        for delay in delays: # Loop model fit for same parameter set

            # Set up the model
            model = eLATER('Accumulator 1', time = np.arange(1,2000), muR = params[0], muS = params[1],
                           sigR = params[2], sigS = params[3], maxT = 700)
            model.setDelayTimes(np.linspace(delay, delay, 1)) # Set delay times equal to first delay bin
            model.computeModel() # Run Model

            # Align reaction times to delay bin
            data = delayData[str(delay)]['Reaction Time: First Target'] + delay
            data = data.to_numpy()
            data_pdf, _ = np.histogram(data, bins = trialBins, range =(750, 2000), density = True)
            
            # Sample data from model 
            model_sample = np.random.choice(1999, 10000, p = normalize_rt_pdfs(model.combined))
            model_pdf, _ = np.histogram(model_sample, bins = trialBins, range =(750, 2000), density = True)

            # Rescale model and data
            data_pdf = data_pdf + shift
            model_pdf = model_pdf + shift
            model_pdf = np.nan_to_num(model_pdf)
            data_pdf = np.nan_to_num(data_pdf)

            # Compute entropy between model and data, add to previous delay entropy
            model_error += entropy(model_pdf, data_pdf) * scale 
            
        return model_error #model error summed over all delay periods

    except:
        print('probabilities contain nan') #throw out bad models
        return np.inf


def fitBaselineModel(params, data, delayBins):
    """Fit baseline dynamic model to binned delayed saccade data with differential evolution. Model is fit to each delay bin RT distribution with the same parameter set,
    with the only difference being the onset time of the delay bin, which controls the baseline + timing

    Args:
        params[0]: x0 (see) 
        params[1]: theta
        params[2]: kappa
        params[3]: sigma
        params[4]: muR
        params[5]: sR
        params[6]: k 
        data (df): behavioural dataframe from free choice experiment. reaction times are returned in this function

    Returns:
        model_error: entropy calculation summed over delay bins OR
        np.inf: if model fails
    """

    # Set delay time bins to fit data to
    minDelay = 750
    maxDelay = 1100
    delays = np.linspace(minDelay, maxDelay, delayBins)

    # Set constants for model fitting
    shift = 0.5
    scale = 100
    trialBins = 50

    # Get Indices for delay bins
    delayData = dict()
    for i in range(delayBins):
        if i == delayBins-1:
            delayData[str(delays[i])] = data[(data['First Target Onset'] >= delays[i])] 
        else:
            delayData[str(delays[i])] = data[(data['First Target Onset'] >= delays[i]) & (data['First Target Onset'] < delays[i+1])]

    # Initialize model error
    model_error = 0 

    # Fit the model
    try:
        for delay in delays: # Loop model fit for same parameter set

            # Set up the model
            model = baselineDynamicIntegrator('Accumulator 1', time = np.arange(1,2000), xO = params[0],  #Initialize Model
                                theta = params[1], kappa = params[2], sigma = params[3], muR = params[4],
                                sR = params[5], k = params[6], scale = params[7], maxT = 700)
            model.setDelayTimes(np.linspace(delay, delay, 1)) # Set delay times equal to first delay bin
            model.BDI() # Run Model

            # Align reaction times to delay bin
            data = delayData[str(delay)]['Reaction Time: First Target'] + delay
            data = data.to_numpy()
            data_pdf, _ = np.histogram(data, bins = trialBins, range =(750, 2000), density = True)
            
            # Sample data from model 
            model_sample = np.random.choice(1999, 10000, p = model.combined / sum(model.combined))
            model_pdf, _ = np.histogram(model_sample, bins = trialBins, range =(750, 2000), density = True)

            # Rescale model and data
            data_pdf = data_pdf + shift
            model_pdf = model_pdf + shift
            model_pdf = np.nan_to_num(model_pdf)
            data_pdf = np.nan_to_num(data_pdf)

            # Compute entropy between model and data, add to previous delay entropy
            model_error += entropy(np.cumsum(model_pdf), np.cumsum(data_pdf)) * scale 
            
        return model_error #model error summed over all delay periods

    except:
        print('probabilities contain nan') #throw out bad models
        return np.inf

def normalize_rt_pdfs(pRT):
    pT = np.sum(pRT, 1) / np.shape(pRT)[1] 
    return pT / np.sum(pT) # Needs time dimension as axis [0] and probability density as axis [1]

def normalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def kl_divergence(p, q):
    p = p + .5
    q = q + .5
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))

def combine_anticipatory_sensory(pTrialOnset, pAntic, k):
    pAntic = pAntic * k
    t = np.stack((normalize_rt_pdfs(pTrialOnset), pAntic), axis = 1)
    return normalize_rt_pdfs(t)

def plotBDIFit(modelFit, params, data, numBins, minDelay, maxDelay, plotLabel):
    """ Plotting function for returning pdf and cdf of model vs data. 
    Arguments follow previous conventions. 
    """

    # Compute recovered model
    if params == None:
        recoveredModel = baselineDynamicIntegrator('Fitted Model', time = np.arange(1,2000), xO = modelFit.x[0], 
                        theta = modelFit.x[1] , kappa = modelFit.x[2], sigma = modelFit.x[3], muR = modelFit.x[4],
                        sR = modelFit.x[5], k = modelFit.x[6], scale = modelFit.x[7], maxT = 700)
    else:
        recoveredModel = baselineDynamicIntegrator('Fitted Model', time = np.arange(1,2000), xO = params[0], 
                    theta = params[1] , kappa = params[2], sigma = params[3], muR = params[4],
                    sR = params[5], k = params[6], scale = params[7], maxT = 700)
       
    
    # Return binned delay data for plotting
    if isinstance(data, pd.DataFrame):
        delayidx = binDelaydata(data, minDelay = minDelay, maxDelay = maxDelay, numBins = numBins)            
    delays= np.linspace(minDelay, maxDelay, numBins)
    
    # Set plotting colours
    _ , axs = plt.subplots(2, 2, figsize=(12,4))
    color = iter(cm.BuPu(np.linspace(0.5, 1, 10)))
    RT = []
    
    # Loop over delay bins and plot data overlayed with model
    for delay in delays:

        # Run model on single delay 
        recoveredModel.setDelayTimes(np.linspace(delay, delay, 1))
        recoveredModel.BDI()

        # Plot results
        c = next(color)
        axs[0,0].plot(recoveredModel.combined, c = c)

        if isinstance(data, pd.DataFrame):
            RT = delayidx[str(delay)]['Reaction Time: First Target'] + delay #Align to delay time 
            RT = RT.to_numpy()
            axs[0,0].hist(RT, bins = 50, density = True, color = c)
            
        axs[0,0].set_title('PDF Model vs Data')
        axs[0,0].set_xlim([0, 2000])
        axs[1,0].plot(np.cumsum(recoveredModel.combined), c =  c)
        axs[1,0].set_title('CDF Model')

        axs[0,1].plot(recoveredModel.ouMean)
        axs[0,1].fill_between(np.arange(1,2000), recoveredModel.ouMean - recoveredModel.ouVar,
                        recoveredModel.ouMean + recoveredModel.ouVar)

        axs[1,1].plot(recoveredModel.pAnticipatory)

    plt.savefig(rf"C:\Users\Brandon\Desktop\PhD\Baseline Dynamics\Baseline-Dynamics\Figures\{plotLabel}_model_fit", format='svg')
    plt.show()

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
    if numBins == 1:
        delayidx[str(delays[0])] = data
    else:
        for i in range(numBins):
            if i == numBins-1:
                delayidx[str(delays[i])] = data[(data['First Target Onset'] >= delays[i])] 
            else:
                delayidx[str(delays[i])] = data[(data['First Target Onset'] >= delays[i]) & (data['First Target Onset'] < delays[i+1])]

    return delayidx

def plotParamRTByDelay(params, var, percent_change, 
                       numSteps, numBins, minDelay, maxDelay, medorstd):
    """ Plotting function for visualizing different model parameters
    """       
    # Set base parameter set
    param_names = ['Starting Point', 'Theta', 'Kappa', 
               'Sigma', 'Rate of Rise', 'Variance of Rise', 'Scale', 'Mixing']       

    param_step = np.linspace(params[var]- percent_change * params[var], 
                             params[var]+ percent_change *params[var], numSteps)
    color = iter(cm.bone(np.linspace(0, 1, numSteps)))

    for step in param_step:
        params[var] = np.float64(step)
        model = baselineDynamicIntegrator('Fitted Model', time = np.arange(1,2000), xO = params[0], 
                    theta = params[1] + step , kappa = params[2], sigma = params[3], muR = params[4],
                    sR = params[5], k = params[6], scale = params[7], maxT = 700)

        # Get delay intervals
        delays= np.linspace(minDelay, maxDelay, numBins)
        
        # Initialize array for median RT
        compare_model = []

        # Loop over delay bins and plot data overlayed with model
        for delay in delays:

            # Run model on single delay 
            model.setDelayTimes(np.linspace(delay, delay, 1))
            model.BDI()

            # Sample model
            model_sample = np.random.choice(1999, 10000, p = model.combined / sum(model.combined))
            if medorstd == 'median':
                compare_model = np.append(compare_model, np.median(model_sample))
            else:
                compare_model = np.append(compare_model, np.std(model_sample))
        
        compare_model = compare_model / compare_model[0] #normalize to first value to see difference
        # Plot results
        c = next(color)
        plt.plot(compare_model, color = c)
    plt.title(param_names[var])

def plotModelParameters(params, minDelay, maxDelay, numBins):
    """ Plot model output without corresponding data fit

    Args:
        params (_type_): _description_
        minDelay (_type_): _description_
        maxDelay (_type_): _description_
        numBins (_type_): _description_
    """

    recoveredModel = baselineDynamicIntegrator('Fitted Model', time = np.arange(1,2000), xO = params[0], 
                theta = params[1] , kappa = params[2], sigma = params[3], muR = params[4],
                sR = params[5], k = params[6], scale = params[7], maxT = 700)
    delays= np.linspace(minDelay, maxDelay, numBins)
    
    # Set plotting colours
    _ , axs = plt.subplots(2, 2, figsize=(12,4))
    color = iter(cm.BuPu(np.linspace(0.5, 1, 10)))

    # Loop over delay bins and plot data overlayed with model
    for delay in delays:

        # Run model on single delay 
        recoveredModel.setDelayTimes(np.linspace(delay, delay, 1))
        recoveredModel.BDI()

        # Plot results
        c = next(color)
        axs[0,0].plot(recoveredModel.combined, c = c)
        axs[0,0].set_title('PDF Model vs Data')

        axs[1,0].plot(np.cumsum(recoveredModel.combined), c =  c)
        axs[1,0].set_title('CDF Model')

        axs[0,1].plot(recoveredModel.ouMean)
        axs[0,1].fill_between(np.arange(1,2000), recoveredModel.ouMean - recoveredModel.ouVar,
                              recoveredModel.ouMean + recoveredModel.ouVar)
        axs[1,1].plot(recoveredModel.pAnticipatory)

    plt.show()