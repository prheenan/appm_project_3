# force floating point division. Can still use integer with //
from __future__ import division
# This file is used for importing the common utilities classes.
import numpy as np
import matplotlib.pyplot as plt
# need to add the utilities class. Want 'home' to be platform independent
# import the patrick-specific utilities
import GenUtilities  as pGenUtil
import PlotUtilities as pPlotUtil
import CheckpointUtilities as pCheckUtil
from scipy.stats import norm

outDir = "./out/"
g_title = 20
g_label = 18

def getBinomials(nVals,p,nPoints):
    nTrials = nVals.size
    dataMatrix = np.zeros((nTrials,nPoints))
    for i,n in enumerate(nVals):
        dataMatrix[i,:] = np.random.binomial(n,p,nPoints)
    return dataMatrix

def q12Dist(xBar):
    return -2 * np.log( 1 - xBar/2)

def normHist(data,bins,**kwargs):
    counts,bins = np.histogram(data,bins,normed=False)
    normalized = counts / sum(counts * np.diff(bins))
    plt.bar(left=bins[:-1],height=normalized,width=np.mean(np.diff(bins)),
            **kwargs)
    return normalized

def plotBinomials(dataMatrix,nVals,p):
    nTrials = nVals.size # rows are the trials
    nPoints = 200
    for i,n in enumerate(nVals):
        fig = pPlotUtil.figure()
        xTrials = dataMatrix[i,:]
        # mu: expected value of Binomial(n,p)
        # effectie variance
        mu = n*p
        sigma= n*p*(1-p)
        gMu = q12Dist(mu/n)
        gPrimeMu = 1/(1-p/2)
        normalVar = (sigma*gPrimeMu)**2
        normalStd = np.sqrt(normalVar)
        normal = norm(loc=0,scale=normalStd)
        dist = np.sqrt(n) * (q12Dist(xTrials/n) - gMu)
        # n/p is the number of possible values for anything in xTrials
        # taking the log ceiling of this gives an upper bound for the number 
        # of bins for the log of xTrials
        nBins = np.ceil(np.log(n/p))
        xVals = np.linspace(min(dist),max(dist),1000)
        distMean = np.mean(dist)
        distVar = np.std(dist)**2
        nBins = np.arange(min(dist),max(dist),q12Dist(p))
        normalized = normHist(dist,nBins,
                              label=("sqrt(n)*(g(Xbar)-g(mu), "+
                                     "Mean={:.3f},Stdev={:.3f}").\
                              format(distMean,distVar))
        rawPDF = norm.pdf(xVals)
        theoryDist =  rawPDF 
        plt.plot(xVals,theoryDist,'r-',linewidth=3.0,
                 label="Normal(mu=0,var=[sigma*g'(mu)]^2). sqrt(var)={:.2f}".\
                 format(normalStd))
        plt.title("Histogram for g(x) for n={:d},p={:.2f}".format(int(n),p),
                  fontsize=g_title)
        plt.xlabel("sqrt(n)*(g(Xbar)-g(mu)) ~ Normal(0,[g'(x)*sigma]^2",
                   fontsize=g_label)
        plt.ylabel("Proportion",fontsize=g_label)
        plt.legend()
        pPlotUtil.tickAxisFont()
        pPlotUtil.savefig(fig,outDir + "trial_n{:d}".format(int(n)))

_nVals = np.array([10,50,100])
pGenUtil.ensureDirExists(outDir)
_p=1/3.
_nPoints = 1e5
dataMatrix = getBinomials(_nVals,_p,_nPoints)
plotBinomials(dataMatrix,_nVals,_p)
