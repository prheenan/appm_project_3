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

def getBinomials(nVals,p,nPoints):
    nTrials = nVals.size
    dataMatrix = np.zeros((nTrials,nPoints))
    for i,n in enumerate(nVals):
        dataMatrix[i,:] = np.random.binomial(n,p,nPoints)
    return dataMatrix

def q12Dist(X,n):
    return -2 * np.log( 1 - X/(2*n))

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
        mu = n*p
        # g(mu)
        g_mu = q12Dist(mu,n)
        xBar = np.mean(xTrials)
        # g(XBar)
        g_xBar = q12Dist(xBar,n)
        # g'(mu)
        gPrimeMu = 1/(n-mu/2)
        # effectie variance
        stdevTheory= n*p*(1-p)
        stdevEff = np.abs((gPrimeMu*stdevTheory)/n)
        normal = norm(loc=g_mu,scale=stdevEff)
        dist = -2 * np.log(1-xTrials/(2*n))
        xVals = np.linspace(0,max(dist),nPoints)
        distApprox = normal.pdf(xVals)
        normalized = normHist(dist,max(n/20,10))
        plt.plot(xVals,(distApprox/max(distApprox))*max(normalized),'r-')
        plt.xlabel("-2 * ln(1-X/2n)")
        plt.ylabel("Proportion")
        pPlotUtil.savefig(fig,outDir + "trial_n{:d}".format(int(n)))

_nVals = np.array([10,50,100,1e3,1e4,1e5])
pGenUtil.ensureDirExists(outDir)
_p=1/3.
_nPoints = 1e4
dataMatrix = getBinomials(_nVals,_p,_nPoints)
plotBinomials(dataMatrix,_nVals,_p)
