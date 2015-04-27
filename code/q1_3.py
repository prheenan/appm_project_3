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

def q12Dist(xBar,normalizer=2):
    return -normalizer * np.log( 1 - xBar/normalizer)

def normHist(data,bins,**kwargs):
    counts,bins = np.histogram(data,bins)
    norm = counts / sum( np.diff(bins) * counts )
    plt.bar(left=bins[:-1],height=norm,width=np.diff(bins),**kwargs)
    return norm

def getDeltaModelDistr(p,n,xTrials,distFunc = q12Dist,coverage=10):
    # distFunc; what to call to get the distribution
    mu = n*p
    sigma= np.sqrt(p*(1-p))
    gMu = q12Dist(p)
    gPrimeMu = (1/(1-p/2) )
    normalStd = abs(gPrimeMu) * sigma / np.sqrt(n)
    normalDist = norm(loc=0,scale=normalStd)
    dist = (distFunc(xTrials/n) - gMu)
    # n/p is the number of possible values for anything in xTrials
    # taking the log ceiling of this gives an upper bound for the number 
    # of bins for the log of xTrials
    distMean = np.mean(dist)
    distVar = np.std(dist)**2
    sortedUniDist = np.sort(np.unique(dist))
    minStep = np.min(np.abs(np.diff(sortedUniDist)))
    xVals = np.linspace(-max(dist),max(dist),2*coverage*max(dist)/minStep)
    nBins = np.arange(-max(dist),max(dist),minStep)
    return xVals,dist,nBins,distMean,distVar,normalStd,normalDist

def plotSingleHist(xTrials,n,p,outDir):
    # coverage is just a plotting artifact
    fig = pPlotUtil.figure()
    # mu: expected value of Binomial(n,p)
    # effectie variance
    xVals,dist,nBins,distMean,distVar,normalStd,normalDist = getDeltaModelDistr(p,n,xTrials)
    normV = normHist(dist,nBins,\
                     label=("Actual Distr: Mean={:.4f},Stdev={:.4f}").\
                     format(distMean,np.sqrt(distVar)))
    rawPDF = normalDist.pdf(xVals)
    plt.plot(xVals,rawPDF,'r--',linewidth=5.0,
             label="Theorertical Distr: Stdev={:.4f}".\
             format(normalStd))
    plt.title("Histogram for g(x) for n={:d},p={:.2f}".format(int(n),p),
              fontsize=g_title)
    plt.xlabel("(g(Xbar)-g(mu)) ~ Normal(0,[g'(x)*sigma]^2/n)",
               fontsize=g_label)
    plt.ylabel("Proportion",fontsize=g_label)
    plt.legend(frameon=False)
    pPlotUtil.tickAxisFont()
    catArr = list(rawPDF) + list(normV)
    plt.ylim([0,max(catArr)*1.2])
    plt.xlim([-max(nBins),max(nBins)])
    pPlotUtil.savefig(fig,outDir + "trial_n{:d}".format(int(n)))
    #return the statistics for plotting
    return distMean,distVar,normalStd**2

def plotBinomials(dataMatrix,nVals,p):
    nTrials = nVals.size # rows are the trials
    # same the mean and variances...
    means = np.zeros(nTrials)
    varReal = np.zeros(nTrials)
    varDist = np.zeros(nTrials)
    for i,n in enumerate(nVals):
        means[i],varReal[i],varDist[i] =\
                    plotSingleHist(dataMatrix[i,:],n,p,outDir)
    # plot the means and variances
    fig = pPlotUtil.figure()
    plt.subplot(1,2,1)
    plt.title("Mean of g(x)-g(mu)\n approaches 0")
    plt.plot(nVals,means,'ko',label="Actual Mean",linewidth=0)
    plt.axhline(0,color='b',linestyle='--',
                label="Expected Mean")
    plt.ylim(-np.mean(np.abs(means)),max(means)*1.1)
    plt.xlabel("Value of n for binomial")
    plt.ylabel("Value of g(x) mean")
    plt.legend()
    plt.subplot(1,2,2)
    plt.semilogy(nVals,varReal,'ko-',label="Actual Variance")
    plt.semilogy(nVals,varDist,'b--',label="Expected Variance")    
    plt.title("Variance of g(x)-g(mu)\n approaches expected")
    plt.xlabel("Value of n for binomial")
    plt.ylabel("Value of g(x) variance")
    plt.legend()
    pPlotUtil.savefig(fig,outDir + "MeanVar")


_nVals = np.array([10,20,50,75,100,150,200,350,500,1000])
pGenUtil.ensureDirExists(outDir)
_p=1/3.
_nPoints = 1e5
dataMatrix = getBinomials(_nVals,_p,_nPoints)
plotBinomials(dataMatrix,_nVals,_p)
