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
fontsize=g_label

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

def getXBar(n,xTrials,distFunc):
    return distFunc(xTrials/n) 

def getDeltaStats(n,p,xTrials,normalizer=2):
    mu = n*p
    distFunc = lambda x : q12Dist(x,normalizer)
    gXBar = getXBar(n,xTrials,distFunc)
    sigma= np.sqrt(p*(1-p))
    gMu = distFunc(p)
    gPrimeMu = (1/(1-p/normalizer) )
    normalStd = abs(gPrimeMu) * sigma / np.sqrt(n)
    return gXBar,gMu,normalStd

def getDeltaModel(n,p,xTrials,normalizer=2,normMean=True):
    gXBar,gMu, normalStd = getDeltaStats(n,p,xTrials,normalizer)
    normalDist = norm(loc=0,scale=normalStd)
    dist = (gXBar- (gMu))
    distMean = np.mean(dist)
    distVar = np.std(dist)**2
    return dist,distMean,distVar,normalDist,normalStd

def getDeltaModelDistr(n,p,xTrials,coverage=10):
    # distFunc; what to call to get the distribution
    # n/p is the number of possible values for anything in xTrials
    # taking the log ceiling of this gives an upper bound for the number 
    # of bins for the log of xTrials
    dist,distMean,distVar,normalDist,normalStd = \
                        getDeltaModel(n,p,xTrials)
    sortedUniDist = np.sort(np.unique(dist))
    minStep = np.min(np.abs(np.diff(sortedUniDist)))
    xVals = np.linspace(-max(dist),max(dist),2*coverage*max(dist)/minStep)
    nBins = np.arange(-max(dist),max(dist),minStep)
    return dist,distMean,distVar,normalStd,normalDist,xVals,nBins

def plotSingleHist(n,p,xTrials,outDir):
    # coverage is just a plotting artifact
    fig = pPlotUtil.figure()
    # mu: expected value of Binomial(n,p)
    # effectie variance
    dist,distMean,distVar,normalStd,normalDist,xVals,nBins = \
                            getDeltaModelDistr(n,p,xTrials)
    normV = normHist(dist,nBins,\
                     label=("Actual Distr: Mean={:.4f},Stdev={:.4f}").\
                     format(distMean,np.sqrt(distVar)))
    rawPDF = normalDist.pdf(xVals)
    plt.plot(xVals,rawPDF,'r--',linewidth=5.0,
             label="Theorertical Distr: Stdev={:.4f}".\
             format(normalStd))
    plt.title("Histogram for g(xBar)-g(mu) for n={:d},p={:.2f}".\
              format(int(n),p),fontsize=g_title)
    plt.xlabel("(g(Xbar)-g(mu)) ~ Normal(0,[g'(x)*sigma]^2/n)",
               fontsize=g_label)
    plt.ylabel("Proportion",fontsize=g_label)
    plt.legend(frameon=False)
    pPlotUtil.tickAxisFont()
    catArr = list(rawPDF) + list(normV)
    plt.ylim([0,max(catArr)*1.2])
    plt.xlim([-max(nBins),max(nBins)])
    pPlotUtil.tickAxisFont()
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
                    plotSingleHist(n,p,dataMatrix[i,:],outDir)
    # plot the means and variances
    fig = pPlotUtil.figure()
    plt.subplot(1,2,1)
    plt.title("Mean of g(xBar)-g(mu) approaches 0",fontsize=fontsize)
    expMean = 0
    plt.plot(nVals,means,'ko',label="Actual Mean")
    plt.axhline(expMean,color='b',linestyle='--',
                label="Expected Mean: {:.2g}".format(expMean))
    plt.ylim(-min(means),max(means)*1.1)
    plt.xlabel("Value of n for binomial",fontsize=fontsize)
    plt.ylabel("Value of g(xBar)-g(mu)",fontsize=fontsize)
    plt.legend(fontsize=fontsize)
    pPlotUtil.tickAxisFont()
    plt.subplot(1,2,2)
    plt.semilogy(nVals,varReal,'ko',label="Actual Variance")
    plt.semilogy(nVals,varDist,'b--',label="Expected Variance")    
    plt.title("Variance of g(xBar)-g(mu)\n approaches expected",
              fontsize=fontsize)
    plt.xlabel("Value of n for binomial",fontsize=fontsize)
    plt.ylabel("Value of g(xBar) variance",fontsize=fontsize)
    pPlotUtil.tickAxisFont()
    plt.legend(fontsize=fontsize)
    pPlotUtil.savefig(fig,outDir + "MeanVar")

if __name__ == '__main__':
    _nVals = np.array([10,20,50,75,100,150,200,350,500,1000])
    pGenUtil.ensureDirExists(outDir)
    _p=1/3.
    _nPoints = 1e5
    dataMatrix = getBinomials(_nVals,_p,_nPoints)
    plotBinomials(dataMatrix,_nVals,_p)
