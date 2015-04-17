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

outDir = "./out/"

def getBinomials(nVals,p,nPoints):
    nTrials = nVals.size
    dataMatrix = np.zeros((nTrials,nPoints))
    for i,n in enumerate(nVals):
        dataMatrix[i,:] = np.random.binomial(n,p,nPoints)
    return dataMatrix

def plotBinomials(dataMatrix,nVals,p):
    nTrials = nVals.size # rows are the trials
    for i,n in enumerate(nVals):
        fig = pPlotUtil.figure()
        xTrials = dataMatrix[i,:]
        dist = -2 * np.log(1-xTrials/(2*n))
        plt.hist(x=dist,bins=max(n/20,10))
        plt.xlabel("-2 * ln(1-X/2n)")
        plt.ylabel("Proportion")
        pPlotUtil.savefig(fig,outDir + "trial_n{:d}".format(int(n)))

_nVals = np.array([10,50,100,1e3,1e4,1e5])
pGenUtil.ensureDirExists(outDir)
_p=1/3.
_nPoints = 1e4
dataMatrix = getBinomials(_nVals,_p,_nPoints)
plotBinomials(dataMatrix,_nVals,_p)
