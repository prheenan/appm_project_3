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
pGenUtil.ensureDirExists(outDir)

mean = 0 
stdev = 1
epsilon = stdev/100 
nPoints = 1000
normDist = norm(loc=mean,scale=stdev)
offsets = np.linspace(mean-3*stdev,mean+3*stdev,nPoints)
probability = 2*(normDist.cdf((offsets+epsilon-mean)/stdev)-
                 normDist.cdf((offsets-epsilon-mean)/stdev))

fig = pPlotUtil.figure()
plt.plot(offsets,probability,'r-',
         label="mu = {:.1f}, sigma = {:.1f}, epsilon = {:.2f}".\
         format(mean,stdev,epsilon))
plt.xlabel("offset for CDF, c0")
plt.ylabel("Probability (arbitrary units) to land within epsilon of c0")
plt.axvline(0,color='k',linestyle='--',
            label="Maximum probability when centered near mu")
plt.legend(loc='best')
plt.title("Probability of landing within epsilon of c0 maximized near mu")
pPlotUtil.savefig(fig,outDir + "q1_1")
