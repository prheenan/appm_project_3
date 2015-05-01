# force floating point division. Can still use integer with //
from __future__ import division
# This file is used for importing the common utilities classes.
import numpy as np
import matplotlib.pyplot as plt
# need to add the utilities class. Want 'home' to be platform independent
from os.path import expanduser
home = expanduser("~")
# get the utilties directory (assume it lives in ~/utilities/python)
# but simple to change
path= home +"/utilities/python"
import sys
sys.path.append(path)
# import the patrick-specific utilities
import GenUtilities  as pGenUtil
import PlotUtilities as pPlotUtil
import CheckpointUtilities as pCheckUtil
from q1_3 import getDeltaStats,q12Dist

# import biopython stuff
from Bio.Seq import Seq

from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Alphabet import IUPAC
from Bio import pairwise2
from collections import Counter
# parallelize the dynamic programming stuff.
from multiprocessing import Pool
# for plotting histograms
from scipy.stats import norm

from Bio.SubsMat import MatrixInfo as matlist
# according to this, BLOSUM-45 is better for long matrices
#http://www.ncbi.nlm.nih.gov/blast/html/sub_matrix.html
alignMat = matlist.blosum45
#return a global alignment for *translated* version of two nucleotide sequences
def globalAlign(humSeq, ratSeq, useBlosum=False,
                match=2, mismatch=-3, gapStart=-11, gapExt=-2):
# gap start/extension are defaults from: http://blast.ncbi.nlm.nih.gov/Blast.cgi?PAGE_TYPE=BlastSearch&PROG_DEF=blastn&BLAST_PROG_DEF=blastn&BLAST_SPEC=GlobalAln&LINK_LOC=BlastHomeLink
# match/mismatch from defaults here: http://blast.ncbi.nlm.nih.gov/Blast.cgi?PROGRAM=blastn&BLAST_PROGRAMS=blastn&PAGE_TYPE=BlastSearch&BLAST_SPEC=GlobalAln&LINK_LOC=blasttab&LAST_PAGE=blastp&BLAST_INIT=GlobalAln
  #scores: match, mismatch, gap start, gap extend
    # thhe BLOSUM matrices use 'X' for stop.
    stopStr = 'X'
    humTx = str(Seq(humSeq).translate(stop_symbol=stopStr))
    ratTx = str(Seq(ratSeq).translate(stop_symbol=stopStr))
    if (useBlosum):
        return pairwise2.align.globalds(humTx, ratTx, alignMat, gapStart,
                                        gapExt,one_alignment_only=True)
    else: 
        return pairwise2.align.globalms(humTx, ratTx, 
                                        match, mismatch, gapStart,
                                        gapExt, one_alignment_only=True)

#generate n shuffled versions of humSeq (a biopython Seq object)
#for each, find optimal alignment to ratSeq (DNA)
#given: Seq, Seq 
#return: [alignment scores]
def getShuffleAlign(args):
  i,humStr,ratStr,blosum = args
  alignment = globalAlign(humStr, ratStr,blosum)
  print("Global alignment: iteration... {:d} got score {:.5g}".\
        format(i,getAlignScore(alignment)))
  return alignment

def shuffledAligns(humSeq, ratSeq, n,blosum,nProc=10):
  alignments = []
  humList = list(humSeq)
  ratStr = str(ratSeq)
  p = Pool(nProc)
  # bit of a kludge; parallelization (apparently) messed up the pseudo random
  # number generator. just call shuffle again and save. works fine for
  # O(10^4) trials.
  shuffles = [ ''.join(humList) for i in range(n) 
               if not np.random.shuffle(humList)]
  # omg python is so awesome. make a pool of processes, map to our function
  # only wonky bit: have to pass the arguments as a tuple
  alignments = p.map(getShuffleAlign,[ (i,shuffles[i],ratStr,blosum) 
                                        for i in range(n)] )
  return alignments

#determine the fraction of alignments from shuffledAligns with higher scores
#than the alignment between the two sequences
#n is the number of shuffled sequences to create
#humSeq and ratSeq are expected as DNA sequences
def getPValue(humSeq, ratSeq, n=1000):
  optAlign = getAlignScore(globalAlign(str(humSeq), str(ratSeq)))
  opt = getAlignScore(optAlign)
  alignments = shuffledAligns(humSeq, ratSeq, n)
  scores = [getAlignScore(a) for a in alignments]
  highScores = len(filter(lambda x: x>opt, scores))
  return float(highScores)/n

def getCDS(record):
    # more or less copied from here:
#http://www2.warwick.ac.uk/fac/sci/moac/people/students/peter_cock/python/genbank/
    features = record.features
    for feature in features:
        # loo for the CDS, get its bounds and the actually translated thing
        if (feature.type == "CDS"):
            cdsObj =feature.location
            # for some reason, getting the translation is hairy.
            # returns an array, get the first element to get the string
            return int(cdsObj.start),int(cdsObj.end)

def getSequenceAndRec(fileName,formatIO="genbank"):
    with open(fileName, "r") as handle:
        records = [r for r in SeqIO.parse(handle, formatIO)]
        recTmp = records[0]
        return str(recTmp.seq),recTmp

def getSeqAndCDS(fileName):
    seq,record = getSequenceAndRec(fileName)
    startCDS,endCDS = getCDS(record)
    cdsSlice = slice(startCDS,endCDS)
    cdsStr = seq[cdsSlice]
    cdsSeq = Seq(cdsStr)
    translatedDNA = str(cdsSeq.translate())
    return seq,cdsSlice,translatedDNA,startCDS,endCDS

def saveTranslatedNucleotides(files,labels,dataDir,fileEx,printV=True):
    seqToAlign = []
    seqTxToAlign = []
    for f,label in zip(files,labels):
        txFileSave = dataDir + f + "_tx.fasta"
        seq, cds,tx,cdsI,cdsF =  getSeqAndCDS(dataDir + f + fileEx)
        cdsLen = len(seq[cds])
        seqLen = len(seq)
        utr5Len = cdsI
        utr3Len = seqLen-cdsLen-utr5Len
        cdsArr = [cdsI,cdsF]
        utr5=[0,cdsI-1]
        utr3=[cdsF,seqLen]
        if (printV):
            print(("{:s}({:s})\n" +
                   "\t5'UTR: {:s}, Len {:d}\n"+
                   "\tCDS  : {:s}, Len {:d}\n"+
                   "\t3'UTR: {:s}, Len {:d}\n"+
                   "\tTotal Len        {:d}\n"+
                   "\tTx (len {:d}) saved as {:s}_tx.fasta)").\
                  format(label,f,
                         utr5,utr5Len,
                         cdsArr,cdsLen,
                         utr3,utr3Len,seqLen,
                         len(tx),txFileSave))
        with open(txFileSave,'w') as fh:
            record = SeqRecord(Seq(tx,IUPAC.protein),
                   id=f)
            SeqIO.write(record, fh, "fasta")
        # get the coding sequence and its translation
        seqToAlign.append(seq[cds])
        seqTxToAlign.append(tx)
    return seqToAlign,seqTxToAlign
    
def getProportionsAtIdx(idx,seq,chars,normalize):
    normChar = lambda x : x.upper()
    seqTmp = normChar(seq)
    seqTmp = str([seqTmp[i] for i in idx])
    counts = np.array([seqTmp.count(normChar(c)) for c in chars])
    props = counts / normalize
    return props


def getMatchIdx(string1,string2):
  # get where the matches and mismatches between string1 and string2
  # (both from global alignment) occur, returning match/mismatch/gaps 
  # for string1
    gapOffset = 0
    matchIdx = []
    misMatchIdx = []
    gapIdx = []
    gapStr = '-'
    # XXX assume the sequences are of the same length.
    for idx,(char1,char2) in enumerate(zip(string1,
                                           string2)):
        effIdx1 = idx - gapOffset
        if (char1 == char2):
            matchIdx.append(effIdx1)
        elif (char1 == gapStr or char2 == gapStr):
            gapIdx.append(effIdx1)
        else:
            # simple mismatch
            misMatchIdx.append(effIdx1)
    return matchIdx,misMatchIdx,gapIdx
                
def getAlignScore(alignment):
  # return the first element (best), second score
  return alignment[0][2]

def getNonGapProportions(nucleo1,nucleo2,chars,forceReAlign,blosum,label):
  # translate the nucleotides, then globally align them.
    alignObj =pCheckUtil.getCheckpoint("./out/alignment" + label + ".pkl",
                                       globalAlign,forceReAlign,nucleo1,nucleo2,
                                       blosum)
    # get the first (only) alignment score
    alignS1S2ScoreStartEnd = alignObj[0]
    txAlign1,txAlign2 = alignS1S2ScoreStartEnd[0],alignS1S2ScoreStartEnd[1]
    alignScore = getAlignScore(alignObj)
    matchIdx,misMatchIdx,gapIdx = getMatchIdx(txAlign1,txAlign2)
    matchOrMisMatchIdx = matchIdx + misMatchIdx
    # POST: know the matching / mismatching codons.
    # next, need to figure out 
    # make sure there are no duplicates
    nAmino = len(txAlign1)
    nMatch = len(matchIdx)
    nMisMatch = len(misMatchIdx)
    nGap = len(gapIdx)
    fullLen = (nMatch + nMisMatch + nGap)
    assert (fullLen == nAmino) , \
               "Lengths {:d}/{:d} don't match".format(fullLen,nAmino)
    # make sure the indices cover everything
    assert (set(matchIdx +  misMatchIdx + gapIdx) ==  
            set([i for i in range(nAmino)]) )
    # make sure the indices have no overlap
    assert (set(matchIdx) & set(misMatchIdx) & set(gapIdx)) == set()
    # normalize by the total number of nucleotides at codon 'i' (equiv to
    # total number of amino acids)
    codonSize = 3
    # Pi_a
    propTotal = np.zeros((codonSize,len(chars)))
    # D : number of mismatches 
    dMismatch = np.zeros((codonSize,len(chars)))
    idxTx = lambda x,offset: np.array(x)*codonSize+offset
    for offset in range(codonSize):
        # transform the index into what we want
        allIdx = idxTx(matchOrMisMatchIdx,offset)
        propTotal[offset,:] = getProportionsAtIdx(allIdx,nucleo1,
                                                  chars,nAmino)
        # the 'real' mismatches can occur two ways.
        # (1) The amino acids match, but the nucleotides are off
        # (2) The amino acids don't match, neither do the nucleotides
        # We expect (1) to happen more frequently on the third codon
        # In order to check, we need to check *all* match and mismatch indices
        realMisIdx = [ i for i in allIdx
                       if  (i < min(len(nucleo1),len(nucleo2))) and
                       (nucleo1[i] != nucleo2[i]) ]
        # use a normalization of one, so we just get D                       
        dMismatch[offset,:] = getProportionsAtIdx(realMisIdx,nucleo1,
                                                     chars,1)
    dTotal = np.sum(dMismatch,axis=1)
    nNonGap = nMatch+nMisMatch
    print("{:d} gaps".format(nGap))
    print("{:d} matches".format(nMatch))
    print("{:d} mismatches".format(nMisMatch))
    print("{:d} total matches and mismatches".format(nMatch+nMisMatch))
    return propTotal,dTotal,nNonGap,alignScore

def printAminoInfo(piA,chars):
    delim = "\t\t"
    print("Nucleotide \t" + delim.join(chars))
    for i,row in enumerate(piA):
        print("Position {:d}\t".format(i) + 
              delim.join("{:.4g}".format(r) for r in row))


def get1981ModelCodonPos(piA,D,length):
    # piA is the proportion of base, row for each codon position, 
    # column for each {A,T,G,C}
    # D is the total count of bases. 
    codonSize = 3
    H = np.zeros(codonSize)
    H[:] = np.sum(piA * (1-piA),axis=1) # use [:] to make sure no funny indexing
    xVals = D/(H)
    n = lenV
    # maximum likelihood estimator
    p = xVals/n
    gXBar,xx,normalStd = getDeltaStats(n,p,xVals,normalizer=H)
    return gXBar,normalStd

def plotAll(outDir,gXBar,normalStd,label):
    fig = plt.figure()
    fontSize = 18
    positionX = np.arange(gXBar.size)
    plt.plot(positionX,gXBar,'ro',label="Data for K-hat")
    plt.errorbar(x=positionX,y=gXBar,yerr=normalStd,fmt='bx',
                 label="Theory")
    plt.xlabel("Position on codon",fontsize=fontSize)
    plt.ylabel("K-hat, E[substitutions/homologous base].",
               fontsize=fontSize)
    plt.legend(loc='best',fontsize=fontSize)
    fudge = 0.1
    plt.xlim([-fudge,max(positionX)+fudge])
    plt.title("K-hat versus expected K \n"+
              "lambda={:.3g}[1/year], tau={:.3g}[years]".\
              format(lambdaV,tau),fontsize=fontSize)
    pPlotUtil.savefig(fig,outDir + "q2_4_Khat" + label)
    delim = "\t"
    print(delim.join(["Pos","K-Hat(var)","K-hat(stdv)"]))
    for i,(measured,theoryStd) \
        in enumerate(zip(gXBar,normalStd)):
        print("{:d}\t{:.3g}({:.3g})\t{:.3g}".format(i,measured,
                                                    theoryStd**2,theoryStd))
    

def plotScoreHistograms(scores,fontSize,edgecolor='none'):
    meanScore = np.mean(scores)
    stdevScore = np.std(scores)
    step = 1
    bins = np.arange(min(scores)-step,max(scores)+step,1)
    plt.hist(scores, bins,normed=True,label="Distr. "+
             "mean:{:.3g}, var: {:.3g})".format(meanScore,stdevScore**2),
             alpha=0.25,edgecolor=edgecolor)
    plt.xlabel("Optimal alignment score for sequence",fontsize=fontSize)
    plt.ylabel("Occurence of score",fontsize=fontSize)
    pPlotUtil.tickAxisFont()
    return meanScore,stdevScore,bins

def plotAlignments(outDir,alignments,scoreOptimal,label):
    fontSize = 25
    scores = [ getAlignScore(a) for a in alignments ] 
    fig = pPlotUtil.figure(xSize=24,ySize=12)
    plt.subplot(1,2,1)
    meanScore,stdevScore,bins = plotScoreHistograms(scores,fontSize,'k')
    plt.title("Shuffled DNA alignment Histogram",
              fontsize=fontSize)
    pdfFunc = norm(loc=meanScore,scale=stdevScore).pdf(bins)
    plotPDF = lambda :  plt.plot(bins,pdfFunc,'g--',linewidth=3.0,
                                 label="Normal(mean,var)")
    plotPDF()
    plt.legend(fontsize=fontSize)
    ax = plt.subplot(1,2,2)
    plotScoreHistograms(scores,fontSize)
    plotPDF()
    zScore = (scoreOptimal-meanScore)/stdevScore
    print("Z Score for {:s} is {:.2f}".format(label,zScore))
    # ??? is this the real p Value? Dont think so
    extrProb = 1-norm().cdf(zScore)
    plt.title(("Histogram of optimal alignment score for {:d} trials\n" + 
               "Optimal score: {:d}*sigma from shuffled mean.\n"
               "P(shuffled score>=optimal) ~ {:.5g}").\
              format(len(scores),int(zScore),extrProb),fontsize=fontSize)
    plt.axvline(scoreOptimal,color='r',linestyle='--',
                label="Optimal global alignment score using {:s}: {:d}".\
                format(label,int(scoreOptimal)))
    plt.legend(loc='best',fontsize=fontSize)
    pPlotUtil.savefig(fig,outDir+ "q2Histograms" + label)

if __name__ == '__main__':
    dataDir = "../data/"
    # use the same seed always[!]
    np.random.seed(42) 
    outDir = "./out/"
    pGenUtil.ensureDirExists(outDir)
    files = ['sequence_human',
             'sequence_rat']
    fileEx = ".gb"
    # XXX make these labels better?
    fileLabels = ["human","rat"]
    # nucleotides
    chars = ['A','C','G','T']
    determineMSA = True
    determineTx = True
    printPiA = True
    forceAlignments = False
    seqAlignFile = "tx_align.fasta"
    histogramSize = 10000
    # Posada, Selecting models of evolution, Chapter 10, Figure 10.3
    # If we model serum albumin as ~ ...
    # tau < 100 Millon years (mammals)
    tau = 80e6
    # substitutions: after 100 million years, expect 50 substitutions per 100
    # residues. so 0.5 /tau residue per unit time
    lambdaV = (50/100) * (1/tau)
    # generate the files for nucleotides translated (tx) coding sequences
    workDir = "./tmp/"
    cdsSeq,cdsTx = pCheckUtil.getCheckpoint(workDir + "seq.pkl",\
                                            saveTranslatedNucleotides,\
                                            determineTx,files,fileLabels,
                                            dataDir,fileEx)
    ratSeq = cdsSeq[fileLabels.index('rat')]
    humanSeq = cdsSeq[fileLabels.index('human')]
    # get piA, the number of proportion of each matched or mismatched
    # nucleic acid, D, and l fo the Felsenstein 1981 model
    # use blosum in one, not in the other
    for blosum in [True,False]:
        label = "blosum" if blosum else "NCBI"
        ending = label + ".pkl"
        piA,dMismatch,lenV,score = \
                pCheckUtil.getCheckpoint(workDir + "align" + ending,\
                                         getNonGapProportions,determineMSA,
                                         humanSeq,ratSeq,chars,determineMSA,
                                         blosum,label)
        gXBar,normalStd = get1981ModelCodonPos(piA,dMismatch,lenV)
        plotAll(outDir,gXBar,normalStd,label)
        alignments = pCheckUtil.getCheckpoint(workDir+"alignHistogram"+ending,\
                                              shuffledAligns,forceAlignments,
                                              humanSeq,ratSeq,histogramSize,
                                              blosum)
        plotAlignments(outDir,alignments,score,label)
        if (printPiA):
            printAminoInfo(piA,chars)

        

    


    
