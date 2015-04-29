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
        seqToAlign.append(seq)
        seqTxToAlign.append(tx)
    return seqToAlign,seqTxToAlign
    
def getProportionsAtIdx(idx,seq,chars,normalize):
    normChar = lambda x : x.upper()
    seqTmp = normChar(seq)
    seqTmp = str([seqTmp[i] for i in idx])
    counts = np.array([seqTmp.count(normChar(c)) for c in chars])
    props = counts / normalize
    return props

def getOptimalAminoAcidAlignment(nucleo1,nucleo2):
    tx1 = str(Seq(nucleo1))
    tx2 = str(Seq(nucleo2))
    print(tx1)
    print(tx2)
    alignments = pairwise2.align.globalxx(tx1, tx2)
    print('aligned!')
    bestAlignment = alignments[0]
    # return the actual strings associated with the alignment
    # dont necessarily care about the scores...
    return bestAlignment[0],bestAlignment[1]

def getMatchIdx(string1,string2):
    gapOffset1 = 0
    gapOffset2 = 0
    matchIdx1 = []
    matchIdx2 = []
    misMatchIdx1 = []
    misMatchIdx2 = []
    gapIdx1 = []
    gapIdx2 = []
    gapStr = '-'
    # XXX assume the sequences are of the same length.
    for idx,(char1,char2) in enumerate(zip(string1,
                                           string2)):
        effIdx1 = idx - gapOffset1
        effIdx2 = idx - gapOffset2
        if (char1 == char2):
            matchIdx1.append(effIdx1)
            matchIdx2.append(effIdx2)
        elif (char1 == gapStr or char2 == gapStr):
            # one was a gap
            if (char1 == gapStr):
                gapOffset1 += 1
            else:
                gapOffset2 += 1
            gapIdx1.append(effIdx1)
            gapIdx2.append(effIdx1)
        else:
            # simple mismatch
            misMatchIdx1.append(effIdx1)
            misMatchIdx2.append(effIdx1)
    # XXX for now, just return the mathc indices of the first
    return matchIdx1,misMatchIdx1,gapIdx1
            
    
def getNonGapProportions(nucleo1,nucleo2,chars):
    txAlign1,txAlign2 = getOptimalAminoAcidAlignment(nucleo1,nucleo2)
    matchIdx1,misMatchIdx1,gapIdx1 = getMatchIdx(txAlign1,txAlign2)
    # POST: know the matching / mismatching codons.
    # next, need to figure out 
    # make sure there are no duplicates
    assert (len(matchIdx1) + len(misMatchIdx1) + len(gapIdx1) == len(nucleo1))
    # make sure the indices cover everything
    assert (set(matchIdx1 +  misMatchIdx1 + gapIdx1) ==  
            set([i for i in range(len(nucleo1))]) )
    # make sure the indices have no overlap
    assert (set(matchIdx1) & set(mismatchIdx) & set(gapIdx1)) == set()
    # normalize by the total number of nucleotides at codon 'i' (equiv to
    # total number of amino acids)
    nAminoAcids = maxStrLen-nGaps
    codonSize = 3
    # Pi_a
    propTotal = np.zeros((codonSize,len(chars)))
    # D : number of mismatches 
    dMismatch = np.zeros((codonSize,len(chars)))
    idxTx = lambda x,offset: np.array(x)*codonSize+offset
    for offset in range(codonSize):
        # transform the index into what we want
        allIdx = idxTx(matchOrMisIdx,offset)
        propTotal[offset,:] = getProportionsAtIdx(allIdx,alignToNucleo,
                                                  chars,nAminoAcids)
        misIdx = idxTx(mismatchIdx,offset)
        # use a normalization of one, so we just get D 
        dMismatch[offset,:] = getProportionsAtIdx(misIdx,alignToNucleo,
                                                     chars,1)
        
    dTotal = np.sum(dMismatch,axis=1)
    print("{:d} gaps".format(nGaps))
    print("{:d} matches".format(len(matchIdx)))
    print("{:d} mismatches".format(len(mismatchIdx)))
    print("{:d} total matches and mismatches".format(nMatchMisMatch))
    return propTotal,dTotal,nMatchMisMatch

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
    print(piA)
    H = np.zeros(codonSize)
    H[:] = np.sum(piA * (1-piA),axis=1) # use [:] to make sure no funny indexing
    xVals = D/(H)
    n = lenV
    mFunc = lambda x: q12Dist(x,normalizer=H)
    # maximum likelihood estimator
    p = xVals/n
    gXBar,xx,normalStd = getDeltaStats(n,p,xVals,distFunc=mFunc)
    return gXBar,normalStd

def plotAll(outDir,gXBar,normalStd):
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
    pPlotUtil.savefig(fig,outDir + "q2_4_Khat")
    delim = "\t"
    print(delim.join(["Pos","K-Hat(var)","K-hat(stdv)"]))
    for i,(measured,theoryStd) \
        in enumerate(zip(gXBar,normalStd)):
        print("{:d}\t{:.3g}({:.3g})\t{:.3g}".format(i,measured,
                                                    theoryStd**2,theoryStd))
    
if __name__ == '__main__':
    dataDir = "../data/"
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
    seqAlignFile = "tx_align.fasta"
    # Posada, Selecting models of evolution, Chapter 10, Figure 10.3
    # If we model serum albumin as ~ ...
    # tau < 100 Millon years (mammals)
    tau = 80e6
    # substitutions: after 100 million years, expect 50 substitutions per 100
    # residues. so 0.5 /tau residue per unit time
    lambdaV = (50/100) * (1/tau)
    # generate the files for nucleotides translated (tx) coding sequences
    if determineTx:
        cdsSeq,cdsTx = saveTranslatedNucleotides(files,fileLabels,dataDir,
                                                 fileEx)
    if determineMSA:
        # choose the index for human...
        idxFirst = fileLabels.index('human')
        idxSecond = fileLabels.index('rat')
        # get piA, the number of proportion of each matched or mismatched
        # nucleic acid, D, and l fo the Felsenstein 1981 model
        piA,dMismatch,lenV = getNonGapProportions(cdsSeq[idxFirst],
                                                  cdsSeq[idxSecond],
                                                  chars)
        gXBar,normalStd = get1981ModelCodonPos(piA,dMismatch,lenV)
        plotAll(outDir,gXBar,normalStd)
    if (printPiA):
        printAminoInfo(piA,chars)


        

    


    
