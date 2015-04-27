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
            translatedDNA = feature.qualifiers['translation'][0]
            return cdsObj.start,cdsObj.end,translatedDNA

def getSequenceAndRec(fileName,formatIO="genbank"):
    with open(fileName, "r") as handle:
        records = [r for r in SeqIO.parse(handle, formatIO)]
        recTmp = records[0]
        return str(recTmp.seq),recTmp

def getSeqAndCDS(fileName):
    seq,record = getSequenceAndRec(fileName)
    startCDS,endCDS,translatedDNA = getCDS(record)
    cdsSlice = slice(startCDS,endCDS)
    return seq,cdsSlice,translatedDNA

def saveTranslatedNucleotides(files,labels,dataDir,fileEx):
    seqToAlign = []
    seqTxToAlign = []
    for f,label in zip(files,labels):
        txFileSave = dataDir + f + "_tx.fasta"
        seq, cds,tx =  getSeqAndCDS(dataDir + f + fileEx)
        cdsLen = len(seq)
        print(("{:s}({:s}) has cds of (zero indexed) {:s}, CDS(len {:d})."+
               " Tx to (saving as {:s}_tx.fasta)").\
              format(label,f,cds,cdsLen,txFileSave))
        with open(txFileSave,'w') as fh:
            record = SeqRecord(Seq(tx,IUPAC.protein),
                   id=f)
            SeqIO.write(record, fh, "fasta")
        seqToAlign.append(seq)
        seqTxToAlign.append(tx)
    return seqToAlign,seqTxToAlign
    
def getProportionsAtIdx(idx,seq,chars,normalize):
    seqTmp = seq.upper()
    seqTmp = str([seqTmp[i] for i in idx])
    counts = np.array([seqTmp.count(c) for c in chars])
    props = counts / normalize
    return props

def getNonGapProportions(pairwiseFile,alignToNucleo,alignToTranslated,chars):
    seqAlign,rec = getSequenceAndRec(pairwiseFile,'fasta')
    compareLen = min(len(seqAlign),len(alignToTranslated))
    # go through all the matching or mismatched codons.
    nucleoCompare = seqAlign[:compareLen]
    originalCompare = alignToTranslated[:compareLen]
    # XXX assume gaps are ' ' for now...
    gapStr = " " 
    # check that (1) not looking at a gap (2) our condition (match/mis) is true
    genIdx = lambda condition : \
             [i for i in range(compareLen) 
              if ( (nucleoCompare[i] != gapStr) and
                   condition(nucleoCompare[i],originalCompare[i]))]
    matchIdx = genIdx(lambda s1,s2: s1 == s2)
    mismatchIdx = genIdx(lambda s1,s2: s1 != s2)
    matchOrMisIdx = matchIdx + mismatchIdx
    assert (len(matchIdx) + len(mismatchIdx)) ==  compareLen
    assert (set(matchIdx) & set(mismatchIdx)) == set()
    # normalize by the total number of nucleotides at codon 'i' (equiv to
    # total number of amino acids)
    nAminoAcids = compareLen
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
    return propTotal,dTotal,compareLen

def getFelsensteinModel():
    

if __name__ == '__main__':
    dataDir = "../data/"
    files = ['sequence_human',
             'sequence_rat']
    fileEx = ".gb"
    # XXX make these labels better?
    fileLabels = ["human","rat"]
    # nucleotides
    chars = ['A','T','G','C']
    determineMSA = True
    determineTx = True
    seqAlignFile = "tx_align.fasta"
    # generate the files for nucleotides translated (tx) coding sequences
    if determineTx:
        cdsSeq,cdsTx = saveTranslatedNucleotides(files,fileLabels,dataDir,
                                                 fileEx)
    if determineMSA:
        # choose the index for human...
        idxToChoose = fileLabels.index('human')
        nucleoAlign = cdsSeq[idxToChoose]
        aminoAlign = cdsTx[idxToChoose]
        # use the BLAST file to figure out the 'pi' vector 
        # (propoprtion of bases)
        pairwiseFile = dataDir + seqAlignFile 
        # get piA, the number of proportion of each matched or mismatched
        # nucleic acid, D, and l fo the Felsenstein 1981 model
        piA,dMismatch,lenV = getNonGapProportions(pairwiseFile,nucleoAlign,
                                                 aminoAlign,chars)
        print(piA)
        print(dMismatch)
        print(lenV)
        

    


    
