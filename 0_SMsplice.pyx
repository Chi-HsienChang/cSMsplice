#distutils: extra_link_args=-fopenmp
from cython import parallel
from cython.parallel import prange
import numpy as np
import time
from sklearn.neighbors import KernelDensity
from awkde import GaussianKDE
from math import exp 
from libc.math cimport exp as c_exp
cimport openmp


def baseToInt(str base):
    if base == 'a': return 0
    elif base == 'c': return 1
    elif base == 'g': return 2
    elif base == 't': return 3
    else:
        print("nonstandard base encountered:", base)
        return -1

def intToBase(int i):
    if i == 0: return 'a'
    elif i == 1: return 'c'
    elif i == 2: return 'g'
    elif i == 3: return 't'
    else: 
        print("nonbase integer encountered:", i)
        return ''

def hashSequence(str seq):
    cdef int i
    cdef int sum = 0 
    cdef int l = len(seq)
    for i in range(l):
        sum += (4**(l-i-1))*baseToInt(seq[i])
    return sum
    
def unhashSequence(int num, int l):
    seq = ''
    for i in range(l):
        seq += intToBase(num // 4**(l-i-1))
        num -= (num // 4**(l-i-1))*(4**(l-i-1))
    return seq
    
def trueSequencesCannonical(genes, annotations, E = 0, I = 1, B3 = 3, B5 = 5):
    # Converts gene annotations to sequences of integers indicating whether the sequence is exonic, intronic, or splice site,
    # Inputs
    #   - genes: a biopython style dictionary of the gene sequences
    #   - annotations: the splicing annotations dictionary
    #   - E, I, B3, B5: the integer indicators for exon, intron, 3'ss, and 5'ss, respectively
    trueSeqs = {}
    for gene in annotations.keys():
        if gene not in genes.keys(): 
            print(gene, 'has annotation, but was not found in the fasta file of genes') 
            continue
        
        transcript = annotations[gene]
        if len(transcript) == 1: 
            trueSeqs[gene] = np.zeros(len(genes[gene]), dtype = int) + E
            continue # skip the rest for a single exon case
        
        # First exon 
        true = np.zeros(len(genes[gene]), dtype = int) + I
        three = transcript[0][0] - 1 # Marking the beginning of the first exon
        five = transcript[0][1] + 1
        true[range(three+1, five)] = E
        true[five] = B5
        
        # Internal exons 
        for exon in transcript[1:-1]:
            three = exon[0] - 1
            five = exon[1] + 1
            true[three] = B3
            true[five] = B5
            true[range(three+1, five)] = E
            
        # Last exon 
        three = transcript[-1][0] - 1
        true[three] = B3
        five = transcript[-1][1] + 1 # Marking the end of the last exon
        true[range(three+1, five)] = E
                
        trueSeqs[gene] = true
        
    return(trueSeqs)

def trainAllTriplets(sequences, cutoff = 10**(-5)):
    # Train maximum entropy models from input sequences with triplet conditions
    train = np.zeros((len(sequences),len(sequences[0])), dtype = int)
    for (i, seq) in enumerate(sequences):
        for j in range(len(seq)):
            train[i,j] = baseToInt(seq[j])
    prob = np.log(np.zeros(4**len(sequences[0])) + 4**(-len(sequences[0])))
    Hprev = -np.sum(prob*np.exp(prob))/np.log(2)
    H = -1
    sequences = np.zeros((4**len(sequences[0]),len(sequences[0])), dtype = int)
    l = len(sequences[0]) - 1 
    for i in range(sequences.shape[1]):
        sequences[:,i] = ([0]*4**(l-i) + [1]*4**(l-i) + [2]*4**(l-i) +[3]*4**(l-i))*4**i
    while np.abs(Hprev - H) > cutoff:
        #print(np.abs(Hprev - H))
        Hprev = H
        for pos in range(sequences.shape[1]):
            for base in range(4):
                Q = np.sum(train[:,pos] == base)/float(train.shape[0])
                if Q == 0: continue
                Qhat = np.sum(np.exp(prob[sequences[:,pos] == base]))
                prob[sequences[:,pos] == base] += np.log(Q) - np.log(Qhat)
                prob[sequences[:,pos] != base] += np.log(1-Q) - np.log(1-Qhat)
                
                for pos2 in np.setdiff1d(range(sequences.shape[1]), range(pos+1)):
                    for base2 in range(4):
                        Q = np.sum((train[:,pos] == base)*(train[:,pos2] == base2))/float(train.shape[0])
                        if Q == 0: continue
                        which = (sequences[:,pos] == base)*(sequences[:,pos2] == base2)
                        Qhat = np.sum(np.exp(prob[which]))
                        prob[which] += np.log(Q) - np.log(Qhat)
                        prob[np.invert(which)] += np.log(1-Q) - np.log(1-Qhat)
                        
                        for pos3 in np.setdiff1d(range(sequences.shape[1]), range(pos2+1)):
                            for base3 in range(4):
                                Q = np.sum((train[:,pos] == base)*(train[:,pos2] == base2)*(train[:,pos3] == base3))/float(train.shape[0])
                                if Q == 0: continue
                                which = (sequences[:,pos] == base)*(sequences[:,pos2] == base2)*(sequences[:,pos3] == base3)
                                Qhat = np.sum(np.exp(prob[which]))
                                prob[which] += np.log(Q) - np.log(Qhat)
                                prob[np.invert(which)] += np.log(1-Q) - np.log(1-Qhat)
        H = -np.sum(prob*np.exp(prob))/np.log(2)
    return np.exp(prob)

def structuralParameters(genes, annotations, minIL = 0):
    # Get the empirical length distributions for introns and single, first, middle, and last exons, as well as number exons per gene
    
    # Transitions
    numExonsPerGene = [] 
    
    # Length Distributions
    lengthSingleExons = []
    lengthFirstExons = []
    lengthMiddleExons = []
    lengthLastExons = []
    lengthIntrons = []
    
    for gene in genes:
        if len(annotations[gene]) == 0: 
            print('missing annotation for', gene)
            continue
        numExons = 0
        introns = []
        singleExons = []
        firstExons = []
        middleExons = []
        lastExons = []
        
        for transcript in annotations[gene].values():
            numExons += len(transcript)
            
            # First exon 
            three = transcript[0][0] # Make three the first base
            five = transcript[0][1] + 1
            if len(transcript) == 1: 
                singleExons.append((three, five-1))
                continue # skip the rest for a single exon case
            firstExons.append((three, five-1)) # since three is the first base
            
            # Internal exons 
            for exon in transcript[1:-1]:
                three = exon[0] - 1 
                introns.append((five+1,three-1))
                five = exon[1] + 1
                middleExons.append((three+1, five-1))
                
            # Last exon 
            three = transcript[-1][0] - 1
            introns.append((five+1,three-1))
            five = transcript[-1][1] + 1
            lastExons.append((three+1, five-1))
        
        geneIntronLengths = [minIL]
        for intron in set(introns):
            geneIntronLengths.append(intron[1] - intron[0] + 1)
        
        if np.min(geneIntronLengths) < minIL: continue
        
        for intron in set(introns): lengthIntrons.append(intron[1] - intron[0] + 1)
        for exon in set(singleExons): lengthSingleExons.append(exon[1] - exon[0] + 1)
        for exon in set(firstExons): lengthFirstExons.append(exon[1] - exon[0] + 1)
        for exon in set(middleExons): lengthMiddleExons.append(exon[1] - exon[0] + 1)
        for exon in set(lastExons): lengthLastExons.append(exon[1] - exon[0] + 1)
            
        numExonsPerGene.append(float(numExons)/len(annotations[gene]))
        
    return(numExonsPerGene, lengthSingleExons, lengthFirstExons, lengthMiddleExons, lengthLastExons, lengthIntrons)

def adaptive_kde_tailed(lengths, N, geometric_cutoff = .8, lower_cutoff=0):
    adaptive_kde = GaussianKDE(alpha = 1) 
    adaptive_kde.fit(np.array(lengths)[:,None]) 
    
    lengths = np.array(lengths)
    join = np.sort(lengths)[int(len(lengths)*geometric_cutoff)] 
    
    smoothed = np.zeros(N)
    smoothed[:join+1] = adaptive_kde.predict(np.arange(join+1)[:,None])
    
    s = 1-np.sum(smoothed)
    p = smoothed[join]
    smoothed[join+1:] = np.exp(np.log(p) + np.log(s/(s+p))*np.arange(1,len(smoothed)-join))
    smoothed[:lower_cutoff] = 0
    smoothed /= np.sum(smoothed)
    
    return(smoothed)
    
def geometric_smooth_tailed(lengths, N, bandwidth, join, lower_cutoff=0):
    lengths = np.array(lengths)
    smoothing = KernelDensity(bandwidth = bandwidth).fit(lengths[:, np.newaxis]) 
    
    smoothed = np.zeros(N)
    smoothed[:join+1] = np.exp(smoothing.score_samples(np.arange(join+1)[:,None]))
    
    s = 1-np.sum(smoothed)
    p = smoothed[join]
    smoothed[join+1:] = np.exp(np.log(p) + np.log(s/(s+p))*np.arange(1,len(smoothed)-join))
    smoothed[:lower_cutoff] = 0
    smoothed /= np.sum(smoothed)
    
    return(smoothed)

def maxEnt5(geneNames, genes, dir):
    # Get all the 5'SS maxent scores for each of the genes in geneNames
    scores = {}
    prob = np.load(dir + '/maxEnt5_prob.npy')
    prob0 = np.load(dir + '/maxEnt5_prob0.npy') 
        
    for gene in geneNames:
        sequence = str(genes[gene].seq).lower()
        sequence5 = np.array([hashSequence(sequence[i:i+9]) for i in range(len(sequence)-9+1)])
        scores[gene] = np.zeros(len(sequence)) - np.inf
        scores[gene][3:-5] = np.log2(prob[sequence5]) - np.log2(prob0[sequence5])
        scores[gene] = np.exp2(scores[gene])
    
    return scores
    
def maxEnt5_single(str seq, str dir):
    prob = np.load(dir + 'maxEnt5_prob.npy')
    prob0 = np.load(dir + 'maxEnt5_prob0.npy')
    
    seq = seq.lower()
    sequence5 = np.array([hashSequence(seq[i:i+9]) for i in range(len(seq)-9+1)])
    scores = np.log2(np.zeros(len(seq)))
    scores[3:-5] = np.log2(prob[sequence5]) - np.log2(prob0[sequence5])
    return np.exp2(scores)
    
def maxEnt3(geneNames, genes, dir):
    # Get all the 3'SS maxent scores for each of the genes in geneNames
    scores = {}
    prob0 = np.load(dir + 'maxEnt3_prob0.npy')
    prob1 = np.load(dir + 'maxEnt3_prob1.npy')
    prob2 = np.load(dir + 'maxEnt3_prob2.npy')
    prob3 = np.load(dir + 'maxEnt3_prob3.npy')
    prob4 = np.load(dir + 'maxEnt3_prob4.npy')
    prob5 = np.load(dir + 'maxEnt3_prob5.npy')
    prob6 = np.load(dir + 'maxEnt3_prob6.npy')
    prob7 = np.load(dir + 'maxEnt3_prob7.npy')
    prob8 = np.load(dir + 'maxEnt3_prob8.npy')
    
    prob0_0 = np.load(dir + 'maxEnt3_prob0_0.npy')
    prob1_0 = np.load(dir + 'maxEnt3_prob1_0.npy')
    prob2_0 = np.load(dir + 'maxEnt3_prob2_0.npy')
    prob3_0 = np.load(dir + 'maxEnt3_prob3_0.npy')
    prob4_0 = np.load(dir + 'maxEnt3_prob4_0.npy')
    prob5_0 = np.load(dir + 'maxEnt3_prob5_0.npy')
    prob6_0 = np.load(dir + 'maxEnt3_prob6_0.npy')
    prob7_0 = np.load(dir + 'maxEnt3_prob7_0.npy')
    prob8_0 = np.load(dir + 'maxEnt3_prob8_0.npy')
    
    for gene in geneNames:
        sequence = str(genes[gene].seq).lower()
        sequences23 = [sequence[i:i+23] for i in range(len(sequence)-23+1)]
        hash0 = np.array([hashSequence(seq[0:7]) for seq in sequences23])
        hash1 = np.array([hashSequence(seq[7:14]) for seq in sequences23])
        hash2 = np.array([hashSequence(seq[14:]) for seq in sequences23])
        hash3 = np.array([hashSequence(seq[4:11]) for seq in sequences23])
        hash4 = np.array([hashSequence(seq[11:18]) for seq in sequences23])
        hash5 = np.array([hashSequence(seq[4:7]) for seq in sequences23])
        hash6 = np.array([hashSequence(seq[7:11]) for seq in sequences23])
        hash7 = np.array([hashSequence(seq[11:14]) for seq in sequences23])
        hash8 = np.array([hashSequence(seq[14:18]) for seq in sequences23])
        
        probs = np.log2(prob0[hash0]) + np.log2(prob1[hash1]) + np.log2(prob2[hash2]) + \
            np.log2(prob3[hash3]) + np.log2(prob4[hash4]) - np.log2(prob5[hash5]) - \
            np.log2(prob6[hash6]) - np.log2(prob7[hash7]) - np.log2(prob8[hash8]) - \
            (np.log2(prob0_0[hash0]) + np.log2(prob1_0[hash1]) + np.log2(prob2_0[hash2]) + \
            np.log2(prob3_0[hash3]) + np.log2(prob4_0[hash4]) - np.log2(prob5_0[hash5]) - \
            np.log2(prob6_0[hash6]) - np.log2(prob7_0[hash7]) - np.log2(prob8_0[hash8]))
            
        scores[gene] = np.zeros(len(sequence)) - np.inf
        scores[gene][19:-3] = probs
        scores[gene] = np.exp2(scores[gene])
    
    return scores
    
def maxEnt3_single(str seq, str dir):
    prob0 = np.load(dir + 'maxEnt3_prob0.npy')
    prob1 = np.load(dir + 'maxEnt3_prob1.npy')
    prob2 = np.load(dir + 'maxEnt3_prob2.npy')
    prob3 = np.load(dir + 'maxEnt3_prob3.npy')
    prob4 = np.load(dir + 'maxEnt3_prob4.npy')
    prob5 = np.load(dir + 'maxEnt3_prob5.npy')
    prob6 = np.load(dir + 'maxEnt3_prob6.npy')
    prob7 = np.load(dir + 'maxEnt3_prob7.npy')
    prob8 = np.load(dir + 'maxEnt3_prob8.npy')
    
    prob0_0 = np.load(dir + 'maxEnt3_prob0_0.npy')
    prob1_0 = np.load(dir + 'maxEnt3_prob1_0.npy')
    prob2_0 = np.load(dir + 'maxEnt3_prob2_0.npy')
    prob3_0 = np.load(dir + 'maxEnt3_prob3_0.npy')
    prob4_0 = np.load(dir + 'maxEnt3_prob4_0.npy')
    prob5_0 = np.load(dir + 'maxEnt3_prob5_0.npy')
    prob6_0 = np.load(dir + 'maxEnt3_prob6_0.npy')
    prob7_0 = np.load(dir + 'maxEnt3_prob7_0.npy')
    prob8_0 = np.load(dir + 'maxEnt3_prob8_0.npy')
    
    seq = seq.lower()
    sequences23 = [seq[i:i+23] for i in range(len(seq)-23+1)]
    hash0 = np.array([hashSequence(seq[0:7]) for seq in sequences23])
    hash1 = np.array([hashSequence(seq[7:14]) for seq in sequences23])
    hash2 = np.array([hashSequence(seq[14:]) for seq in sequences23])
    hash3 = np.array([hashSequence(seq[4:11]) for seq in sequences23])
    hash4 = np.array([hashSequence(seq[11:18]) for seq in sequences23])
    hash5 = np.array([hashSequence(seq[4:7]) for seq in sequences23])
    hash6 = np.array([hashSequence(seq[7:11]) for seq in sequences23])
    hash7 = np.array([hashSequence(seq[11:14]) for seq in sequences23])
    hash8 = np.array([hashSequence(seq[14:18]) for seq in sequences23])
    
    probs = np.log2(prob0[hash0]) + np.log2(prob1[hash1]) + np.log2(prob2[hash2]) + \
            np.log2(prob3[hash3]) + np.log2(prob4[hash4]) - np.log2(prob5[hash5]) - \
            np.log2(prob6[hash6]) - np.log2(prob7[hash7]) - np.log2(prob8[hash8]) - \
            (np.log2(prob0_0[hash0]) + np.log2(prob1_0[hash1]) + np.log2(prob2_0[hash2]) + \
            np.log2(prob3_0[hash3]) + np.log2(prob4_0[hash4]) - np.log2(prob5_0[hash5]) - \
            np.log2(prob6_0[hash6]) - np.log2(prob7_0[hash7]) - np.log2(prob8_0[hash8]))
            
    scores = np.log2(np.zeros(len(seq)))
    scores[19:-3] = probs
    return np.exp2(scores)

def sreScores_single(str seq, double [:] sreScores, int kmer = 6):
    indices = [hashSequence(seq[i:i+kmer]) for i in range(len(seq)-kmer+1)]
    sequenceSRES = [sreScores[indices[i]] for i in range(len(indices))]
    return sequenceSRES

def get_all_5ss(gene, reference, genes):
    # Get all the 5'SS for a gene based on the annotation in reference
    info = genes[gene].description.split(' ')
    if reference.loc[gene,6] == ',': exonEnds = []
    else: exonEnds = [int(start)-1 for start in reference.loc[gene,6].split(',')[:-1]] # intron starts -> exon ends
    if reference.loc[gene,7] == ',': exonStarts = []
    else: exonStarts = [int(end)+1 for end in reference.loc[gene,7].split(',')[:-1]] # exon starts -> intron ends
    
    if info[6] == 'Strand:-': 
        stop = int(info[5][5:])
        annnotation = [stop - exonStarts[i-1] + 2 for i in range(len(exonStarts),0,-1)]      
        
    elif info[6] == 'Strand:+': 
        start = int(info[4][6:])
        annnotation = [exonEnds[i] - start + 1 for i in range(len(exonEnds))]
        
    return(annnotation)

def get_all_3ss(gene, reference, genes):
    # Get all the 3'SS for a gene based on the annotation in reference
    info = genes[gene].description.split(' ')
    if reference.loc[gene,6] == ',': exonEnds = []
    else: exonEnds = [int(start)-1 for start in reference.loc[gene,6].split(',')[:-1]] # intron starts -> exon ends
    if reference.loc[gene,7] == ',': exonStarts = []
    else: exonStarts = [int(end)+1 for end in reference.loc[gene,7].split(',')[:-1]] # exon starts -> intron ends
    
    if info[6] == 'Strand:-': 
        stop = int(info[5][5:])
        annnotation = [stop - exonEnds[i-1] - 2 for i in range(len(exonEnds),0,-1)]      
        
    elif info[6] == 'Strand:+': 
        start = int(info[4][6:])
        annnotation = [exonStarts[i] - start - 3 for i in range(len(exonStarts))]
        
    return(annnotation)

def get_hexamer_real_decoy_counts(geneNames, trueSeqs, decoySS, genes, kmer, sreEffect5_exon, sreEffect5_intron, sreEffect3_exon, sreEffect3_intron, B3 = 3, B5 = 5):
    # Get the counts of hexamers in the flanking regions for real and decoy ss with restriction to exons and introns for the real ss
    true_counts_5_intron = np.zeros(4**kmer, dtype = np.dtype("i"))
    true_counts_5_exon = np.zeros(4**kmer, dtype = np.dtype("i"))
    true_counts_3_intron = np.zeros(4**kmer, dtype = np.dtype("i"))
    true_counts_3_exon = np.zeros(4**kmer, dtype = np.dtype("i"))
    decoy_counts_5_intron = np.zeros(4**kmer, dtype = np.dtype("i"))
    decoy_counts_5_exon = np.zeros(4**kmer, dtype = np.dtype("i"))
    decoy_counts_3_intron = np.zeros(4**kmer, dtype = np.dtype("i"))
    decoy_counts_3_exon = np.zeros(4**kmer, dtype = np.dtype("i"))
    
    for gene in geneNames:
        trueThrees = np.nonzero(trueSeqs[gene] == B3)[0][:-1]
        trueFives = np.nonzero(trueSeqs[gene] == B5)[0][1:]
        for i in range(len(trueThrees)):
            three = trueThrees[i]
            five = trueFives[i]
            
            # 3'SS
            sequence = str(genes[gene].seq[three+4:three+sreEffect3_exon+1].lower())
            if len(sequence) == 0: sequence = str(genes[gene].seq[three+4:].lower())
            if five-3 < three+sreEffect3_exon+1: sequence = str(genes[gene].seq[three+4:five-3].lower())
            if len(sequence) < kmer: continue
            sequence = np.array([hashSequence(sequence[i:i+kmer]) for i in range(len(sequence)-kmer+1)])
            for s in sequence: true_counts_3_exon[s] += 1
            
            sequence = str(genes[gene].seq[three-sreEffect3_intron:three-19].lower())
            if len(sequence) == 0: sequence = str(genes[gene].seq[:three-19].lower())
            if len(sequence) < kmer: continue
            sequence = np.array([hashSequence(sequence[i:i+kmer]) for i in range(len(sequence)-kmer+1)])
            for s in sequence: true_counts_3_intron[s] += 1
                
            # 5'SS
            sequence = str(genes[gene].seq[five-sreEffect5_exon:five-3].lower())
            if len(sequence) == 0: sequence = str(genes[gene].seq[:five-3].lower())
            if five-sreEffect5_exon < three+4: sequence = str(genes[gene].seq[three+4:five-3].lower())
            if len(sequence) < kmer: continue
            sequence = np.array([hashSequence(sequence[i:i+kmer]) for i in range(len(sequence)-kmer+1)])
            for s in sequence: true_counts_5_exon[s] += 1
            
            sequence = str(genes[gene].seq[five+6:five+sreEffect5_intron+1].lower())
            if len(sequence) == 0: sequence = str(genes[gene].seq[five+6:].lower())
            if len(sequence) < kmer: continue
            sequence = np.array([hashSequence(sequence[i:i+kmer]) for i in range(len(sequence)-kmer+1)])
            for s in sequence: true_counts_5_intron[s] += 1
        
        decoyThrees = np.nonzero(decoySS[gene] == B3)[0]
        decoyFives = np.nonzero(decoySS[gene] == B5)[0]
        for ss in decoyFives:
            sequence = str(genes[gene].seq[ss-sreEffect5_exon:ss-3].lower())
            if len(sequence) == 0: sequence = str(genes[gene].seq[:ss-3].lower())
            if len(sequence) < kmer: continue
            sequence = np.array([hashSequence(sequence[i:i+kmer]) for i in range(len(sequence)-kmer+1)])
            for s in sequence: decoy_counts_5_exon[s] += 1
            
            sequence = str(genes[gene].seq[ss+6:ss+sreEffect5_intron+1].lower())
            if len(sequence) == 0: sequence = str(genes[gene].seq[ss+6:].lower())
            if len(sequence) < kmer: continue
            sequence = np.array([hashSequence(sequence[i:i+kmer]) for i in range(len(sequence)-kmer+1)])
            for s in sequence: decoy_counts_5_intron[s] += 1
    
        for ss in decoyThrees:
            sequence = str(genes[gene].seq[ss+4:ss+sreEffect3_exon+1].lower())
            if len(sequence) == 0: sequence = str(genes[gene].seq[ss+4:].lower())
            if len(sequence) < kmer: continue
            sequence = np.array([hashSequence(sequence[i:i+kmer]) for i in range(len(sequence)-kmer+1)])
            for s in sequence: decoy_counts_3_exon[s] += 1
            
            sequence = str(genes[gene].seq[ss-sreEffect3_intron:ss-19].lower())
            if len(sequence) == 0: sequence = str(genes[gene].seq[:ss-19].lower())
            if len(sequence) < kmer: continue
            sequence = np.array([hashSequence(sequence[i:i+kmer]) for i in range(len(sequence)-kmer+1)])
            for s in sequence: decoy_counts_3_intron[s] += 1
    
    return(true_counts_5_intron, true_counts_5_exon, true_counts_3_intron, true_counts_3_exon, 
           decoy_counts_5_intron, decoy_counts_5_exon, decoy_counts_3_intron, decoy_counts_3_exon)

def get_hexamer_counts(geneNames, set1, set2, genes, kmer, sreEffect5_exon, sreEffect5_intron, sreEffect3_exon, sreEffect3_intron, B3 = 3, B5 = 5):
    # Get the counts of hexamers in the flanking regions for two sets of ss
    set1_counts_5_intron = np.zeros(4**kmer, dtype = np.dtype("i"))
    set1_counts_5_exon = np.zeros(4**kmer, dtype = np.dtype("i"))
    set1_counts_3_intron = np.zeros(4**kmer, dtype = np.dtype("i"))
    set1_counts_3_exon = np.zeros(4**kmer, dtype = np.dtype("i"))
    set2_counts_5_intron = np.zeros(4**kmer, dtype = np.dtype("i"))
    set2_counts_5_exon = np.zeros(4**kmer, dtype = np.dtype("i"))
    set2_counts_3_intron = np.zeros(4**kmer, dtype = np.dtype("i"))
    set2_counts_3_exon = np.zeros(4**kmer, dtype = np.dtype("i"))
    
    for gene in geneNames:
        set1Threes = np.nonzero(set1[gene] == B3)[0]
        set1Fives = np.nonzero(set1[gene] == B5)[0]
        set2Threes = np.nonzero(set2[gene] == B3)[0]
        set2Fives = np.nonzero(set2[gene] == B5)[0]
        
        for ss in set1Fives:
            sequence = str(genes[gene].seq[ss-sreEffect5_exon:ss-3].lower())
            if len(sequence) == 0: sequence = str(genes[gene].seq[:ss-3].lower())
            if len(sequence) < kmer: continue
            sequence = np.array([hashSequence(sequence[i:i+kmer]) for i in range(len(sequence)-kmer+1)])
            for s in sequence: set1_counts_5_exon[s] += 1
            
            sequence = str(genes[gene].seq[ss+6:ss+sreEffect5_intron+1].lower())
            if len(sequence) == 0: sequence = str(genes[gene].seq[ss+6:].lower())
            if len(sequence) < kmer: continue
            sequence = np.array([hashSequence(sequence[i:i+kmer]) for i in range(len(sequence)-kmer+1)])
            for s in sequence: set1_counts_5_intron[s] += 1
    
        for ss in set1Threes:
            sequence = str(genes[gene].seq[ss+4:ss+sreEffect3_exon+1].lower())
            if len(sequence) == 0: sequence = str(genes[gene].seq[ss+4:].lower())
            if len(sequence) < kmer: continue
            sequence = np.array([hashSequence(sequence[i:i+kmer]) for i in range(len(sequence)-kmer+1)])
            for s in sequence: set1_counts_3_exon[s] += 1
            
            sequence = str(genes[gene].seq[ss-sreEffect3_intron:ss-19].lower())
            if len(sequence) == 0: sequence = str(genes[gene].seq[:ss-19].lower())
            if len(sequence) < kmer: continue
            sequence = np.array([hashSequence(sequence[i:i+kmer]) for i in range(len(sequence)-kmer+1)])
            for s in sequence: set1_counts_3_intron[s] += 1
        
        for ss in set2Fives:
            sequence = str(genes[gene].seq[ss-sreEffect5_exon:ss-3].lower())
            if len(sequence) == 0: sequence = str(genes[gene].seq[:ss-3].lower())
            if len(sequence) < kmer: continue
            sequence = np.array([hashSequence(sequence[i:i+kmer]) for i in range(len(sequence)-kmer+1)])
            for s in sequence: set2_counts_5_exon[s] += 1
            
            sequence = str(genes[gene].seq[ss+6:ss+sreEffect5_intron+1].lower())
            if len(sequence) == 0: sequence = str(genes[gene].seq[ss+6:].lower())
            if len(sequence) < kmer: continue
            sequence = np.array([hashSequence(sequence[i:i+kmer]) for i in range(len(sequence)-kmer+1)])
            for s in sequence: set2_counts_5_intron[s] += 1
    
        for ss in set2Threes:
            sequence = str(genes[gene].seq[ss+4:ss+sreEffect3_exon+1].lower())
            if len(sequence) == 0: sequence = str(genes[gene].seq[ss+4:].lower())
            if len(sequence) < kmer: continue
            sequence = np.array([hashSequence(sequence[i:i+kmer]) for i in range(len(sequence)-kmer+1)])
            for s in sequence: set2_counts_3_exon[s] += 1
            
            sequence = str(genes[gene].seq[ss-sreEffect3_intron:ss-19].lower())
            if len(sequence) == 0: sequence = str(genes[gene].seq[:ss-19].lower())
            if len(sequence) < kmer: continue
            sequence = np.array([hashSequence(sequence[i:i+kmer]) for i in range(len(sequence)-kmer+1)])
            for s in sequence: set2_counts_3_intron[s] += 1
    
    return(set1_counts_5_intron, set1_counts_5_exon, set1_counts_3_intron, set1_counts_3_exon, 
           set2_counts_5_intron, set2_counts_5_exon, set2_counts_3_intron, set2_counts_3_exon)

def get_hexamer_real_decoy_scores(geneNames, trueSeqs, decoySS, genes, kmer, sreEffect5_exon, sreEffect5_intron, sreEffect3_exon, sreEffect3_intron):
    # Get the real versus decoy scores for all hexamers
    true_counts_5_intron, true_counts_5_exon, true_counts_3_intron, true_counts_3_exon, decoy_counts_5_intron, decoy_counts_5_exon, decoy_counts_3_intron, decoy_counts_3_exon = get_hexamer_real_decoy_counts(geneNames, trueSeqs, decoySS, genes, kmer = kmer, sreEffect5_exon = sreEffect5_exon, sreEffect5_intron = sreEffect5_intron, sreEffect3_exon = sreEffect3_exon, sreEffect3_intron = sreEffect3_intron)
    
    # Add pseudocounts
    true_counts_5_intron = true_counts_5_intron + 1
    true_counts_5_exon = true_counts_5_exon + 1
    true_counts_3_intron = true_counts_3_intron + 1
    true_counts_3_exon = true_counts_3_exon + 1
    decoy_counts_5_intron = decoy_counts_5_intron + 1
    decoy_counts_5_exon = decoy_counts_5_exon + 1
    decoy_counts_3_intron = decoy_counts_3_intron + 1
    decoy_counts_3_exon = decoy_counts_3_exon + 1
    
    true_counts_intron = true_counts_5_intron + true_counts_3_intron
    true_counts_exon = true_counts_5_exon + true_counts_3_exon
    decoy_counts_intron = decoy_counts_5_intron + decoy_counts_3_intron
    decoy_counts_exon = decoy_counts_5_exon + decoy_counts_3_exon
    
    trueFreqs_intron = np.exp(np.log(true_counts_intron) - np.log(np.sum(true_counts_intron))) 
    decoyFreqs_intron = np.exp(np.log(decoy_counts_intron) - np.log(np.sum(decoy_counts_intron)))
    trueFreqs_exon = np.exp(np.log(true_counts_exon) - np.log(np.sum(true_counts_exon)))
    decoyFreqs_exon = np.exp(np.log(decoy_counts_exon) - np.log(np.sum(true_counts_exon)))
    
    sreScores_intron = np.exp(np.log(true_counts_intron) - np.log(np.sum(true_counts_intron)) 
                              - np.log(decoy_counts_intron) + np.log(np.sum(decoy_counts_intron)))
    sreScores_exon = np.exp(np.log(true_counts_exon) - np.log(np.sum(true_counts_exon)) 
                            - np.log(decoy_counts_exon) + np.log(np.sum(decoy_counts_exon)))
    
    sreScores3_intron = np.exp(np.log(true_counts_3_intron) - np.log(np.sum(true_counts_3_intron)) 
                                - np.log(decoy_counts_3_intron) + np.log(np.sum(decoy_counts_3_intron)))
    sreScores3_exon = np.exp(np.log(true_counts_3_exon) - np.log(np.sum(true_counts_3_exon)) 
                              - np.log(decoy_counts_3_exon) + np.log(np.sum(decoy_counts_3_exon)))
    
    sreScores5_intron = np.exp(np.log(true_counts_5_intron) - np.log(np.sum(true_counts_5_intron)) 
                                - np.log(decoy_counts_5_intron) + np.log(np.sum(decoy_counts_5_intron)))
    sreScores5_exon = np.exp(np.log(true_counts_5_exon) - np.log(np.sum(true_counts_5_exon)) 
                              - np.log(decoy_counts_5_exon) + np.log(np.sum(decoy_counts_5_exon)))
    
    return(sreScores_intron, sreScores_exon, sreScores3_intron, sreScores3_exon, sreScores5_intron, sreScores5_exon)
    
def score_sequences(sequences, double [:, :] exonicSREs5s, double [:, :] exonicSREs3s, double [:, :] intronicSREs5s, double [:, :] intronicSREs3s, int k = 6, int sreEffect5_exon = 80, int sreEffect5_intron = 80, int sreEffect3_exon = 80, int sreEffect3_intron = 80, meDir = ''): #/home/kmccue/projects/scfg/code/maxEnt/triplets-9mer-max/
    # Inputs:
    #  - sequences: batch_size lenght list of sequence strings, maybe need about 21 kb/base for the longest string
    #  - transitions: list of two values [pME, pEE]
    #  - pIL: numpy array of intron length distribution
    #  - pELS: decay coefficient for transcribing state eg 1/10000
    #  - pELF: numpy array of first exon length distribution
    #  - pELM: numpy array of middle exon length distribution
    #  - pELL: numpy array of last exon length distribution
    #  - delay: number of bases to delay predictions by
    #  - sreScores: sre scores for each kmer
    #  - meDir: location of the maxent model lookups
    # Outputs:
    #  - bestPath: numpy array containing the cotranscriptional viterbi best paths for each input sequence
    #  - loglik: numpy array containing the calculated log likelihoods
    #  - tbStates: numpy array containing the states at which a traceback event occurred
    #  - tblast3: numpy array containing the last 3'ss when which a traceback event occurred
    #  - emissions5: final 5' emissions
    #  - emissions3: final 3' emissions
    #  - traceback5: the 5'ss tracebacks
    #  - traceback3: the 3'ss tracebacks
    cdef int batch_size, g, L, d, i, t, ssRange
    batch_size = len(sequences)
    cdef int [:] lengths = np.zeros(batch_size, dtype=np.dtype("i"))
    
    # Collect the lengths of each sequence in the batch
    for g in range(batch_size): 
        lengths[g] = len(sequences[g])
    L = np.max(lengths.base)
    
    cdef double [:, :] emissions3 = np.log(np.zeros((batch_size, L), dtype=np.dtype("d")))
    cdef double [:, :] emissions5 = np.log(np.zeros((batch_size, L), dtype=np.dtype("d")))
    
    # Get the emissions and apply sre scores to them
    for g in range(batch_size): 
        # 5'SS exonic effects (upstream)
        ssRange = 3
        emissions5.base[g,:lengths[g]] = np.log(maxEnt5_single(sequences[g].lower(), meDir))
        emissions5.base[g,k+ssRange:lengths[g]] += np.cumsum(exonicSREs5s.base[g,:lengths[g]-k+1])[:-1-ssRange]
        emissions5.base[g,sreEffect5_exon+1:lengths[g]] -= np.cumsum(exonicSREs5s.base[g,:lengths[g]-k+1])[:-(sreEffect5_exon+1)+(k-1)]
        
        # 3'SS intronic effects (upstream)
        ssRange = 19
        emissions3.base[g,:lengths[g]] = np.log(maxEnt3_single(sequences[g].lower(), meDir))
        emissions3.base[g,k+ssRange:lengths[g]] += np.cumsum(intronicSREs3s.base[g,:lengths[g]-k+1])[:-1-ssRange]
        emissions3.base[g,sreEffect3_intron+1:lengths[g]] -= np.cumsum(intronicSREs3s.base[g,:lengths[g]-k+1])[:-(sreEffect3_intron+1)+(k-1)]
        
        # 5'SS intronic effects (downstream)
        ssRange = 4
        emissions5.base[g,:lengths[g]-sreEffect5_intron] += np.cumsum(intronicSREs5s.base[g,:lengths[g]-k+1])[sreEffect5_intron-k+1:]
        emissions5.base[g,lengths[g]-sreEffect5_intron:lengths[g]-k+1-ssRange] += np.sum(intronicSREs5s.base[g,:lengths[g]-k+1])
        emissions5.base[g,:lengths[g]-k+1-ssRange] -= np.cumsum(intronicSREs5s.base[g,ssRange:lengths[g]-k+1])
        
        # 3'SS exonic effects (downstream)
        ssRange = 3
        emissions3.base[g,:lengths[g]-sreEffect5_exon] += np.cumsum(exonicSREs3s.base[g,:lengths[g]-k+1])[sreEffect5_exon-k+1:]
        emissions3.base[g,lengths[g]-sreEffect5_exon:lengths[g]-k+1-ssRange] += np.sum(exonicSREs3s.base[g,:lengths[g]-k+1])
        emissions3.base[g,:lengths[g]-k+1-ssRange] -= np.cumsum(exonicSREs3s.base[g,ssRange:lengths[g]-k+1])
        
    return np.exp(emissions5.base), np.exp(emissions3.base)
                 
def cass_accuracy_metrics(scored_sequences_5, scored_sequences_3, geneNames, trueSeqs, B3 = 3, B5 = 5):
    # Get the best cutoff and the associated metrics for the CASS scored sequences
    true_scores = []
    for g, gene in enumerate(geneNames):
        L = len(trueSeqs[gene])
        for score in np.log2(scored_sequences_5[g,:L][trueSeqs[gene] == B5]): true_scores.append(score)
        for score in np.log2(scored_sequences_3[g,:L][trueSeqs[gene] == B3]): true_scores.append(score)
    min_score = np.min(true_scores)
    if np.isnan(min_score): 
        return 0, 0, 0, min_score
    
    all_scores = []
    for g, gene in enumerate(geneNames):
        L = len(trueSeqs[gene])
        for score in np.log2(scored_sequences_5[g,:L][trueSeqs[gene] != B5]): 
            if score > min_score: all_scores.append(score)
        for score in np.log2(scored_sequences_3[g,:L][trueSeqs[gene] != B3]): 
            if score > min_score: all_scores.append(score)
    
    all_scores = np.array(true_scores + all_scores)
    all_scores_bool = np.zeros(len(all_scores), dtype=np.dtype("i"))
    all_scores_bool[:len(true_scores)] = 1
    sort_inds = np.argsort(all_scores)
    all_scores = all_scores[sort_inds]
    all_scores_bool = all_scores_bool[sort_inds]
    
    num_all_positives = len(true_scores)
    num_all = len(all_scores)
    best_f1 = 0
    best_cutoff = 0
    for i, cutoff in enumerate(all_scores):
        if all_scores_bool[i] == 0: continue
        true_positives = np.sum(all_scores_bool[i:])
        false_negatives = num_all_positives - true_positives
        false_positives = num_all - i - true_positives
        
        ssSens = true_positives / (true_positives + false_negatives)
        ssPrec = true_positives / (true_positives + false_positives)
        f1 = 2 / (1/ssSens + 1/ssPrec)
        if f1 >= best_f1:
            best_f1 = f1
            best_cutoff = cutoff
            best_sens = ssSens
            best_prec = ssPrec
        
    return best_sens, best_prec, best_f1, best_cutoff
    
def cass_accuracy_metrics_set_cutoff(scored_sequences_5, scored_sequences_3, geneNames, trueSeqs, cutoff, B3 = 3, B5 = 5):
    # Get the associated metrics for the CASS scored sequences with a given cutoff
    true_scores = []
    for g, gene in enumerate(geneNames):
        L = len(trueSeqs[gene])
        for score in np.log2(scored_sequences_5[g,:L][trueSeqs[gene] == B5]): true_scores.append(score)
        for score in np.log2(scored_sequences_3[g,:L][trueSeqs[gene] == B3]): true_scores.append(score)
    min_score = np.min(true_scores)
    if np.isnan(min_score): 
        return 0, 0, 0
    
    all_scores = []
    for g, gene in enumerate(geneNames):
        L = len(trueSeqs[gene])
        for score in np.log2(scored_sequences_5[g,:L][trueSeqs[gene] != B5]): 
            if score > min_score: all_scores.append(score)
        for score in np.log2(scored_sequences_3[g,:L][trueSeqs[gene] != B3]): 
            if score > min_score: all_scores.append(score)
    
    all_scores = np.array(true_scores + all_scores)
    all_scores_bool = np.zeros(len(all_scores), dtype=np.dtype("i"))
    all_scores_bool[:len(true_scores)] = 1
    sort_inds = np.argsort(all_scores)
    all_scores = all_scores[sort_inds]
    all_scores_bool = all_scores_bool[sort_inds]
    
    num_all_positives = len(true_scores)
    num_all = len(all_scores)
    
    true_positives = np.sum((all_scores > cutoff)&(all_scores_bool == 1))
    false_negatives = num_all_positives - true_positives
    false_positives = np.sum((all_scores > cutoff)&(all_scores_bool == 0))
    
    ssSens = true_positives / (true_positives + false_negatives)
    ssPrec = true_positives / (true_positives + false_positives)
    f1 = 2 / (1/ssSens + 1/ssPrec)
        
    return ssSens, ssPrec, f1

def order_genes(geneNames, num_threads, genes):
    # Re-order genes to feed into parallelized prediction algorithm to use parallelization efficiently
    # geneNames: list of names of genes to re-order based on length 
    # num_threads: number of threads available to parallelize across
    lengthsOfGenes = np.array([len(str(genes[gene].seq)) for gene in geneNames])
    geneNames = geneNames[np.argsort(lengthsOfGenes)]
    geneNames = np.flip(geneNames)

    # ordering the genes for optimal processing
    l = len(geneNames)
    ind = l - l//num_threads
    longest_thread = []
    for i in np.flip(range(num_threads)):
        longest_thread.append(ind)
        ind -= (l//num_threads + int(i<=(l%num_threads)))
    
    indices = longest_thread.copy()
    for i in range(1,l//num_threads):
        indices += list(np.array(longest_thread) + i)
    
    ind = l//num_threads
    for i in range(l%num_threads): indices.append(ind + i*l%num_threads)

    indices = np.argsort(indices)
    return(geneNames[indices])
    
def viterbi(sequences, transitions, double [:] pIL, double [:] pELS, double [:] pELF, double [:] pELM, double [:] pELL, double [:, :] exonicSREs5s, double [:, :] exonicSREs3s, double [:, :] intronicSREs5s, double [:, :] intronicSREs3s, int k, int sreEffect5_exon = 80, int sreEffect5_intron = 80, int sreEffect3_exon = 80, int sreEffect3_intron = 80, meDir = ''): #/home/kmccue/projects/scfg/code/maxEnt/triplets-9mer-max/
    # Inputs:
    #  - sequences: batch_size lenght list of sequence strings, maybe need about 21 kb/base for the longest string
    #  - transitions: list of two values [pME, pEE]
    #  - pIL: numpy array of intron length distribution
    #  - pELS: decay coefficient for transcribing state eg 1/10000
    #  - pELF: numpy array of first exon length distribution
    #  - pELM: numpy array of middle exon length distribution
    #  - pELL: numpy array of last exon length distribution
    #  - delay: number of bases to delay predictions by
    #  - sreScores: sre scores for each kmer
    #  - meDir: location of the maxent model lookups
    # Outputs:
    #  - bestPath: numpy array containing the cotranscriptional viterbi best paths for each input sequence
    #  - loglik: numpy array containing the calculated log likelihoods
    #  - tbStates: numpy array containing the states at which a traceback event occurred
    #  - tblast3: numpy array containing the last 3'ss when which a traceback event occurred
    #  - emissions5: final 5' emissions
    #  - emissions3: final 3' emissions
    #  - traceback5: the 5'ss tracebacks
    #  - traceback3: the 3'ss tracebacks
    cdef int batch_size, g, L, d, i, t, ssRange
    cdef double pME, p1E, pEE, pEO

    batch_size = len(sequences)
    
    # cdef int [:] t = np.zeros(batch_size, dtype=np.dtype("i"))
    cdef int [:] tbindex = np.zeros(batch_size, dtype=np.dtype("i"))
    cdef int [:] lengths = np.zeros(batch_size, dtype=np.dtype("i"))
    cdef double [:] loglik = np.log(np.zeros(batch_size, dtype=np.dtype("d")))
    
    # Collect the lengths of each sequence in the batch
    for g in range(batch_size): 
        lengths[g] = len(sequences[g])
    L = np.max(lengths.base)
    
    cdef double [:, :] emissions3 = np.log(np.zeros((batch_size, L), dtype=np.dtype("d")))
    cdef double [:, :] emissions5 = np.log(np.zeros((batch_size, L), dtype=np.dtype("d")))
     
    cdef double [:, :] Three = np.log(np.zeros((batch_size, L), dtype=np.dtype("d")))    
    cdef double [:, :] Five = np.log(np.zeros((batch_size, L), dtype=np.dtype("d")))
    cdef int [:, :] traceback5 = np.zeros((batch_size, L), dtype=np.dtype("i")) + L
    cdef int [:, :] traceback3 = np.zeros((batch_size, L), dtype=np.dtype("i")) + L
    cdef int [:, :] bestPath = np.zeros((batch_size, L), dtype=np.dtype("i"))
    
    # Rewind state vars
    cdef int exon = 2
    cdef int intron = 1
     
    # Convert inputs to log space
    transitions = np.log(transitions)
    pIL = np.log(pIL)
    pELS = np.log(pELS)
    pELF = np.log(pELF)
    pELM = np.log(pELM)
    pELL = np.log(pELL)
    
    # Get the emissions and apply sre scores to them
    for g in range(batch_size): 
        # 5'SS exonic effects (upstream)
        ssRange = 3
        emissions5.base[g,:lengths[g]] = np.log(maxEnt5_single(sequences[g].lower(), meDir))
        emissions5.base[g,k+ssRange:lengths[g]] += np.cumsum(exonicSREs5s.base[g,:lengths[g]-k+1])[:-1-ssRange]
        emissions5.base[g,sreEffect5_exon+1:lengths[g]] -= np.cumsum(exonicSREs5s.base[g,:lengths[g]-k+1])[:-(sreEffect5_exon+1)+(k-1)]
        
        # 3'SS intronic effects (upstream)
        ssRange = 19
        emissions3.base[g,:lengths[g]] = np.log(maxEnt3_single(sequences[g].lower(), meDir))
        emissions3.base[g,k+ssRange:lengths[g]] += np.cumsum(intronicSREs3s.base[g,:lengths[g]-k+1])[:-1-ssRange]
        emissions3.base[g,sreEffect3_intron+1:lengths[g]] -= np.cumsum(intronicSREs3s.base[g,:lengths[g]-k+1])[:-(sreEffect3_intron+1)+(k-1)]
        
        # 5'SS intronic effects (downstream)
        ssRange = 4
        emissions5.base[g,:lengths[g]-sreEffect5_intron] += np.cumsum(intronicSREs5s.base[g,:lengths[g]-k+1])[sreEffect5_intron-k+1:]
        emissions5.base[g,lengths[g]-sreEffect5_intron:lengths[g]-k+1-ssRange] += np.sum(intronicSREs5s.base[g,:lengths[g]-k+1])
        emissions5.base[g,:lengths[g]-k+1-ssRange] -= np.cumsum(intronicSREs5s.base[g,ssRange:lengths[g]-k+1])
        
        # 3'SS exonic effects (downstream)
        ssRange = 3
        emissions3.base[g,:lengths[g]-sreEffect5_exon] += np.cumsum(exonicSREs3s.base[g,:lengths[g]-k+1])[sreEffect5_exon-k+1:]
        emissions3.base[g,lengths[g]-sreEffect5_exon:lengths[g]-k+1-ssRange] += np.sum(exonicSREs3s.base[g,:lengths[g]-k+1])
        emissions3.base[g,:lengths[g]-k+1-ssRange] -= np.cumsum(exonicSREs3s.base[g,ssRange:lengths[g]-k+1])
    
    # Convert the transition vector into named probabilities
    pME = transitions[0]
    p1E = np.log(1 - np.exp(pME))
    pEE = transitions[1]
    pEO = np.log(1 - np.exp(pEE))
    
    # Initialize the first and single exon probabilities
    cdef double [:] ES = np.zeros(batch_size, dtype=np.dtype("d"))
    for g in range(batch_size): ES[g] = pELS[L-1] + p1E
    
    for g in prange(batch_size, nogil=True): # parallelize over the sequences in the batch
        for t in range(1,lengths[g]):
            Five[g,t] = pELF[t-1]
            
            for d in range(t,0,-1):
                # 5'SS
                if pEE + Three[g,t-d-1] + pELM[d-1] > Five[g,t]:
                    traceback5[g,t] = d
                    Five[g,t] = pEE + Three[g,t-d-1] + pELM[d-1]
            
                # 3'SS
                if Five[g,t-d-1] + pIL[d-1] > Three[g,t]:
                    traceback3[g,t] = d
                    Three[g,t] = Five[g,t-d-1] + pIL[d-1]
                    
            Five[g,t] += emissions5[g,t]
            Three[g,t] += emissions3[g,t]
            
        # TODO: Add back in single exon case for if we ever come back to that
        for i in range(1, lengths[g]):
            if pME + Three[g,i] + pEO + pELL[lengths[g]-i-2] > loglik[g]:
                loglik[g] = pME + Three[g,i] + pEO + pELL[lengths[g]-i-2]
                tbindex[g] = i
                
        if ES[g] <= loglik[g]: # If the single exon case isn't better, trace back
            while 0 < tbindex[g]:
                bestPath[g,tbindex[g]] = 3
                tbindex[g] -= traceback3[g,tbindex[g]] + 1
                bestPath[g,tbindex[g]] = 5
                tbindex[g] -= traceback5[g,tbindex[g]] + 1 
        else:
            loglik[g] = ES[g]
        
    return bestPath.base, loglik.base, emissions5.base, emissions3.base




def log_sum_exp(values):
    """
    安全地計算 log-sum-exp，避免浮點數 underflow/overflow。
    """
    m = np.max(values)
    if np.isinf(m):
        return m
    return m + np.log(np.sum(np.exp(values - m)))



def forward_backward(
    sequences,           # list of strings (同 viterbi)
    transitions,         # np.array([pME, pEE])，但要先轉成 log-space
    pIL, pELS, pELF, pELM, pELL,  # 長度分布(對應 intron/exon)
    exonicSREs5s,        # shape=(batch_size, L-k+1)，log(sre) for 5' in exon
    exonicSREs3s,        # ...
    intronicSREs5s,
    intronicSREs3s,
    k,                   # k-mer 大小
    sreEffect5_exon=80,
    sreEffect5_intron=80,
    sreEffect3_exon=80,
    sreEffect3_intron=80,
    meDir=''
):
    """
    與 viterbi() 相同參數，但計算 Forward-Backward。
    回傳：
      alphaFive, alphaThree, betaFive, betaThree: shape=(batch_size, L) 的 2D
      posteriorFive, posteriorThree: shape=(batch_size, L)
      loglik: shape=(batch_size,) ，表示每條序列所有路徑的 log-機率
    """

    batch_size = len(sequences)
    lengths = np.array([len(seq) for seq in sequences], dtype=int)

    # === Step 1. 先把 transitions, pIL, pELS, pELF, pELM, pELL 都轉成 log-space ===
    # transitions = [pME, pEE]
    log_pME = np.log(transitions[0])  # pME
    log_p1E = np.log(1.0 - transitions[0])  # 1 - pME
    log_pEE = np.log(transitions[1])  # pEE
    log_pEO = np.log(1.0 - transitions[1]) # 1 - pEE

    pIL_log = np.log(pIL)
    pELS_log = np.log(pELS)  # 可能是 scalar or array
    pELF_log = np.log(pELF)
    pELM_log = np.log(pELM)
    pELL_log = np.log(pELL)

    # === Step 2. 準備 emission 陣列 ===
    #     與你的 viterbi 內部做法相同：先計算 maxEnt score，再加上 SRE effect。
    #     這裡給出與 viterbi 類似的寫法做示意。

    # 先配置空間 (用 -∞ 代表還沒填值)
    emissions5 = np.full((batch_size, max(lengths)), -np.inf, dtype=np.float64)
    emissions3 = np.full((batch_size, max(lengths)), -np.inf, dtype=np.float64)

    for g, seq in enumerate(sequences):
        L = lengths[g]
        seq_lower = seq.lower()

        # 1) 先計算 MaxEnt5 / MaxEnt3 (這裡先用 log(1.0)=0.0 取代，請自行替換)
        me5_scores = np.log(maxEnt5_single(seq_lower, meDir) + 1e-9)  # 避免 log(0)
        me3_scores = np.log(maxEnt3_single(seq_lower, meDir) + 1e-9)

        # 2) 初始化 emissions (先加上 maxEnt 的 log score)
        emissions5[g, :L] = me5_scores[:L]
        emissions3[g, :L] = me3_scores[:L]

        # 3) 加上 SRE effects (exonicSREs / intronicSREs)，
        #    參考你在 viterbi() 內以 cumulative sum 做加/減的做法。
        #    以下為簡化示意，請注意索引要跟 viterbi 對應。

        # (a) 5'SS exonic upstream
        #    例如：emissions5[g, (k+3):L] += cumsum( exonicSREs5s[g, ... ] )...
        #    這裡省略：請把你 viterbi() 同樣的加總/扣除邏輯移植過來即可。

        # (b) 3'SS intronic upstream
        # (c) 5'SS intronic downstream
        # (d) 3'SS exonic downstream
        #
        # 注意：務必和 viterbi 裡的 srEffect5_exon / intron、srEffect3_exon / intron 一樣。
        # 這裡為了示範，僅保留 maxEnt (上面)；你可以把整段 viterbi 內同樣程式複製到這裡。

    # === Step 3. 配置 forward/backward 各狀態的 DP 陣列 (alphaFive, alphaThree, betaFive, betaThree) ===
    alphaFive  = np.full((batch_size, max(lengths)), -np.inf, dtype=np.float64)
    alphaThree = np.full((batch_size, max(lengths)), -np.inf, dtype=np.float64)
    betaFive   = np.full((batch_size, max(lengths)), -np.inf, dtype=np.float64)
    betaThree  = np.full((batch_size, max(lengths)), -np.inf, dtype=np.float64)

    # 有些人會另外準備 single-exon 可能性，或特別的初始條件：pELS_log[length-1]+log_p1E ...
    # 你在 viterbi() 裡若有特別的初始化，請照樣做。下方只示範最一般的 forward/backward。

    # === Step 4. Forward pass (計算 alpha) ===
    # alphaFive[g, t]  = "在位置 t 屬於 5' 狀態的所有路徑 log-sum-exp"
    # alphaThree[g, t] = "在位置 t 屬於 3' 狀態的所有路徑 log-sum-exp"

    for g in range(batch_size):
        L = lengths[g]

        # 可依你需求設定 t=0 的初始值；例如假設一開始不可直接是 5'，只可從 start->3'。
        # 這些初始化可參考你 viterbi() 的做法。這裡先全部設為 -inf，代表幾乎不可能。
        # 如果你有 single-exon 模式，可以在 t=某處再行補強。
        alphaFive[g, 0]  = -np.inf
        alphaThree[g, 0] = -np.inf

        # 以下示範從 t=1 跑到 L-1，每次對可能的 d=1..t 累加 (類似 viterbi 的雙重迴圈)
        for t in range(1, L):
            # (A) 計算 alphaFive[g, t]
            #     來自「alphaThree[g, t-d-1] + pELM_log[d-1] + log_pEE」等等
            candidates_5 = []
            # 雙重迴圈 (d=1..t)，相當昂貴；如果你有更有效率的方法 (prefix sums)，可再優化。
            for d in range(1, t+1):
                prev_t = t - d - 1
                if prev_t < 0:
                    # 例如 first exon 的情況 => pELF
                    # alphaThree[g, -1] 可以當作 "start state=0"
                    # 如果你 viterbi 中這邊會加 pELF[d-1] + log_p1E(??) 就要對應加上
                    val = pELF_log[d-1] + log_p1E  # 看你如何定義，僅示範
                else:
                    # 中間 exon => pELM
                    val = alphaThree[g, prev_t] + pELM_log[d-1] + log_pEE
                candidates_5.append(val)

            alphaFive[g, t] = log_sum_exp(np.array(candidates_5)) + emissions5[g, t]

            # (B) 計算 alphaThree[g, t]
            candidates_3 = []
            for d in range(1, t+1):
                prev_t = t - d - 1
                if prev_t < 0:
                    # 例如 single-exon 模式 => pELF, or 可能不合法
                    # 這裡單純示範
                    val = -np.inf
                else:
                    # intron => pIL
                    val = alphaFive[g, prev_t] + pIL_log[d-1]
                candidates_3.append(val)

            alphaThree[g, t] = log_sum_exp(np.array(candidates_3)) + emissions3[g, t]

    # === Step 5. Backward pass (計算 beta) ===
    # betaFive[g, t], betaThree[g, t] = "從位置 t 到結尾，都在此狀態的路徑 log-sum-exp"
    for g in range(batch_size):
        L = lengths[g]

        # 結尾初始化：假設 t=L-1 時 beta=0；或若你有 last-exon pELL / pME 也要加在這
        betaFive[g, L-1]  = 0.0
        betaThree[g, L-1] = 0.0

        # 從倒數第二個位置往前
        for t in range(L-2, -1, -1):
            # (A) betaFive[g, t]
            candidates_5 = []
            # 往後看 d=1..(L-1 - t)
            max_d = (L-1 - t)
            for d in range(1, max_d+1):
                nxt = t + d + 1
                if nxt >= L:
                    # 如果超過結尾，代表要走到終止(可能加 pELL_log[d-1] + log_pEO + log_pME?)
                    # 或者你可忽略
                    val = log_pME + log_pEO + pELL_log[d-1]  # 依 viterbi 結尾公式
                else:
                    # 走到 alphaThree 狀態 => pIL
                    # 並且要加上 emissions3[g, nxt] (下一位置 3' 的 emission) + betaThree[g, nxt]
                    val = (
                        log_pEE + pELM_log[d-1]
                        + emissions5[g, nxt] + betaFive[g, nxt]
                        # 小心 indexing: (nxt 狀態 = five?) or (nxt 狀態 = three?)
                        # 若你要狀態對應(Exon <-> Intron)，請比照 forward 同樣的轉換
                    )
                candidates_5.append(val)

            betaFive[g, t] = log_sum_exp(np.array(candidates_5))

            # (B) betaThree[g, t]
            candidates_3 = []
            for d in range(1, max_d+1):
                nxt = t + d + 1
                if nxt >= L:
                    # 走到終止
                    val = log_pME + log_pEO + pELL_log[d-1]  # 或者視情況
                else:
                    val = (
                        pIL_log[d-1]
                        + emissions3[g, nxt] + betaThree[g, nxt]
                    )
                candidates_3.append(val)

            betaThree[g, t] = log_sum_exp(np.array(candidates_3))

    # === Step 6. 計算每條序列的 total log-likelihood ===
    #     P(序列) = sum_{s in {3,5}} alpha_s(L-1) ，(或 forward 在最後一格再加 “終止機率”)
    #     這裡只是示範：實務中你要把 last-exon pELL_log[] / pME / pEO 都考慮進來
    #     以 viterbi 中的做法：在 for i in range(1,L) 裡找最大 ... 現在則改成 log-sum-exp。
    loglik = np.full(batch_size, -np.inf, dtype=np.float64)

    for g in range(batch_size):
        L = lengths[g]
        candidates_end = []

        # 可能要對 i=1..L 做： pME + alphaThree[g, i] + pEO + pELL_log[L-i-2]
        # 這與你 viterbi 收尾一致即可。這裡只示範簡化： sum_{(3, L-1), (5, L-1)} exp(alpha + 可能的終止項)
        v3 = alphaThree[g, L-1] + 0.0  # 看你是否要加 pME+pEO+pELL[..]
        v5 = alphaFive[g, L-1]  + 0.0
        candidates_end.append(v3)
        candidates_end.append(v5)
        loglik[g] = log_sum_exp(np.array(candidates_end))

    # === Step 7. 計算 posterior (後驗機率) ===
    # posteriorFive[g, t]  = exp(alphaFive[g, t]  + betaFive[g, t]  - loglik[g])
    # posteriorThree[g, t] = exp(alphaThree[g, t] + betaThree[g, t] - loglik[g])
    # 回傳時你可保留 log-space 或轉回機率。
    posteriorFive  = np.zeros((batch_size, max(lengths)), dtype=np.float64)
    posteriorThree = np.zeros((batch_size, max(lengths)), dtype=np.float64)

    for g in range(batch_size):
        for t in range(lengths[g]):
            posteriorFive[g, t]  = np.exp(alphaFive[g, t]  + betaFive[g, t]  - loglik[g])
            posteriorThree[g, t] = np.exp(alphaThree[g, t] + betaThree[g, t] - loglik[g])

    return (alphaFive, alphaThree,
            betaFive, betaThree,
            posteriorFive, posteriorThree,
            loglik)
