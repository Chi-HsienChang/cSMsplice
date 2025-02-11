import numpy as np
from sklearn.neighbors import KernelDensity
from awkde import GaussianKDE
from ipdb import set_trace

def baseToInt(base):
    if base == 'a': return 0
    elif base == 'c': return 1
    elif base == 'g': return 2
    elif base == 't': return 3
    else:
        print("nonstandard base encountered:", base)
        return -1
        
def hashSequence(seq):
    sum = 0 
    l = len(seq)
    for i in range(l):
        sum += (4**(l-i-1))*baseToInt(seq[i])
    return sum

def intToBase(i):
    if i == 0: return 'a'
    elif i == 1: return 'c'
    elif i == 2: return 'g'
    elif i == 3: return 't'
    else: 
        print("nonbase integer encountered:", i)
        return ''
    
def unhashSequence(num, l):
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
    
def maxEnt5_single(seq, dir):
    # Get all the 5'SS maxent scores for the input sequence
    prob = np.load(dir + 'maxEnt5_prob.npy')
    prob0 = np.load(dir + 'maxEnt5_prob0.npy')
    
    seq = seq.lower()
    # set_trace()
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
    
# def maxEnt5_single(seq, dir):
#     # Get all the 5'SS maxent scores for the input sequence
#     prob = np.load(dir + 'maxEnt5_prob.npy')
#     prob0 = np.load(dir + 'maxEnt5_prob0.npy')
    
#     seq = seq.lower()
#     sequence5 = np.array([hashSequence(seq[i:i+9]) for i in range(len(seq)-9+1)])
#     scores = np.log2(np.zeros(len(seq)))
#     scores[3:-5] = np.log2(prob[sequence5]) - np.log2(prob0[sequence5])
#     return np.exp2(scores)

def maxEnt3_single(seq, dir):
    # Get all the 3'SS maxent scores for the input sequence
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

def sreScores_single(seq, sreScores, kmer = 6):
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
    
def score_sequences(sequences, exonicSREs5s, exonicSREs3s, intronicSREs5s, intronicSREs3s, k = 6, sreEffect5_exon = 80, sreEffect5_intron = 80, sreEffect3_exon = 80, sreEffect3_intron = 80, meDir = ''): 
    # Get the CASS scores for the input sequences
    
    batch_size = len(sequences)
    lengths = np.zeros(batch_size, dtype=np.dtype("i"))
    
    # Collect the lengths of each sequence in the batch
    for g in range(batch_size): 
        lengths[g] = len(sequences[g])
    L = np.max(lengths)
    
    emissions3 = np.log(np.zeros((batch_size, L), dtype=np.dtype("d")))
    emissions5 = np.log(np.zeros((batch_size, L), dtype=np.dtype("d")))
    
    # Get the emissions and apply sre scores to them
    for g in range(batch_size): 
        # 5'SS exonic effects (upstream)
        ssRange = 3
        emissions5[g,:lengths[g]] = np.log(maxEnt5_single(sequences[g].lower(), meDir)) # np.log >>> np.log2
        emissions5[g,k+ssRange:lengths[g]] += np.cumsum(exonicSREs5s[g,:lengths[g]-k+1])[:-1-ssRange]
        emissions5[g,sreEffect5_exon+1:lengths[g]] -= np.cumsum(exonicSREs5s[g,:lengths[g]-k+1])[:-(sreEffect5_exon+1)+(k-1)]
        
        # 3'SS intronic effects (upstream)
        ssRange = 19
        emissions3[g,:lengths[g]] = np.log(maxEnt3_single(sequences[g].lower(), meDir))
        emissions3[g,k+ssRange:lengths[g]] += np.cumsum(intronicSREs3s[g,:lengths[g]-k+1])[:-1-ssRange]
        emissions3[g,sreEffect3_intron+1:lengths[g]] -= np.cumsum(intronicSREs3s[g,:lengths[g]-k+1])[:-(sreEffect3_intron+1)+(k-1)]
        
        # 5'SS intronic effects (downstream)
        ssRange = 4
        emissions5[g,:lengths[g]-sreEffect5_intron] += np.cumsum(intronicSREs5s[g,:lengths[g]-k+1])[sreEffect5_intron-k+1:]
        emissions5[g,lengths[g]-sreEffect5_intron:lengths[g]-k+1-ssRange] += np.sum(intronicSREs5s[g,:lengths[g]-k+1])
        emissions5[g,:lengths[g]-k+1-ssRange] -= np.cumsum(intronicSREs5s[g,ssRange:lengths[g]-k+1])
        
        # 3'SS exonic effects (downstream)
        ssRange = 3
        emissions3[g,:lengths[g]-sreEffect5_exon] += np.cumsum(exonicSREs3s[g,:lengths[g]-k+1])[sreEffect5_exon-k+1:]
        emissions3[g,lengths[g]-sreEffect5_exon:lengths[g]-k+1-ssRange] += np.sum(exonicSREs3s[g,:lengths[g]-k+1])
        emissions3[g,:lengths[g]-k+1-ssRange] -= np.cumsum(exonicSREs3s[g,ssRange:lengths[g]-k+1])
        
    return np.exp(emissions5), np.exp(emissions3)

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



def viterbi(sequences, transitions, pIL, pELS, pELF, pELM, pELL, exonicSREs5s, exonicSREs3s, intronicSREs5s, intronicSREs3s, k, sreEffect5_exon, sreEffect5_intron, sreEffect3_exon, sreEffect3_intron, meDir = ''): 
    # Get the best parses of all the input sequences
    batch_size = len(sequences) # 基因數量 = 2
    tbindex = np.zeros(batch_size, dtype=np.dtype("i")) # array([0, 0], dtype=int32)
    lengths = np.zeros(batch_size, dtype=np.dtype("i")) # array([0, 0], dtype=int32)
    loglik = np.log(np.zeros(batch_size, dtype=np.dtype("d"))) # array([-inf, -inf])
    
    # Collect the lengths of each sequence in the batch
    for g in range(batch_size): 
        lengths[g] = len(sequences[g])
    L = np.max(lengths)
    
    emissions3 = np.log(np.zeros((batch_size, L), dtype=np.dtype("d")))
    emissions5 = np.log(np.zeros((batch_size, L), dtype=np.dtype("d"))) 
    Three = np.log(np.zeros((batch_size, L), dtype=np.dtype("d")))    
    Five = np.log(np.zeros((batch_size, L), dtype=np.dtype("d")))
    traceback5 = np.zeros((batch_size, L), dtype=np.dtype("i")) + L
    traceback3 = np.zeros((batch_size, L), dtype=np.dtype("i")) + L
    bestPath = np.zeros((batch_size, L), dtype=np.dtype("i"))
    
    # Rewind state vars
    exon = 2
    intron = 1
     
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
        emissions5[g,:lengths[g]] = np.log(maxEnt5_single(sequences[g].lower(), meDir))
        emissions5[g,k+ssRange:lengths[g]] += np.cumsum(exonicSREs5s[g,:lengths[g]-k+1])[:-1-ssRange]
        emissions5[g,sreEffect5_exon+1:lengths[g]] -= np.cumsum(exonicSREs5s[g,:lengths[g]-k+1])[:-(sreEffect5_exon+1)+(k-1)]
        
        # 3'SS intronic effects (upstream)
        ssRange = 19
        emissions3[g,:lengths[g]] = np.log(maxEnt3_single(sequences[g].lower(), meDir))
        emissions3[g,k+ssRange:lengths[g]] += np.cumsum(intronicSREs3s[g,:lengths[g]-k+1])[:-1-ssRange]
        emissions3[g,sreEffect3_intron+1:lengths[g]] -= np.cumsum(intronicSREs3s[g,:lengths[g]-k+1])[:-(sreEffect3_intron+1)+(k-1)]
        
        # 5'SS intronic effects (downstream)
        ssRange = 4
        emissions5[g,:lengths[g]-sreEffect5_intron] += np.cumsum(intronicSREs5s[g,:lengths[g]-k+1])[sreEffect5_intron-k+1:]
        emissions5[g,lengths[g]-sreEffect5_intron:lengths[g]-k+1-ssRange] += np.sum(intronicSREs5s[g,:lengths[g]-k+1])
        emissions5[g,:lengths[g]-k+1-ssRange] -= np.cumsum(intronicSREs5s[g,ssRange:lengths[g]-k+1])
        
        # 3'SS exonic effects (downstream)
        ssRange = 3
        emissions3[g,:lengths[g]-sreEffect5_exon] += np.cumsum(exonicSREs3s[g,:lengths[g]-k+1])[sreEffect5_exon-k+1:]
        emissions3[g,lengths[g]-sreEffect5_exon:lengths[g]-k+1-ssRange] += np.sum(exonicSREs3s[g,:lengths[g]-k+1])
        emissions3[g,:lengths[g]-k+1-ssRange] -= np.cumsum(exonicSREs3s[g,ssRange:lengths[g]-k+1])
    
    # Convert the transition vector into named probabilities
    pME = transitions[0]
    p1E = np.log(1 - np.exp(pME))
    pEE = transitions[1]
    pEO = np.log(1 - np.exp(pEE))
    
    # Initialize the first and single exon probabilities
    ES = np.zeros(batch_size, dtype=np.dtype("d"))
    for g in range(batch_size): ES[g] = pELS[L-1] + p1E
    
    for g in range(batch_size): # loop the sequences in the batch
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
        
    return bestPath, loglik, emissions5, emissions3








# import numpy as np

# 假設你的環境裡已經有這兩個函式 (原程式出現過)
# def maxEnt5_single(seq, meDir=''): ...
# def maxEnt3_single(seq, meDir=''): ...

# def viterbi_second_best(
#     sequences, 
#     transitions, 
#     pIL, pELS, pELF, pELM, pELL, 
#     exonicSREs5s, exonicSREs3s, 
#     intronicSREs5s, intronicSREs3s, 
#     k, sreEffect5_exon, sreEffect5_intron, 
#     sreEffect3_exon, sreEffect3_intron, 
#     meDir=''
# ):
#     """
#     與原先的 viterbi() 大致相同，但額外保留「次佳」(second-best) 分數與路徑。
    
#     Returns:
#     --------
#     bestPath         : shape (batch_size, max_length)
#     secondBestPath   : shape (batch_size, max_length)
#     bestLogLik       : shape (batch_size,)，各序列的最佳對數似然
#     secondBestLogLik : shape (batch_size,)，各序列的第二佳對數似然
#     emissions5, emissions3
#     """
    
#     # --------------------
#     # 0) 前置初始化
#     # --------------------
#     batch_size = len(sequences)
#     tbindex = np.zeros(batch_size, dtype=np.int32)
#     lengths = np.zeros(batch_size, dtype=np.int32)
    
#     # 收集每條序列長度
#     for g in range(batch_size):
#         lengths[g] = len(sequences[g])
#     L = np.max(lengths) if batch_size > 0 else 0
    
#     # 最終對數似然 (最佳 / 次佳)
#     bestLogLik = np.full(batch_size, -np.inf, dtype=float)
#     secondBestLogLik = np.full(batch_size, -np.inf, dtype=float)
    
#     # 初始化發射分數 (log space)
#     # 注意: 用 -inf 來填滿「超過序列長度」的區域
#     emissions5 = np.full((batch_size, L), -np.inf, dtype=float)
#     emissions3 = np.full((batch_size, L), -np.inf, dtype=float)
    
#     Three  = np.full((batch_size, L), -np.inf, dtype=float)
#     Five   = np.full((batch_size, L), -np.inf, dtype=float)
#     # 新增 secondBest DP 陣列
#     secondThree = np.full((batch_size, L), -np.inf, dtype=float)
#     secondFive  = np.full((batch_size, L), -np.inf, dtype=float)
    
#     # 回溯用
#     traceback5  = np.full((batch_size, L), L, dtype=int)
#     traceback3  = np.full((batch_size, L), L, dtype=int)
#     # 次佳回溯
#     secondTraceback5 = np.full((batch_size, L), L, dtype=int)
#     secondTraceback3 = np.full((batch_size, L), L, dtype=int)
    
#     # 最終路徑標記
#     bestPath = np.zeros((batch_size, L), dtype=int)
#     secondBestPath = np.zeros((batch_size, L), dtype=int)
    
#     # 狀態常數 (若需要，可使用)
#     exon = 2
#     intron = 1
    
#     # --------------------
#     # 1) Convert inputs to log space
#     # --------------------
#     transitions = np.log(transitions)
#     pIL = np.log(pIL)
#     pELS = np.log(pELS)
#     pELF = np.log(pELF)
#     pELM = np.log(pELM)
#     pELL = np.log(pELL)
    
#     # --------------------
#     # 2) 計算初始發射分數 + SRE 效果
#     # --------------------
#     for g in range(batch_size):
#         seq_len = lengths[g]
#         if seq_len == 0:
#             continue
        
#         # 先計算 5' / 3' 的最大熵 (基礎發射分)
#         # 注意: maxEntX_single(...) 須回傳「單一分數」；若回傳陣列，需自行處理
#         for t in range(seq_len):
#             emissions5[g, t] = np.log(maxEnt5_single(sequences[g].lower(), meDir))
#             emissions3[g, t] = np.log(maxEnt3_single(sequences[g].lower(), meDir))
        
#         # 以下模擬原程式把 SRE 效果套加到 emissionsX 中
#         # -------------- 5'SS exonic effects (upstream) --------------
#         ssRange = 3
#         if seq_len > k + ssRange:
#             emissions5[g, k+ssRange:seq_len] += np.cumsum(exonicSREs5s[g, :seq_len - k + 1])[:-1 - ssRange]
#         if seq_len > sreEffect5_exon + 1:
#             emissions5[g, sreEffect5_exon+1:seq_len] -= np.cumsum(exonicSREs5s[g, :seq_len - k + 1])[:-(sreEffect5_exon+1)+(k-1)]
        
#         # -------------- 3'SS intronic effects (upstream) --------------
#         ssRange = 19
#         if seq_len > k + ssRange:
#             emissions3[g, k+ssRange:seq_len] += np.cumsum(intronicSREs3s[g, :seq_len - k + 1])[:-1 - ssRange]
#         if seq_len > sreEffect3_intron + 1:
#             emissions3[g, sreEffect3_intron+1:seq_len] -= np.cumsum(intronicSREs3s[g, :seq_len - k + 1])[:-(sreEffect3_intron+1)+(k-1)]
        
#         # -------------- 5'SS intronic effects (downstream) --------------
#         ssRange = 4
#         if (seq_len - sreEffect5_intron) > 0:
#             # 以下示範原程式做法，但必須非常注意切片邊界
#             emissions5[g, :seq_len - sreEffect5_intron] += np.cumsum(intronicSREs5s[g, :seq_len - k + 1])[sreEffect5_intron - k + 1:]
#         if (seq_len - sreEffect5_intron) < (seq_len - k + 1 - ssRange):
#             # 如果 still in range
#             emissions5[g, seq_len - sreEffect5_intron : seq_len - k + 1 - ssRange] += np.sum(intronicSREs5s[g, :seq_len - k + 1])
#         if (seq_len - k + 1 - ssRange) > 0:
#             emissions5[g, :seq_len - k + 1 - ssRange] -= np.cumsum(intronicSREs5s[g, ssRange : seq_len - k + 1])
        
#         # -------------- 3'SS exonic effects (downstream) --------------
#         ssRange = 3
#         if (seq_len - sreEffect5_exon) > 0:
#             emissions3[g, :seq_len - sreEffect5_exon] += np.cumsum(exonicSREs3s[g, :seq_len - k + 1])[sreEffect5_exon - k + 1:]
#         if (seq_len - sreEffect5_exon) < (seq_len - k + 1 - ssRange):
#             emissions3[g, seq_len - sreEffect5_exon : seq_len - k + 1 - ssRange] += np.sum(exonicSREs3s[g, :seq_len - k + 1])
#         if (seq_len - k + 1 - ssRange) > 0:
#             emissions3[g, :seq_len - k + 1 - ssRange] -= np.cumsum(exonicSREs3s[g, ssRange : seq_len - k + 1])
    
#     # --------------------
#     # 3) 轉移機率命名
#     # --------------------
#     # transitions[0] = pME, transitions[1] = pEE
#     pME = transitions[0]
#     p1E = np.log(1 - np.exp(pME))
#     pEE = transitions[1]
#     pEO = np.log(1 - np.exp(pEE))
    
#     # --------------------
#     # 4) single-exon 機率 (ES)
#     # --------------------
#     ES = np.zeros(batch_size, dtype=float)
#     for g in range(batch_size):
#         # 在原程式: pELS[L-1] + p1E
#         # 但要注意 L 與實際 lengths[g] 是否一致
#         if (L-1) < len(pELS):
#             ES[g] = pELS[L-1] + p1E
#         else:
#             ES[g] = -np.inf  # 避免超界
    
#     # --------------------
#     # 5) 動態規劃：同時維護最佳/次佳 (best / second best)
#     # --------------------
#     for g in range(batch_size):
#         seq_len = lengths[g]
#         if seq_len <= 0:
#             continue
        
#         # (A) 主要雙層迴圈
#         #     Five[g,t], Three[g,t], secondFive[g,t], secondThree[g,t]
#         for t in range(1, seq_len):
#             # 先初始化
#             best_val_5   = pELF[t-1] if (t-1) < len(pELF) else -np.inf
#             second_val_5 = -np.inf
#             tb_best_5    = L
#             tb_second_5  = L
            
#             # 內層迴圈: d in range(t, 0, -1)
#             # 5'SS
#             for d in range(t, 0, -1):
#                 prev_idx = t - d - 1
#                 if prev_idx < 0:
#                     continue
#                 if (d-1) < len(pELM):
#                     candidate = pEE + Three[g, prev_idx] + pELM[d-1]
#                 else:
#                     candidate = -np.inf
                
#                 # 比較與 best_val_5, second_val_5
#                 if candidate > best_val_5:
#                     second_val_5 = best_val_5
#                     tb_second_5  = tb_best_5
#                     best_val_5   = candidate
#                     tb_best_5    = d
#                 elif candidate > second_val_5:
#                     second_val_5 = candidate
#                     tb_second_5  = d
            
#             # 寫回 DP + 加上發射分數
#             Five[g, t]        = best_val_5   + emissions5[g, t]
#             secondFive[g, t]  = second_val_5 + emissions5[g, t]
#             traceback5[g, t]        = tb_best_5
#             secondTraceback5[g, t]  = tb_second_5
            
#             # 接著處理 3'SS
#             best_val_3   = -np.inf
#             second_val_3 = -np.inf
#             tb_best_3    = L
#             tb_second_3  = L
            
#             for d in range(t, 0, -1):
#                 prev_idx = t - d - 1
#                 if prev_idx < 0:
#                     continue
#                 if (d-1) < len(pIL):
#                     candidate = Five[g, prev_idx] + pIL[d-1]
#                 else:
#                     candidate = -np.inf
                
#                 if candidate > best_val_3:
#                     second_val_3 = best_val_3
#                     tb_second_3  = tb_best_3
#                     best_val_3   = candidate
#                     tb_best_3    = d
#                 elif candidate > second_val_3:
#                     second_val_3 = candidate
#                     tb_second_3  = d
            
#             # 加上發射分
#             Three[g, t]        = best_val_3   + emissions3[g, t]
#             secondThree[g, t]  = second_val_3 + emissions3[g, t]
#             traceback3[g, t]        = tb_best_3
#             secondTraceback3[g, t]  = tb_second_3
        
#         # (B) 找最終最佳/次佳 loglik (類似原程式)
#         #     pME + Three[g,i] + pEO + pELL[lengths[g]-i-2]
#         for i in range(1, seq_len):
#             idx_ell = seq_len - i - 2
#             if idx_ell < 0 or idx_ell >= len(pELL):
#                 continue
#             final_candidate = pME + Three[g, i] + pEO + pELL[idx_ell]
#             # 更新 bestLogLik[g], secondBestLogLik[g]
#             if final_candidate > bestLogLik[g]:
#                 secondBestLogLik[g] = bestLogLik[g]
#                 bestLogLik[g]       = final_candidate
#                 tbindex[g] = i  # 以備回溯
#             elif final_candidate > secondBestLogLik[g]:
#                 secondBestLogLik[g] = final_candidate
        
#         # (C) single exon case
#         #    若 single exon 分數超過了 bestLogLik
#         if ES[g] > bestLogLik[g]:
#             secondBestLogLik[g] = bestLogLik[g]
#             bestLogLik[g] = ES[g]
#             tbindex[g] = -1  # 表示 single-exon 勝出
#         else:
#             # 若 single-exon 只比 second 好，也更新
#             if ES[g] > secondBestLogLik[g]:
#                 secondBestLogLik[g] = ES[g]
    
#     # --------------------
#     # 6) 回溯：bestPath & secondBestPath
#     # --------------------
#     for g in range(batch_size):
#         seq_len = lengths[g]
#         if seq_len <= 0:
#             continue
        
#         # 如果 single exon 贏了 (tbindex[g] == -1)，表示沒有分岔
#         # 這裡先示範：若是 single exon，就不做複雜回溯
#         # 若要「同時」保留 second path，也可自行設計
#         if tbindex[g] == -1:
#             # 表示 bestPath 是 single exon
#             # secondBestPath 也可能保持為 0
#             continue
        
#         # ---------- 回溯「最佳」路徑 ----------
#         i_best = tbindex[g]
#         # 從 i_best (3'SS) 開始
#         idx = i_best
#         cur_state = 3  # 先標記 3'SS
        
#         while idx > 0:
#             if cur_state == 3:
#                 bestPath[g, idx] = 3
#                 step = traceback3[g, idx]  # 取最佳
#                 idx = idx - step - 1
#                 cur_state = 5
#             else:
#                 bestPath[g, idx] = 5
#                 step = traceback5[g, idx]
#                 idx = idx - step - 1
#                 cur_state = 3
        
#         # ---------- 回溯「次佳」路徑 ----------
#         # 我們需要知道「哪個 i」對應 secondBestLogLik[g]。
#         # 這裡簡單做: 再掃一次, 找到對應 secondBestLogLik 之 i
#         i_second = 0
#         best_diff = abs(secondBestLogLik[g] - (-np.inf))
#         for i in range(1, seq_len):
#             idx_ell = seq_len - i - 2
#             if idx_ell < 0 or idx_ell >= len(pELL):
#                 continue
#             val = pME + Three[g, i] + pEO + pELL[idx_ell]
#             # 跟 secondBestLogLik[g] 比對
#             if np.isclose(val, secondBestLogLik[g], atol=1e-7):
#                 i_second = i
#                 break
        
#         idx2 = i_second
#         cur_state = 3
#         while idx2 > 0:
#             if cur_state == 3:
#                 secondBestPath[g, idx2] = 3
#                 step = secondTraceback3[g, idx2]  # 注意: 取次佳 traceback
#                 idx2 = idx2 - step - 1
#                 cur_state = 5
#             else:
#                 secondBestPath[g, idx2] = 5
#                 step = secondTraceback5[g, idx2]
#                 idx2 = idx2 - step - 1
#                 cur_state = 3
    
#     # --------------------
#     # 7) 回傳
#     # --------------------
#     return (
#         bestPath, 
#         secondBestPath, 
#         bestLogLik, 
#         secondBestLogLik, 
#         emissions5, 
#         emissions3
#     )


# import numpy as np

# 假設你的程式中有以下兩個函式 (用於計算 5'SS / 3'SS 的最大熵分數)
# 若其實是回傳多個值，就請確保最後只取單一float即可。
# def maxEnt5_single(seq, meDir=''): ...
# def maxEnt3_single(seq, meDir=''): ...

def viterbi_second_best(
    sequences, transitions, 
    pIL, pELS, pELF, pELM, pELL, 
    exonicSREs5s, exonicSREs3s, 
    intronicSREs5s, intronicSREs3s, 
    k, sreEffect5_exon, sreEffect5_intron, 
    sreEffect3_exon, sreEffect3_intron, 
    meDir=''
):
    """
    與原始程式同樣的流程與切片方式，維持對 emissionsX[] 的操作。
    唯一差異：在計算動態規劃 (DP) 時，同時維護「次佳」分數和回溯，
    最終只回傳 secondBestPath, secondBestLogLik, emissions5, emissions3。
    """
    # 1) 一些初始化
    batch_size = len(sequences)
    tbindex = np.zeros(batch_size, dtype=np.int32)
    lengths = np.zeros(batch_size, dtype=np.int32)
    loglik = np.log(np.zeros(batch_size, dtype=np.float64))  # 初始 -inf
    
    for g in range(batch_size):
        lengths[g] = len(sequences[g])
    L = np.max(lengths) if batch_size > 0 else 0
    
    # 原本初始化
    emissions3 = np.log(np.zeros((batch_size, L), dtype=np.float64))
    emissions5 = np.log(np.zeros((batch_size, L), dtype=np.float64))
    Three      = np.log(np.zeros((batch_size, L), dtype=np.float64))
    Five       = np.log(np.zeros((batch_size, L), dtype=np.float64))
    traceback5 = np.zeros((batch_size, L), dtype=np.int32) + L
    traceback3 = np.zeros((batch_size, L), dtype=np.int32) + L
    bestPath = np.zeros((batch_size, L), dtype=np.int32)
    
    # -------------- 新增：用於 "次佳" 分數 & 回溯 --------------
    secondFive       = np.log(np.zeros((batch_size, L), dtype=np.float64))  # 全為 -inf
    secondThree      = np.log(np.zeros((batch_size, L), dtype=np.float64))  # 全為 -inf
    secondTraceback5 = np.zeros((batch_size, L), dtype=np.int32) + L
    secondTraceback3 = np.zeros((batch_size, L), dtype=np.int32) + L
    
    # 次佳的「最終對數似然」
    secondBestLogLik = np.full(batch_size, -np.inf, dtype=np.float64)
    # 回溯出次佳路徑
    secondBestPath   = np.zeros((batch_size, L), dtype=np.int32)
    
    # Rewind state vars (若需要)
    exon = 2
    intron = 1
    
    # 2) Convert inputs to log space
    transitions = np.log(transitions)
    pIL  = np.log(pIL)
    pELS = np.log(pELS)
    pELF = np.log(pELF)
    pELM = np.log(pELM)
    pELL = np.log(pELL)
    
    # 3) 計算發射分數 (emissions) 並依照原程式加減 SRE
    for g in range(batch_size):
        seq_len = lengths[g]
        if seq_len == 0:
            continue
        
        # (A) 先計算 5' / 3' 的最大熵 (單一浮點)
        val5 = np.log(maxEnt5_single(sequences[g].lower(), meDir))   # 單一 float
        val3 = np.log(maxEnt3_single(sequences[g].lower(), meDir))   # 單一 float
        
        # 將該分數填入 [0: seq_len]
        emissions5[g, :seq_len] = val5
        emissions3[g, :seq_len] = val3
        
        # (B) 與原程式相同的 SRE 切片加成 / 減成
        # ------------- 5'SS exonic (upstream) -------------
        ssRange = 3
        emissions5[g, k+ssRange:seq_len] += np.cumsum(exonicSREs5s[g,:seq_len - k + 1])[:-1-ssRange]
        emissions5[g, sreEffect5_exon+1:seq_len] -= np.cumsum(exonicSREs5s[g,:seq_len - k + 1])[:-(sreEffect5_exon+1)+(k-1)]
        
        # ------------- 3'SS intronic (upstream) -------------
        ssRange = 19
        emissions3[g, k+ssRange:seq_len] += np.cumsum(intronicSREs3s[g,:seq_len - k + 1])[:-1-ssRange]
        emissions3[g, sreEffect3_intron+1:seq_len] -= np.cumsum(intronicSREs3s[g,:seq_len - k + 1])[:-(sreEffect3_intron+1)+(k-1)]
        
        # ------------- 5'SS intronic (downstream) -------------
        ssRange = 4
        emissions5[g, :seq_len - sreEffect5_intron] += np.cumsum(intronicSREs5s[g,:seq_len - k + 1])[sreEffect5_intron - k + 1:]
        emissions5[g, seq_len - sreEffect5_intron : seq_len - k + 1 - ssRange] += np.sum(intronicSREs5s[g,:seq_len - k + 1])
        emissions5[g, :seq_len - k + 1 - ssRange] -= np.cumsum(intronicSREs5s[g, ssRange : seq_len - k + 1])
        
        # ------------- 3'SS exonic (downstream) -------------
        ssRange = 3
        emissions3[g, :seq_len - sreEffect5_exon] += np.cumsum(exonicSREs3s[g,:seq_len - k + 1])[sreEffect5_exon - k + 1:]
        emissions3[g, seq_len - sreEffect5_exon : seq_len - k + 1 - ssRange] += np.sum(exonicSREs3s[g,:seq_len - k + 1])
        emissions3[g, :seq_len - k + 1 - ssRange] -= np.cumsum(exonicSREs3s[g, ssRange : seq_len - k + 1])
    
    # 4) 命名轉移機率
    pME = transitions[0]
    p1E = np.log(1 - np.exp(pME))
    pEE = transitions[1]
    pEO = np.log(1 - np.exp(pEE))
    
    # 5) single-exon 分數 (ES)
    ES = np.zeros(batch_size, dtype=np.float64)
    for g in range(batch_size):
        # 原程式: pELS[L-1] + p1E
        # 若 L=0 或 L-1 >= len(pELS) 則可能要保護
        if L > 0 and (L-1) < len(pELS):
            ES[g] = pELS[L-1] + p1E
        else:
            ES[g] = -np.inf
    
    # 6) 動態規劃：同時維護「最佳」與「次佳」
    for g in range(batch_size):
        seq_len = lengths[g]
        if seq_len <= 0:
            continue
        
        for t in range(1, seq_len):
            # (A) 初始化 5'SS
            Five[g, t] = pELF[t-1] if (t-1)<len(pELF) else -np.inf
            secondFive[g, t] = -np.inf  # 預設次佳是 -inf
            
            # 內層迴圈 (原程式)
            for d in range(t, 0, -1):
                prev = t - d - 1
                if prev < 0:
                    continue
                if (d-1) < len(pELM):
                    candidate = pEE + Three[g, prev] + pELM[d-1]
                else:
                    candidate = -np.inf
                
                # 更新最佳/次佳
                if candidate > Five[g, t]:
                    # 原本 best -> second
                    secondFive[g, t] = Five[g, t]
                    secondTraceback5[g, t] = traceback5[g, t]
                    # 新 candidate -> best
                    Five[g, t] = candidate
                    traceback5[g, t] = d
                elif candidate > secondFive[g, t]:
                    secondFive[g, t] = candidate
                    secondTraceback5[g, t] = d
            
            # 加上發射分數
            Five[g, t] += emissions5[g, t]
            secondFive[g, t] += emissions5[g, t]
            
            # (B) 3'SS
            Three[g, t] = -np.inf
            secondThree[g, t] = -np.inf
            # 原程式: if Five[g, t-d-1] + pIL[d-1] > Three[g, t]:
            for d in range(t, 0, -1):
                prev = t - d - 1
                if prev < 0:
                    continue
                if (d-1) < len(pIL):
                    candidate = Five[g, prev] + pIL[d-1]
                else:
                    candidate = -np.inf
                
                if candidate > Three[g, t]:
                    secondThree[g, t] = Three[g, t]
                    secondTraceback3[g, t] = traceback3[g, t]
                    Three[g, t] = candidate
                    traceback3[g, t] = d
                elif candidate > secondThree[g, t]:
                    secondThree[g, t] = candidate
                    secondTraceback3[g, t] = d
            
            Three[g, t] += emissions3[g, t]
            secondThree[g, t] += emissions3[g, t]
        
        # (C) 更新最佳對數似然 loglik[g] (同原程式)
        for i in range(1, seq_len):
            idx_ell = seq_len - i - 2
            if idx_ell < 0 or idx_ell >= len(pELL):
                continue
            final_val = pME + Three[g, i] + pEO + pELL[idx_ell]
            
            if final_val > loglik[g]:
                loglik[g] = final_val
                tbindex[g] = i
        
        # (D) single exon 是否勝過
        if ES[g] > loglik[g]:
            loglik[g] = ES[g]
            tbindex[g] = -1  # 表示 single exon 狀態
        
        # 這裡還沒更新 secondBestLogLik，等下一步再處理
    
    # 7) 回溯「最佳」路徑 (原程式)，但我們最終不 return bestPath
    for g in range(batch_size):
        if tbindex[g] == -1:
            # 表示 single-exon，原程式也不回溯
            continue
        
        while tbindex[g] > 0:
            bestPath[g, tbindex[g]] = 3
            tbindex[g] -= traceback3[g, tbindex[g]] + 1
            bestPath[g, tbindex[g]] = 5
            tbindex[g] -= traceback5[g, tbindex[g]] + 1
    
    # 8) 找「次佳」最終分數 secondBestLogLik[g] 並回溯 secondBestPath
    for g in range(batch_size):
        seq_len = lengths[g]
        if seq_len <= 0:
            continue
        # 若 single-exon 勝出 (tbindex[g] == -1)，
        # 則原程式中沒有另外計算 second best；這裡可自行設計要不要算
        if tbindex[g] == -1:
            secondBestLogLik[g] = -np.inf
            continue
        
        # (A) 找 possible second best ending
        # 類似對 best log-lik 做檢查，但換成 secondThree[g,i]
        # 其實我們要對「bestThree」和「secondThree」都做檢查，
        #   pME + secondThree[g, i] + pEO + pELL[idx_ell]
        #   以及
        #   pME + bestThree[g, i] (但 best i 不同) => 這需要較複雜條件
        # 這裡先做最簡單版本：掃描 i, 取 pME + secondThree[g, i] + ...
        for i in range(1, seq_len):
            idx_ell = seq_len - i - 2
            if idx_ell < 0 or idx_ell >= len(pELL):
                continue
            val2 = pME + secondThree[g, i] + pEO + pELL[idx_ell]
            if val2 > secondBestLogLik[g]:
                secondBestLogLik[g] = val2
                # 暫存 "i" 做回溯用
                second_i = i
        
        # (B) 回溯次佳路徑
        #    與最佳路徑回溯類似，但是要使用 secondTraceback3, secondTraceback5
        if secondBestLogLik[g] == -np.inf:
            # 代表沒有找到 second best
            continue
        
        # 真正回溯
        idx = second_i
        state = 3
        while idx > 0:
            if state == 3:
                secondBestPath[g, idx] = 3
                step = secondTraceback3[g, idx]
                idx = idx - step - 1
                state = 5
            else:
                secondBestPath[g, idx] = 5
                step = secondTraceback5[g, idx]
                idx = idx - step - 1
                state = 3
    
    # 9) 只回傳次佳路徑 + 次佳 loglik + emissions
    return secondBestPath, secondBestLogLik, emissions5, emissions3



# 假設你在其他檔案或其他區塊中，有定義好的函式:
# def maxEnt5_single(seq, meDir) -> float: ...
# def maxEnt3_single(seq, meDir) -> float: ...
# 這裡不重複定義，只示範如何在主程式裡使用。

# def viterbi_second_best(
#     sequences, 
#     transitions,
#     pIL, pELS, pELF, pELM, pELL,
#     exonicSREs5s, exonicSREs3s, 
#     intronicSREs5s, intronicSREs3s,
#     k,
#     sreEffect5_exon, sreEffect5_intron,
#     sreEffect3_exon, sreEffect3_intron,
#     meDir=''
# ):
#     """
#     基於原本的 viterbi() 函式，擴充「次佳路徑」功能:
#       1. 將發射分數 (emissions5, emissions3) 結合 SRE 效果。
#       2. 在動態規劃陣列中，同時維護 best / second 分數及其回溯。
#       3. 最後回傳:
#          - bestPath        : 最佳路徑標記 (shape: batch_size x max_length)
#          - secondBestPath  : 次佳路徑標記
#          - bestLogLik      : 最佳路徑對數似然
#          - secondLogLik    : 次佳路徑對數似然
#          - emissions5, emissions3 : 方便後續檢查發射分數
#     """

#     # ------------------------
#     # 0) 一些前置初始化
#     # ------------------------
#     batch_size = len(sequences)
    
#     # 記錄每條序列長度
#     lengths = np.array([len(seq) for seq in sequences], dtype=int)
#     L = np.max(lengths) if batch_size > 0 else 0
    
#     # 轉成對數空間 (log space)
#     transitions = np.log(transitions)
#     pIL = np.log(pIL)
#     pELS = np.log(pELS)
#     pELF = np.log(pELF)
#     pELM = np.log(pELM)
#     pELL = np.log(pELL)
    
#     # 初始化回傳用的發射機率
#     # 用 -inf 來填滿是因為長度以外的部分，我們不做計算
#     emissions5 = np.full((batch_size, L), -np.inf, dtype=float)
#     emissions3 = np.full((batch_size, L), -np.inf, dtype=float)
    
#     # ------------------------
#     # 1) 先計算基本的 5'SS / 3'SS 最大熵分數
#     # ------------------------
#     for g in range(batch_size):
#         seq_len = lengths[g]
#         if seq_len == 0:
#             continue
        
#         # 逐位置計算 5' / 3' 的 maxEnt (基礎發射分)
#         for t in range(seq_len):
#             # 注意: 若原程式帶的是 sequence[g].lower(), 請自行對應
#             # 這裡示範帶 meDir 參數
#             emissions5[g, t] = np.log(maxEnt5_single(sequences[g].lower(), meDir))
#             emissions3[g, t] = np.log(maxEnt3_single(sequences[g].lower(), meDir))
    
#     # ------------------------
#     # 2) 將 SRE 效果整合到發射分數
#     #    (以下示範保留與你原程式相似的加法/減法邏輯)
#     # ------------------------
#     for g in range(batch_size):
#         seq_len = lengths[g]
#         if seq_len == 0:
#             continue
        
#         # === 5'SS exonic effects (upstream) ===
#         ssRange = 3
#         # exonicSREs5s[g,:seq_len-k+1] 可能要注意邊界
#         if seq_len > k + ssRange:
#             # 加
#             emissions5[g, k+ssRange:seq_len] += np.cumsum(exonicSREs5s[g, :seq_len - k + 1])[:-1 - ssRange]
#         if seq_len > sreEffect5_exon + 1:
#             # 減
#             emissions5[g, sreEffect5_exon+1:seq_len] -= np.cumsum(exonicSREs5s[g, :seq_len - k + 1])[:-(sreEffect5_exon + 1) + (k - 1)]
        
#         # === 3'SS intronic effects (upstream) ===
#         ssRange = 19
#         if seq_len > k + ssRange:
#             emissions3[g, k+ssRange:seq_len] += np.cumsum(intronicSREs3s[g, :seq_len - k + 1])[:-1 - ssRange]
#         if seq_len > sreEffect3_intron + 1:
#             emissions3[g, sreEffect3_intron+1:seq_len] -= np.cumsum(intronicSREs3s[g, :seq_len - k + 1])[:-(sreEffect3_intron + 1) + (k - 1)]
        
#         # === 5'SS intronic effects (downstream) ===
#         ssRange = 4
#         # 這部分的原程式邏輯也很複雜，請注意邊界
#         # 以下只是模擬把 intronicSREs5s[g,...] 累加到 emissions5
#         if seq_len > sreEffect5_intron:
#             # + ...
#             idx_start = sreEffect5_intron - k + 1
#             if idx_start < 0: idx_start = 0
#             # 加上 cumulative sum ...
#             if (seq_len - k + 1) > 0 and idx_start < len(intronicSREs5s[g, :seq_len - k + 1]):
#                 emissions5[g, :seq_len - sreEffect5_intron] += np.cumsum(intronicSREs5s[g, :seq_len - k + 1])[idx_start:]
        
#         # === 3'SS exonic effects (downstream) ===
#         ssRange = 3
#         # 同理，以下也要注意實際模型中的邏輯
#         if seq_len > sreEffect5_exon:
#             idx_start = sreEffect5_exon - k + 1
#             if idx_start < 0: idx_start = 0
#             if (seq_len - k + 1) > 0 and idx_start < len(exonicSREs3s[g, :seq_len - k + 1]):
#                 emissions3[g, :seq_len - sreEffect5_exon] += np.cumsum(exonicSREs3s[g, :seq_len - k + 1])[idx_start:]
    
#     # ------------------------
#     # 3) 命名 HMM 轉移機率
#     # ------------------------
#     # transitions[0] = pME, transitions[1] = pEE, 其餘看你原先定義
#     pME = transitions[0]  # intron -> exon (舉例)
#     p1E = np.log(1 - np.exp(pME))
#     pEE = transitions[1]  # exon -> exon (舉例)
#     pEO = np.log(1 - np.exp(pEE))
    
#     # ------------------------
#     # 4) 建立 DP 陣列 (best/second)
#     # ------------------------
#     bestFive  = np.full((batch_size, L), -np.inf, dtype=float)
#     bestThree = np.full((batch_size, L), -np.inf, dtype=float)
#     traceback5  = np.full((batch_size, L), 0, dtype=int)
#     traceback3  = np.full((batch_size, L), 0, dtype=int)
    
#     secondFive  = np.full((batch_size, L), -np.inf, dtype=float)
#     secondThree = np.full((batch_size, L), -np.inf, dtype=float)
#     secondTraceback5  = np.full((batch_size, L), 0, dtype=int)
#     secondTraceback3  = np.full((batch_size, L), 0, dtype=int)
    
#     # 用來記錄最終路徑得分
#     bestLogLik = np.full(batch_size, -np.inf, dtype=float)
#     secondLogLik = np.full(batch_size, -np.inf, dtype=float)
    
#     # 單外顯子 (single exon) 的分數 (ES[g])
#     ES = np.full(batch_size, -np.inf, dtype=float)
#     for g in range(batch_size):
#         seq_len = lengths[g]
#         if seq_len == 0:
#             continue
#         # 例如: pELS[ seq_len - 1 ] + p1E
#         ES[g] = pELS[seq_len - 1] + p1E if (seq_len - 1) < len(pELS) else -np.inf
    
#     # ------------------------
#     # 5) Viterbi 主迴圈 (同時維護第一 & 第二名)
#     # ------------------------
#     for g in range(batch_size):
#         seq_len = lengths[g]
#         if seq_len == 0:
#             continue
        
#         # 依原程式邏輯, t=0 通常是初始化; 
#         # 這裡若你需要做任何特別初始化, 可以在此處處理
#         # bestFive[g, 0] = ... (依需求)
#         # bestThree[g, 0] = ...
        
#         # 主要從 t=1 開始更新
#         for t in range(1, seq_len):
#             # -------------------------
#             # (A) 更新 5'SS (Five)
#             # -------------------------
#             # 先以 pELF[t-1] 當成「起始候選值」
#             # (與你的原程式相同邏輯)
#             current_best   = pELF[t-1] if (t-1) < len(pELF) else -np.inf
#             current_second = -np.inf
#             current_tb_best   = 0
#             current_tb_second = 0
            
#             # 在原程式，你會對所有 d in range(t, 0, -1) 做檢查
#             for d in range(t, 0, -1):
#                 # candidate = pEE + bestThree[g, t - d - 1] + pELM[d-1]
#                 # 注意 t-d-1 可能 < 0，要先判斷
#                 prev_idx = t - d - 1
#                 if prev_idx < 0: 
#                     continue
#                 if (d-1) < len(pELM):
#                     candidate_score = pEE + bestThree[g, prev_idx] + pELM[d-1]
#                 else:
#                     candidate_score = -np.inf
                
#                 # 和 best / second 比較
#                 if candidate_score > current_best:
#                     # 原本 best -> second
#                     current_second = current_best
#                     current_tb_second = current_tb_best
#                     # 新 candidate -> best
#                     current_best = candidate_score
#                     current_tb_best = d
#                 elif candidate_score > current_second:
#                     current_second = candidate_score
#                     current_tb_second = d
            
#             # 寫回 DP
#             bestFive[g, t] = current_best + emissions5[g, t]
#             secondFive[g, t] = current_second + emissions5[g, t]
#             traceback5[g, t] = current_tb_best
#             secondTraceback5[g, t] = current_tb_second
            
#             # -------------------------
#             # (B) 更新 3'SS (Three)
#             # -------------------------
#             current_best   = -np.inf
#             current_second = -np.inf
#             current_tb_best   = 0
#             current_tb_second = 0
            
#             for d in range(t, 0, -1):
#                 prev_idx = t - d - 1
#                 if prev_idx < 0:
#                     continue
#                 if (d-1) < len(pIL):
#                     candidate_score = bestFive[g, prev_idx] + pIL[d-1]
#                 else:
#                     candidate_score = -np.inf
                
#                 if candidate_score > current_best:
#                     current_second = current_best
#                     current_tb_second = current_tb_best
#                     current_best = candidate_score
#                     current_tb_best = d
#                 elif candidate_score > current_second:
#                     current_second = candidate_score
#                     current_tb_second = d
            
#             # 寫回 DP
#             bestThree[g, t] = current_best + emissions3[g, t]
#             secondThree[g, t] = current_second + emissions3[g, t]
#             traceback3[g, t] = current_tb_best
#             secondTraceback3[g, t] = current_tb_second
        
#         # -------------------------
#         # (C) 找該序列最終的最佳/次佳 loglik
#         # -------------------------
#         # 與原程式類似: pME + bestThree[g, i] + pEO + pELL[lengths[g] - i - 2]
#         # 注意邊界: i 迭代範圍 & pELL[...] index
#         for i in range(1, seq_len):
#             idx_ell = seq_len - i - 2
#             if idx_ell < 0 or idx_ell >= len(pELL):
#                 continue
#             final_candidate = pME + bestThree[g, i] + pEO + pELL[idx_ell]
            
#             # 比對全局 best / second
#             if final_candidate > bestLogLik[g]:
#                 # 舊的 best -> second
#                 secondLogLik[g] = bestLogLik[g]
#                 # 新的 -> best
#                 bestLogLik[g] = final_candidate
#             elif final_candidate > secondLogLik[g]:
#                 secondLogLik[g] = final_candidate
        
#         # (D) single exon case
#         if ES[g] > bestLogLik[g]:
#             # 若 single exon 分數高於目前 best
#             secondLogLik[g] = bestLogLik[g]
#             bestLogLik[g] = ES[g]
#         else:
#             # 若 single exon 沒有贏過 best, 但可能贏過 second
#             if ES[g] > secondLogLik[g]:
#                 secondLogLik[g] = ES[g]
    
#     # ------------------------
#     # 6) 回溯 (bestPath & secondBestPath)
#     # ------------------------
#     bestPath = np.zeros((batch_size, L), dtype=int)
#     secondBestPath = np.zeros((batch_size, L), dtype=int)
    
#     for g in range(batch_size):
#         seq_len = lengths[g]
#         if seq_len == 0:
#             continue
        
#         # ---- (A) 回溯「最佳」路徑 ----
#         # 你原程式中，是用 tbindex[g] = i 來記錄最終位置
#         # 在此示範: 直接從 argmax(bestThree[g, :seq_len]) 開始往回
#         # 或者從最後計算 pME + bestThree[g, i] + pEO + ... 找到那個 i
#         # 這裡給個簡單示範:
#         i_best = 0
#         best_val = -np.inf
#         for i in range(seq_len):
#             idx_ell = seq_len - i - 2
#             if idx_ell < 0 or idx_ell >= len(pELL):
#                 continue
#             val = pME + bestThree[g, i] + pEO + pELL[idx_ell]
#             if val > best_val:
#                 best_val = val
#                 i_best = i
        
#         # 依你的設計, 從 i_best 沿著 traceback3 -> traceback5 交替回溯
#         tbidx = i_best
#         state_toggle = 3  # 先標記 3'SS, 接著 5'SS
#         while tbidx > 0:
#             if state_toggle == 3:
#                 bestPath[g, tbidx] = 3  # 3'SS
#                 step = traceback3[g, tbidx]  # best traceback
#                 tbidx = tbidx - step - 1
#                 state_toggle = 5
#             else:
#                 bestPath[g, tbidx] = 5  # 5'SS
#                 step = traceback5[g, tbidx]  
#                 tbidx = tbidx - step - 1
#                 state_toggle = 3
        
#         # ---- (B) 回溯「次佳」路徑 ----
#         # 同理：我們可能想找「哪個 i 能得到 secondLogLik[g]」
#         # 或者照同樣方式，把 bestThree[g, i] 換成 secondThree[g, i] 來尋找
#         # 下方示範: 直接找 second best 3'SS dp
#         i_second = 0
#         second_val = -np.inf
#         for i in range(seq_len):
#             idx_ell = seq_len - i - 2
#             if idx_ell < 0 or idx_ell >= len(pELL):
#                 continue
#             val = pME + secondThree[g, i] + pEO + pELL[idx_ell]
#             if val > second_val:
#                 second_val = val
#                 i_second = i
        
#         tbidx2 = i_second
#         state_toggle = 3
#         while tbidx2 > 0:
#             if state_toggle == 3:
#                 secondBestPath[g, tbidx2] = 3
#                 step = secondTraceback3[g, tbidx2]
#                 tbidx2 = tbidx2 - step - 1
#                 state_toggle = 5
#             else:
#                 secondBestPath[g, tbidx2] = 5
#                 step = secondTraceback5[g, tbidx2]
#                 tbidx2 = tbidx2 - step - 1
#                 state_toggle = 3
    
#     # ------------------------
#     # 7) 回傳
#     # ------------------------
#     return (
#         bestPath,         # shape: (batch_size, L)
#         secondBestPath,   # shape: (batch_size, L)
#         bestLogLik,       # shape: (batch_size,)
#         secondLogLik,     # shape: (batch_size,)
#         emissions5,
#         emissions3
#     )




# def viterbi_second_best(
#     sequences, 
#     transitions, 
#     pIL, pELS, pELF, pELM, pELL, 
#     # 下列 SRE/最大熵等參數省略部分細節，只示範 DP 邏輯
#     emissions5_func,  # 假設用來計算 5' 的 emission 分數的函式
#     emissions3_func,  # 假設用來計算 3' 的 emission 分數的函式
#     meDir=''
# ):
#     """
#     傳回:
#         bestPath      : 最佳路徑（同原 Viterbi）
#         secondBestPath: 第二佳路徑
#         bestLogLik    : 最佳路徑對數似然
#         secondLogLik  : 次佳路徑對數似然
#         emissions5, emissions3 : 方便除錯或後續分析
#     """
#     batch_size = len(sequences)
    
#     # 先記錄序列長度
#     lengths = np.array([len(seq) for seq in sequences], dtype=int)
#     L = np.max(lengths)
    
#     # 轉換到對數空間
#     transitions = np.log(transitions)
#     pIL = np.log(pIL)
#     pELS = np.log(pELS)
#     pELF = np.log(pELF)
#     pELM = np.log(pELM)
#     pELL = np.log(pELL)
    
#     # 預先初始化發射分數 (emissions)
#     emissions5 = np.full((batch_size, L), -np.inf, dtype=float)
#     emissions3 = np.full((batch_size, L), -np.inf, dtype=float)
    
#     # 計算 emissions5, emissions3（這裡用假設函式，實際請用你自己的邏輯 / maxEnt 計算）
#     for g in range(batch_size):
#         for t in range(lengths[g]):
#             emissions5[g, t] = emissions5_func(sequences[g], t, meDir)
#             emissions3[g, t] = emissions3_func(sequences[g], t, meDir)
    
#     # 命名對應隱馬可夫狀態轉移機率（維持原邏輯或依需求調整）
#     pME = transitions[0]  # 例如 intron->exon
#     p1E = np.log(1 - np.exp(pME))
#     pEE = transitions[1]  # 例如 exon->exon
#     pEO = np.log(1 - np.exp(pEE))
    
#     # -------------------------------------------------------------------------
#     # 下面是最關鍵的差異：
#     # 為了追蹤「次佳」路徑，多建立 second 相關的 DP 陣列
#     # -------------------------------------------------------------------------
#     # bestFive[g,t], bestThree[g,t]     : 該位置的最佳分數
#     # secondFive[g,t], secondThree[g,t] : 該位置的次佳分數
#     # traceback5[g,t], traceback3[g,t]  : 最佳路徑的回溯
#     # secondTraceback5[g,t], secondTraceback3[g,t] : 次佳路徑的回溯
#     bestFive  = np.full((batch_size, L), -np.inf, dtype=float)
#     bestThree = np.full((batch_size, L), -np.inf, dtype=float)
#     traceback5  = np.full((batch_size, L), -1, dtype=int)
#     traceback3  = np.full((batch_size, L), -1, dtype=int)
    
#     secondFive  = np.full((batch_size, L), -np.inf, dtype=float)
#     secondThree = np.full((batch_size, L), -np.inf, dtype=float)
#     secondTraceback5  = np.full((batch_size, L), -1, dtype=int)
#     secondTraceback3  = np.full((batch_size, L), -1, dtype=int)
    
#     # 最後的 loglik（最佳 & 次佳）
#     bestLogLik = np.full(batch_size, -np.inf, dtype=float)
#     secondLogLik = np.full(batch_size, -np.inf, dtype=float)
    
#     # 同原程式：若要處理單外顯子 (single exon) 的特殊情況，可以額外放在 ES 內
#     ES = np.full(batch_size, -np.inf, dtype=float)
#     for g in range(batch_size):
#         # 這裡假設 single-exon case 的分數
#         ES[g] = pELS[lengths[g]-1] + p1E  # 與原程式雷同
    
#     # -----------------------------
#     # Viterbi 動態規劃主迴圈
#     # -----------------------------
#     for g in range(batch_size):
#         seq_len = lengths[g]
        
#         # 初始化 t=0 的情況（若有需要，可視狀態分配）
#         # 例如: bestFive[g, 0] = some_init, bestThree[g, 0] = some_init
#         # 這裡視實際模型需求而定
#         if seq_len == 0:
#             continue
        
#         # 從 t=1 到 t=seq_len-1
#         for t in range(1, seq_len):
            
#             # -----------------------------------------------------
#             # 更新 bestFive[g, t] / secondFive[g, t]
#             # -----------------------------------------------------
#             # 先假設 bestFive[g, t] = pELF[t-1]，和原程式類似
#             current_best   = pELF[t-1]       # 預設值 (可能是init)
#             current_second = -np.inf
#             current_tb_best   = 0
#             current_tb_second = -1
            
#             # 在原程式，會對所有 d in range(t, 0, -1) 做檢查
#             for d in range(t, 0, -1):
#                 candidate_score = pEE + bestThree[g, t - d - 1] + pELM[d-1]
#                 # (1) 檢查是否比目前 best 分數高
#                 if candidate_score > current_best:
#                     # 原本 best 變成 second
#                     current_second = current_best
#                     current_tb_second = current_tb_best
#                     # 新 candidate 成為 best
#                     current_best = candidate_score
#                     current_tb_best = d
#                 # (2) 沒超過 best，但有機會更新 second
#                 elif candidate_score > current_second:
#                     current_second = candidate_score
#                     current_tb_second = d
            
#             bestFive[g, t] = current_best
#             secondFive[g, t] = current_second
            
#             traceback5[g, t] = current_tb_best
#             secondTraceback5[g, t] = current_tb_second
            
#             # 再加上發射分數 (emissions5)
#             bestFive[g, t]   += emissions5[g, t]
#             secondFive[g, t] += emissions5[g, t]
            
#             # -----------------------------------------------------
#             # 更新 bestThree[g, t] / secondThree[g, t]
#             # -----------------------------------------------------
#             current_best   = -np.inf
#             current_second = -np.inf
#             current_tb_best   = -1
#             current_tb_second = -1
            
#             for d in range(t, 0, -1):
#                 candidate_score = bestFive[g, t - d - 1] + pIL[d-1]
#                 if candidate_score > current_best:
#                     current_second = current_best
#                     current_tb_second = current_tb_best
                    
#                     current_best = candidate_score
#                     current_tb_best = d
#                 elif candidate_score > current_second:
#                     current_second = candidate_score
#                     current_tb_second = d
            
#             bestThree[g, t] = current_best
#             secondThree[g, t] = current_second
            
#             traceback3[g, t] = current_tb_best
#             secondTraceback3[g, t] = current_tb_second
            
#             # 加上發射分數 (emissions3)
#             bestThree[g, t]   += emissions3[g, t]
#             secondThree[g, t] += emissions3[g, t]
        
#         # -----------------------------------------------------
#         # 找最終 (best / second) 的 log-likelihood
#         # （類似原程式的結尾，檢查 pME + Three[g,i] + pEO + pELL[...]）
#         # -----------------------------------------------------
#         for i in range(1, seq_len):
#             final_candidate = pME + bestThree[g, i] + pEO + pELL[seq_len - i - 2]
#             # 檢查是否為全局最佳
#             if final_candidate > bestLogLik[g]:
#                 secondLogLik[g] = bestLogLik[g]
#                 bestLogLik[g]   = final_candidate
#             elif final_candidate > secondLogLik[g]:
#                 secondLogLik[g] = final_candidate
        
#         # -----------------------------------------------------
#         # 若 single exon (ES[g]) 的分數大，可能覆蓋以上結果
#         # -----------------------------------------------------
#         if ES[g] > bestLogLik[g]:
#             bestLogLik[g] = ES[g]
#             # 若 single-exon case 超過了原本的 bestLogLik，
#             # 則 secondLogLik 就要更新成原本的 bestLogLik
#             # 不過依你的需求決定是否要把 single-exon 情況加入 second
#             # 這裡示範簡單邏輯
#             # secondLogLik[g] = max(secondLogLik[g], ???)
    
#     # -----------------------------
#     # 回溯：bestPath & secondBestPath
#     # -----------------------------
#     # 回溯的想法是一樣的，只是要分別沿著
#     # (bestTraceback5, bestTraceback3) 與
#     # (secondTraceback5, secondTraceback3)
#     # 取出兩條不同路徑
#     bestPath = np.zeros((batch_size, L), dtype=int)
#     secondBestPath = np.zeros((batch_size, L), dtype=int)
    
#     for g in range(batch_size):
#         # 以原程式的方式回溯「最佳路徑」
#         # （假設我們最後確定的 best 切點是存放在 tbindex[g]）
#         # 這裡只是概念示範，依實際程式中如何紀錄 tbindex[g] 而定
#         tbidx = np.argmax(bestThree[g, :lengths[g]])  # 假設用 bestThree 來尋找最終位置
#         while tbidx > 0:
#             bestPath[g, tbidx] = 3
#             tbidx -= traceback3[g, tbidx] + 1
#             bestPath[g, tbidx] = 5
#             tbidx -= traceback5[g, tbidx] + 1
        
#         # 「次佳路徑」同理：要根據 secondTracebackX 來回溯
#         tbidx2 = np.argmax(secondThree[g, :lengths[g]])
#         while tbidx2 > 0:
#             secondBestPath[g, tbidx2] = 3
#             tbidx2 -= secondTraceback3[g, tbidx2] + 1
#             secondBestPath[g, tbidx2] = 5
#             tbidx2 -= secondTraceback5[g, tbidx2] + 1
    
#     return bestPath, secondBestPath, bestLogLik, secondLogLik, emissions5, emissions3


