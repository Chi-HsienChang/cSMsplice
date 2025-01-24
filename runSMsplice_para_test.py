#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import time, argparse
from Bio import SeqIO, SeqUtils, motifs
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import scipy.ndimage
import scipy.stats as stats
from SMsplice import *
from tqdm import tqdm

# ========== 關鍵：多行程 & 並行套件 ==========
import multiprocessing
from joblib import Parallel, delayed

startTime = time.time()

# -----------------------------
# 1) 先將 load_gene 函式定義在檔案最外層 (module-level)
#    這樣 multiprocessing 才能 pickle & 傳給子進程
# -----------------------------
def load_gene(gene, canonical, genome):
    """
    單個基因的序列載入與處理 (含 strand 判斷)。
    回傳：(gene_name, SeqRecord物件)；若遇到非標準鹼基或 strand 錯誤就回傳 (gene_name, None)。
    """
    txstart = canonical.loc[gene, 4] - 1
    txend   = canonical.loc[gene, 5]
    chrom   = canonical.loc[gene, 2][3:]  # 去掉 "chr"

    seq_slice = genome[chrom][txstart:txend].seq
    # 檢查是否只有 a, c, g, t
    if not np.array_equiv(['a', 'c', 'g', 't'], np.unique(seq_slice.lower())):
        return (gene, None)

    name  = gene
    geneID= gene
    desc  = (f"{gene} GeneID:{gene} TranscriptID:Canonical "
             f"Chromosome:{canonical.loc[gene,2]} Start:{txstart} "
             f"Stop:{txend} Strand:{canonical.loc[gene,3]}")

    strand = canonical.loc[gene, 3]
    if strand == '-':
        seq_final = seq_slice.reverse_complement()
    elif strand == '+':
        seq_final = seq_slice
    else:
        print(gene, "strand error")
        return (gene, None)

    record = SeqRecord(seq_final, name=name, id=geneID, description=desc)
    return (gene, record)


def parse_arguments():
    #argument parsing 
    parser = argparse.ArgumentParser(description="parsing arguments")
    parser.add_argument("-c", "--canonical_ss", required=True)
    parser.add_argument("-a", "--all_ss", required=True)
    parser.add_argument("-g", "--genome", required=True)
    parser.add_argument("-m", "--maxent_dir", required=True)
    
    parser.add_argument("--prelearned_sres", 
                        choices=['none','human','mouse','zebrafish','fly','moth','arabidopsis'],
                        default='none')
    parser.add_argument("--learn_sres", action="store_true")
    parser.add_argument("--max_learned_scores", type=int, default=1000)
    
    parser.add_argument("--learning_seed", 
                        choices=['none','real-decoy'], 
                        default='none')
    parser.add_argument("--max_learned_weights", type=int, default=15)
    
    parser.add_argument("--print_predictions", action="store_true")
    parser.add_argument("--print_local_scores", action="store_true")
    
    opts = parser.parse_args()
    return opts


# -----------------------------
# 2) 並行計算 SRE 分數的函式
#    (和原本單迴圈差不多，只是封裝一下方便 joblib.parallel)
# -----------------------------
def compute_sre_arrays(sequence, length, kmer,
                       s5_exon, s3_exon, s5_intron, s3_intron,
                       debug_line=None):
    """
    計算該序列在 exonic/intronic、5'/3' 的對數 SRE 分數。
    回傳 (ex5, ex3, in5, in3)，各是 np.array(...長度 = length-kmer+1)。
    """
    if debug_line is not None:
        print(f"{debug_line}: sequence start compute")

    seq_lower = sequence.lower()
    ex5 = np.log(np.array(sreScores_single(seq_lower, s5_exon,   kmer)))
    ex3 = np.log(np.array(sreScores_single(seq_lower, s3_exon,   kmer)))
    in5= np.log(np.array(sreScores_single(seq_lower, s5_intron, kmer)))
    in3= np.log(np.array(sreScores_single(seq_lower, s3_intron, kmer)))

    # 只取長度 (length - kmer + 1)
    return (
        ex5[:length-kmer+1],
        ex3[:length-kmer+1],
        in5[:length-kmer+1],
        in3[:length-kmer+1]
    )


def main():
    args = parse_arguments()
    
    # Load data from arguments
    maxEntDir = args.maxent_dir
    genome    = SeqIO.to_dict(SeqIO.parse(args.genome, "fasta"))
    canonical = pd.read_csv(args.canonical_ss, sep='\t', engine='python', 
                            index_col=0, header=None)
    canonical.index = canonical.index.map(str)
    allSS = pd.read_csv(args.all_ss, sep='\t', engine='python', 
                        index_col=0, header=None)
    allSS.index = allSS.index.map(str)

    # ========== 多行程 載入 genes ========== 
    print("Loading Genes in parallel...")
    genes = {}
    gene_list = list(canonical.index)

    # 準備 starmap 參數
    tasks = [(g, canonical, genome) for g in gene_list]
    cpu_num = min(multiprocessing.cpu_count(), 8)  # 可依需求調整
    with multiprocessing.Pool(processes=cpu_num) as pool:
        results = list(tqdm(
            pool.starmap(load_gene, tasks),
            total=len(gene_list),
            desc="Loading Genes"
        ))
    for gene_name, record in results:
        if record is not None:
            genes[gene_name] = record

    # 其餘參數與原程式相同
    print("Finished loading genes:", len(genes))

    # Additional parameters
    sreEffect = 80
    sreEffect3_intron = sreEffect + 19
    sreEffect3_exon   = sreEffect + 3
    sreEffect5_intron = sreEffect + 5
    sreEffect5_exon   = sreEffect + 3
    np.seterr(divide='ignore')
    
    kmer = 6
    E = 0
    I = 1
    B5 = 5
    B3 = 3
    train_size = 4000
    test_size = 1000
    score_learning_rate = .01

    # Get training, validation, generalization, and test sets
    testGenes = canonical[(canonical[1] == 0)&canonical[2].isin(['chr2','chr4'])].index 
    testGenes = np.intersect1d(testGenes, list(genes.keys()))

    trainGenes = canonical.index
    trainGenes = np.intersect1d(trainGenes, list(genes.keys()))
    trainGenes = np.setdiff1d(trainGenes, testGenes)

    lengthsOfGenes = np.array([len(str(genes[gene].seq)) for gene in trainGenes])
    trainGenes = trainGenes[lengthsOfGenes > sreEffect3_intron]
    lengthsOfGenes = np.array([len(str(genes[gene].seq)) for gene in trainGenes])
    validationGenes = trainGenes[lengthsOfGenes < 200000]

    generalizationGenes = np.intersect1d(validationGenes, canonical[canonical[1]==0].index)
    if len(generalizationGenes) > test_size:
        generalizationGenes = np.random.choice(generalizationGenes, test_size, replace=False)
    validationGenes = np.setdiff1d(validationGenes, generalizationGenes)
    if len(validationGenes) > test_size:
        validationGenes = np.random.choice(validationGenes, test_size, replace=False)

    trainGenes = np.setdiff1d(trainGenes, generalizationGenes)
    trainGenes = np.setdiff1d(trainGenes, validationGenes)
    if len(trainGenes) > train_size:
        trainGenes = np.random.choice(trainGenes, train_size, replace=False)

    cannonical_annotations = {}
    annotations = {}
    for gene in tqdm(genes.keys(), desc='Getting True Sequences'):
        annnotation = []
        info = genes[gene].description.split(' ')
        
        if canonical.loc[gene,6] == ',':
            cannonical_annotations[gene] = [(0, int(info[5][5:]) - int(info[4][6:]) - 1)]
            continue
            
        exonEnds   = [int(start)-1 for start in canonical.loc[gene,6].split(',')[:-1]] + [int(info[5][5:])]
        exonStarts = [int(info[4][6:])] + [int(end)+1 for end in canonical.loc[gene,7].split(',')[:-1]]
        exonEnds[-1] -= 1
        exonStarts[0]+= 2
        
        if info[6] == 'Strand:-': 
            stop = int(info[5][5:])
            for i in range(len(exonEnds),0,-1):
                annnotation.append((stop - exonEnds[i-1] - 1, stop - exonStarts[i-1] + 1))
        elif info[6] == 'Strand:+':
            start = int(info[4][6:])
            for i in range(len(exonEnds)):
                annnotation.append((exonStarts[i]-start-2, exonEnds[i]-start))
                
        cannonical_annotations[gene] = annnotation
        annotations[gene] = {'Canonical': annnotation}

    trueSeqs = trueSequencesCannonical(genes, cannonical_annotations, E, I, B3, B5)

    # Structural parameters
    (numExonsPerGene, lengthSingleExons, lengthFirstExons, 
     lengthMiddleExons, lengthLastExons, lengthIntrons
    ) = structuralParameters(trainGenes, annotations)

    N = int(np.ceil(max(numExonsPerGene)/10.0)*10)
    numExonsHist = np.histogram(numExonsPerGene, bins=N, range=(0,N))[0]
    numExonsDF   = pd.DataFrame(list(zip(np.arange(0,N), numExonsHist)), columns=['Index','Count'])

    p1E = float(numExonsDF['Count'][1]) / numExonsDF['Count'].sum()
    pEO = float(numExonsDF['Count'][2:].sum()) / (numExonsDF['Index'][2:]*numExonsDF['Count'][2:]).sum()
    numExonsDF['Prob'] = [pEO*(1-pEO)**(i-2) for i in range(0,N)]
    numExonsDF.loc[1,'Prob'] = p1E
    numExonsDF.loc[0,'Prob'] = 0
    numExonsDF['Empirical']  = numExonsDF['Count']/numExonsDF['Count'].sum()
    transitions = [1 - p1E, 1 - pEO]

    N = 5000000
    lengthIntronsHist = np.histogram(lengthIntrons, bins=N, range=(0,N))[0]
    lengthIntronsDF   = pd.DataFrame(list(zip(np.arange(0,N), lengthIntronsHist)), 
                                     columns=['Index','Count'])
    lengthIntronsDF['Prob'] = lengthIntronsDF['Count']/lengthIntronsDF['Count'].sum()
    pIL = geometric_smooth_tailed(lengthIntrons, N, 5, 200, lower_cutoff=60)

    lengthFirstExonsHist = np.histogram(lengthFirstExons, bins=N, range=(0,N))[0]
    lengthFirstExonsDF   = pd.DataFrame(list(zip(np.arange(0,N), lengthFirstExonsHist)), 
                                        columns=['Index','Count'])
    lengthFirstExonsDF['Prob'] = lengthFirstExonsDF['Count']/lengthFirstExonsDF['Count'].sum()
    pELF = adaptive_kde_tailed(lengthFirstExons, N)

    lengthMiddleExonsHist = np.histogram(lengthMiddleExons, bins=N, range=(0,N))[0]
    lengthMiddleExonsDF   = pd.DataFrame(list(zip(np.arange(0,N), lengthMiddleExonsHist)), 
                                         columns=['Index','Count'])
    lengthMiddleExonsDF['Prob'] = lengthMiddleExonsDF['Count']/lengthMiddleExonsDF['Count'].sum()
    pELM = adaptive_kde_tailed(lengthMiddleExons, N)

    lengthLastExonsHist = np.histogram(lengthLastExons, bins=N, range=(0,N))[0]
    lengthLastExonsDF   = pd.DataFrame(list(zip(np.arange(0,N), lengthLastExonsHist)), 
                                       columns=['Index','Count'])
    lengthLastExonsDF['Prob'] = lengthLastExonsDF['Count']/lengthLastExonsDF['Count'].sum()
    pELL = adaptive_kde_tailed(lengthLastExons, N)
    pELS = 0*np.copy(pELL)

    sreScores_exon   = np.ones(4**kmer)
    sreScores_intron = np.ones(4**kmer)

    if args.prelearned_sres != 'none':
        sreScores_exon   = np.load('organism_parameters/' + args.prelearned_sres + '/sreScores_exon.npy')
        sreScores_intron = np.load('organism_parameters/' + args.prelearned_sres + '/sreScores_intron.npy')

    sreScores3_exon   = np.copy(sreScores_exon)
    sreScores5_exon   = np.copy(sreScores_exon)
    sreScores3_intron = np.copy(sreScores_intron)
    sreScores5_intron = np.copy(sreScores_intron)

    # Learning seed
    if args.learning_seed == 'real-decoy' and args.learn_sres:
        me5 = maxEnt5(trainGenes, genes, maxEntDir)
        me3 = maxEnt3(trainGenes, genes, maxEntDir)
        
        tolerance = .5
        decoySS = {}
        for gene in tqdm(trainGenes, desc='Learning Seed'):
            decoySS[gene] = np.zeros(len(genes[gene]), dtype=int)
        
        # 5'SS
        five_scores = []
        for gene in tqdm(trainGenes, desc='gene for 5\'SS'):
            for score in np.log2(me5[gene][trueSeqs[gene] == B5][1:]):
                five_scores.append(score)
        five_scores = np.array(five_scores)
        five_scores_tracker = np.flip(np.sort(list(five_scores)))

        for score in tqdm(five_scores_tracker, desc='score for 5\'SS'):
            np.random.shuffle(trainGenes)
            g = 0
            while g < len(trainGenes):
                gene = trainGenes[g]
                g += 1
                true_ss   = get_all_5ss(gene, allSS, genes)
                used_sites= np.nonzero(decoySS[gene] == B5)[0]

                gene5s    = np.log2(me5[gene])
                sort_inds = np.argsort(gene5s)
                sort_inds = sort_inds[~np.in1d(sort_inds, true_ss)]
                sort_inds = sort_inds[~np.in1d(sort_inds, used_sites)]
                L = len(sort_inds)
                gene5s = gene5s[sort_inds]

                up_i   = np.searchsorted(gene5s, score, 'left')
                down_i = up_i - 1
                if down_i >= L: 
                    down_i = L-1
                if up_i >= L:
                    up_i = L-1

                if abs(score - gene5s[down_i]) < tolerance and decoySS[gene][sort_inds[down_i]] == 0:
                    decoySS[gene][sort_inds[down_i]] = B5
                    g = len(trainGenes)
                elif abs(score - gene5s[up_i]) < tolerance and decoySS[gene][sort_inds[up_i]] == 0:
                    decoySS[gene][sort_inds[up_i]] = B5
                    g = len(trainGenes)
                    
        # 3'SS 
        three_scores = []
        for gene in tqdm(trainGenes, desc='gene for 3\'SS'):
            for score in np.log2(me3[gene][trueSeqs[gene] == B3][:-1]):
                three_scores.append(score)
        three_scores = np.array(three_scores)
        three_scores_tracker = np.flip(np.sort(list(three_scores)))

        for score in tqdm(three_scores_tracker, desc='score for 3\'SS'):
            np.random.shuffle(trainGenes)
            g = 0
            while g < len(trainGenes):
                gene = trainGenes[g]
                g += 1
                true_ss   = get_all_3ss(gene, allSS, genes)
                used_sites= np.nonzero(decoySS[gene] == B3)[0]

                gene3s    = np.log2(me3[gene])
                sort_inds = np.argsort(gene3s)
                sort_inds = sort_inds[~np.in1d(sort_inds, true_ss)]
                sort_inds = sort_inds[~np.in1d(sort_inds, used_sites)]
                L = len(sort_inds)
                gene3s = gene3s[sort_inds]

                up_i   = np.searchsorted(gene3s, score, 'left')
                down_i = up_i - 1
                if down_i >= L:
                    down_i = L-1
                if up_i >= L:
                    up_i = L-1

                if abs(score - gene3s[down_i]) < tolerance and decoySS[gene][sort_inds[down_i]] == 0:
                    decoySS[gene][sort_inds[down_i]] = B3
                    g = len(trainGenes)
                elif abs(score - gene3s[up_i]) < tolerance and decoySS[gene][sort_inds[up_i]] == 0:
                    decoySS[gene][sort_inds[up_i]] = B3
                    g = len(trainGenes)

        (sreScores_intron, sreScores_exon, sreScores3_intron, sreScores3_exon, 
         sreScores5_intron, sreScores5_exon
        ) = get_hexamer_real_decoy_scores(
            trainGenes, trueSeqs, decoySS, genes, 
            kmer=kmer, 
            sreEffect5_exon=sreEffect5_exon, sreEffect5_intron=sreEffect5_intron, 
            sreEffect3_exon=sreEffect3_exon, sreEffect3_intron=sreEffect3_intron
        )
        sreScores3_intron = sreScores_intron
        sreScores3_exon   = sreScores_exon
        sreScores5_intron = np.copy(sreScores_intron)
        sreScores5_exon   = np.copy(sreScores_exon)
        
        # Learn weight
        lengths   = np.array([len(str(genes[gene].seq)) for gene in validationGenes])
        sequences = [str(genes[gene].seq) for gene in validationGenes]    

        step_size   = 1
        sre_weights = [0, step_size]
        scores      = []
        for sre_weight in tqdm(sre_weights, desc='Learning Weight'):
            exonicSREs5s   = np.zeros((len(lengths), max(lengths)-kmer+1))
            exonicSREs3s   = np.zeros((len(lengths), max(lengths)-kmer+1))
            intronicSREs5s = np.zeros((len(lengths), max(lengths)-kmer+1))
            intronicSREs3s = np.zeros((len(lengths), max(lengths)-kmer+1))

            # ========== 並行計算 SRE 分數 ==========
            res_sre = Parallel(n_jobs=-1)(
                delayed(compute_sre_arrays)(
                    sequences[g], lengths[g], kmer,
                    np.exp(np.log(sreScores5_exon)*sre_weight),
                    np.exp(np.log(sreScores3_exon)*sre_weight),
                    np.exp(np.log(sreScores5_intron)*sre_weight),
                    np.exp(np.log(sreScores3_intron)*sre_weight)
                )
                for g in range(len(sequences))
            )
            for g, (ex5, ex3, in5, in3) in enumerate(res_sre):
                exonicSREs5s[g, :len(ex5)]   = ex5
                exonicSREs3s[g, :len(ex3)]   = ex3
                intronicSREs5s[g, :len(in5)] = in5
                intronicSREs3s[g, :len(in3)] = in3
            
            pred_all = viterbi(
                sequences=sequences, transitions=transitions,
                pIL=pIL, pELS=pELS, pELF=pELF, pELM=pELM, pELL=pELL,
                exonicSREs5s=exonicSREs5s, exonicSREs3s=exonicSREs3s,
                intronicSREs5s=intronicSREs5s, intronicSREs3s=intronicSREs3s,
                k=kmer, 
                sreEffect5_exon=sreEffect5_exon, sreEffect5_intron=sreEffect5_intron, 
                sreEffect3_exon=sreEffect3_exon, sreEffect3_intron=sreEffect3_intron,
                meDir=maxEntDir
            )
            
            # Get the Sensitivity and Precision
            num_truePositives  = 0
            num_falsePositives = 0
            num_falseNegatives = 0

            for g, gene in tqdm(enumerate(validationGenes), desc='Calculating Sensitivity and Precision'):
                L = lengths[g]
                predThrees = np.nonzero(pred_all[0][g,:L] == 3)[0]
                trueThrees = np.nonzero(trueSeqs[gene] == B3)[0]

                predFives  = np.nonzero(pred_all[0][g,:L] == 5)[0]
                trueFives  = np.nonzero(trueSeqs[gene] == B5)[0]
                
                num_truePositives  += (len(np.intersect1d(predThrees, trueThrees)) 
                                       + len(np.intersect1d(predFives, trueFives)))
                num_falsePositives += (len(np.setdiff1d(predThrees, trueThrees)) 
                                       + len(np.setdiff1d(predFives, trueFives)))
                num_falseNegatives += (len(np.setdiff1d(trueThrees, predThrees)) 
                                       + len(np.setdiff1d(trueFives, predFives)))

            ssSens = num_truePositives / (num_truePositives + num_falseNegatives)
            ssPrec = num_truePositives / (num_truePositives + num_falsePositives)
            f1     = 2 / (1/ssSens + 1/ssPrec)
            scores.append(f1)
        
        scores     = np.array(scores)
        sre_weights= np.array(sre_weights)
        while len(scores) < 15:
            i = np.argmax(scores)
            if i == len(scores) - 1:
                sre_weights_test = [sre_weights[-1] + step_size]
            elif i == 0:
                sre_weights_test = [sre_weights[1]/2]
            else:
                sre_weights_test = [
                    sre_weights[i]/2 + sre_weights[i-1]/2,
                    sre_weights[i]/2 + sre_weights[i+1]/2
                ]
            
            for sre_weight in tqdm(sre_weights_test, desc='Learning sre_weight'):
                sre_weights = np.append(sre_weights, sre_weight)
                
                exonicSREs5s   = np.zeros((len(lengths), max(lengths)-kmer+1))
                exonicSREs3s   = np.zeros((len(lengths), max(lengths)-kmer+1))
                intronicSREs5s = np.zeros((len(lengths), max(lengths)-kmer+1))
                intronicSREs3s = np.zeros((len(lengths), max(lengths)-kmer+1))

                res_sre = Parallel(n_jobs=-1)(
                    delayed(compute_sre_arrays)(
                        sequences[g], lengths[g], kmer,
                        np.exp(np.log(sreScores5_exon)*sre_weight),
                        np.exp(np.log(sreScores3_exon)*sre_weight),
                        np.exp(np.log(sreScores5_intron)*sre_weight),
                        np.exp(np.log(sreScores3_intron)*sre_weight),
                        debug_line=340
                    )
                    for g in range(len(sequences))
                )
                for g, (ex5, ex3, in5, in3) in enumerate(res_sre):
                    exonicSREs5s[g, :len(ex5)]   = ex5
                    exonicSREs3s[g, :len(ex3)]   = ex3
                    intronicSREs5s[g, :len(in5)] = in5
                    intronicSREs3s[g, :len(in3)] = in3

                pred_all = viterbi(
                    sequences=sequences, transitions=transitions,
                    pIL=pIL, pELS=pELS, pELF=pELF, pELM=pELM, pELL=pELL,
                    exonicSREs5s=exonicSREs5s, exonicSREs3s=exonicSREs3s,
                    intronicSREs5s=intronicSREs5s, intronicSREs3s=intronicSREs3s,
                    k=kmer,
                    sreEffect5_exon=sreEffect5_exon, sreEffect5_intron=sreEffect5_intron, 
                    sreEffect3_exon=sreEffect3_exon, sreEffect3_intron=sreEffect3_intron,
                    meDir=maxEntDir
                )
                
                num_truePositives  = 0
                num_falsePositives = 0
                num_falseNegatives = 0
                for g, gene in tqdm(enumerate(validationGenes), desc='Calculating Sensitivity and Precision'):
                    L = lengths[g]
                    predThrees= np.nonzero(pred_all[0][g,:L] == 3)[0]
                    trueThrees= np.nonzero(trueSeqs[gene] == B3)[0]

                    predFives = np.nonzero(pred_all[0][g,:L] == 5)[0]
                    trueFives = np.nonzero(trueSeqs[gene] == B5)[0]
                
                    num_truePositives  += (len(np.intersect1d(predThrees, trueThrees)) 
                                           + len(np.intersect1d(predFives, trueFives)))
                    num_falsePositives += (len(np.setdiff1d(predThrees, trueThrees)) 
                                           + len(np.setdiff1d(predFives, trueFives)))
                    num_falseNegatives += (len(np.setdiff1d(trueThrees, predThrees)) 
                                           + len(np.setdiff1d(trueFives, predFives)))

                ssSens = num_truePositives / (num_truePositives + num_falseNegatives)
                ssPrec = num_truePositives / (num_truePositives + num_falsePositives)
                f1     = 2 / (1/ssSens + 1/ssPrec)
                scores = np.append(scores, f1)

            scores     = scores[np.argsort(sre_weights)]
            sre_weights= sre_weights[np.argsort(sre_weights)]
        
        # Set up scores for score learning
        sre_weight = sre_weights[np.argmax(scores)]
        
        sreScores_exon   = np.exp(np.log(sreScores5_exon)*sre_weight)
        sreScores5_exon  = np.exp(np.log(sreScores5_exon)*sre_weight)
        sreScores3_exon  = np.exp(np.log(sreScores3_exon)*sre_weight)
        sreScores_intron = np.exp(np.log(sreScores5_intron)*sre_weight)
        sreScores5_intron= np.exp(np.log(sreScores5_intron)*sre_weight)
        sreScores3_intron= np.exp(np.log(sreScores3_intron)*sre_weight)

    # Learning
    if args.learn_sres:
        lengthsOfGenes   = np.array([len(str(genes[gene].seq)) for gene in trainGenes])
        trainGenesShort  = trainGenes[lengthsOfGenes < 200000]
        np.random.shuffle(trainGenesShort)
        trainGenesShort  = np.array_split(trainGenesShort, 4)

        trainGenes1     = trainGenesShort[0]
        trainGenes2     = trainGenesShort[1]
        trainGenes3     = trainGenesShort[2]
        trainGenes4     = trainGenesShort[3]
        trainGenesTest  = generalizationGenes
        held_f1         = -1
        
        learning_counter= -1
        doneTime        = 10
        while 1 < doneTime:
            learning_counter += 1
            if learning_counter > args.max_learned_scores:
                break
            update_scores = True
            
            if   learning_counter%5 == 1: trainGenesSub = np.copy(trainGenes1)
            elif learning_counter%5 == 2: trainGenesSub = np.copy(trainGenes2)
            elif learning_counter%5 == 3: trainGenesSub = np.copy(trainGenes3)
            elif learning_counter%5 == 4: trainGenesSub = np.copy(trainGenes4)
            else:
                trainGenesSub = np.copy(trainGenesTest)
                update_scores = False
            
            lengths   = np.array([len(str(genes[gene].seq)) for gene in trainGenesSub])
            sequences = [str(genes[gene].seq) for gene in trainGenesSub]
            
            exonicSREs5s   = np.zeros((len(lengths), max(lengths)-kmer+1))
            exonicSREs3s   = np.zeros((len(lengths), max(lengths)-kmer+1))
            intronicSREs5s = np.zeros((len(lengths), max(lengths)-kmer+1))
            intronicSREs3s = np.zeros((len(lengths), max(lengths)-kmer+1))

            # ========== 並行計算 SRE 分數 ==========
            res_sre = Parallel(n_jobs=-1)(
                delayed(compute_sre_arrays)(
                    sequences[g], lengths[g], kmer,
                    sreScores5_exon, sreScores3_exon,
                    sreScores5_intron, sreScores3_intron,
                    debug_line=421
                )
                for g in range(len(sequences))
            )
            for g, (ex5, ex3, in5, in3) in enumerate(res_sre):
                exonicSREs5s[g, :len(ex5)]   = ex5
                exonicSREs3s[g, :len(ex3)]   = ex3
                intronicSREs5s[g, :len(in5)] = in5
                intronicSREs3s[g, :len(in3)] = in3

            pred_all = viterbi(
                sequences=sequences, transitions=transitions,
                pIL=pIL, pELS=pELS, pELF=pELF, pELM=pELM, pELL=pELL,
                exonicSREs5s=exonicSREs5s, exonicSREs3s=exonicSREs3s,
                intronicSREs5s=intronicSREs5s, intronicSREs3s=intronicSREs3s,
                k=kmer,
                sreEffect5_exon=sreEffect5_exon, sreEffect5_intron=sreEffect5_intron,
                sreEffect3_exon=sreEffect3_exon, sreEffect3_intron=sreEffect3_intron,
                meDir=maxEntDir
            )

            # Get the False Negatives and False Positives
            falsePositives = {}
            falseNegatives = {}

            num_truePositives  = 0
            num_falsePositives = 0
            num_falseNegatives = 0
            
            for g, gene in tqdm(enumerate(trainGenesSub), desc='Calculating Sensitivity and Precision'):
                L = lengths[g]
                falsePositives[gene] = np.zeros(len(genes[gene]), dtype=int)
                falseNegatives[gene] = np.zeros(len(genes[gene]), dtype=int)
            
                predThrees = np.nonzero(pred_all[0][g,:L] == 3)[0]
                trueThrees = np.nonzero(trueSeqs[gene] == B3)[0]

                predFives = np.nonzero(pred_all[0][g,:L] == 5)[0]
                trueFives = np.nonzero(trueSeqs[gene] == B5)[0]
                
                num_truePositives += (len(np.intersect1d(predThrees, trueThrees)) 
                                      + len(np.intersect1d(predFives, trueFives)))

                fp_threes = np.setdiff1d(predThrees, trueThrees)
                fp_fives  = np.setdiff1d(predFives,  trueFives)
                falsePositives[gene][fp_threes] = B3
                falsePositives[gene][fp_fives]  = B5
                num_falsePositives += (len(fp_threes) + len(fp_fives))

                fn_threes = np.setdiff1d(trueThrees, predThrees)
                fn_fives  = np.setdiff1d(trueFives,  predFives)
                falseNegatives[gene][fn_threes] = B3
                falseNegatives[gene][fn_fives]  = B5
                num_falseNegatives += (len(fn_threes) + len(fn_fives))

            ssSens = num_truePositives / (num_truePositives + num_falseNegatives)
            ssPrec = num_truePositives / (num_truePositives + num_falsePositives)
            f1     = 2 / (1/ssSens + 1/ssPrec)
            
            if update_scores:
                (set1_counts_5_intron, set1_counts_5_exon,
                 set1_counts_3_intron, set1_counts_3_exon,
                 set2_counts_5_intron, set2_counts_5_exon,
                 set2_counts_3_intron, set2_counts_3_exon
                ) = get_hexamer_counts(
                    trainGenesSub, falseNegatives, falsePositives, genes,
                    kmer=kmer,
                    sreEffect5_exon=sreEffect5_exon, sreEffect5_intron=sreEffect5_intron,
                    sreEffect3_exon=sreEffect3_exon, sreEffect3_intron=sreEffect3_intron
                )

                set1_counts_intron = set1_counts_5_intron + set1_counts_3_intron
                set1_counts_exon   = set1_counts_5_exon   + set1_counts_3_exon
                set2_counts_intron = set2_counts_5_intron + set2_counts_3_intron
                set2_counts_exon   = set2_counts_5_exon   + set2_counts_3_exon

                psuedocount_denominator_intron = (np.sum(set1_counts_intron) 
                                                  + np.sum(set2_counts_intron))
                set1_counts_intron += np.sum(set1_counts_intron)/psuedocount_denominator_intron
                set2_counts_intron += np.sum(set2_counts_intron)/psuedocount_denominator_intron

                frequency_ratio_intron = (
                    (set1_counts_intron/np.sum(set1_counts_intron)) /
                    (set2_counts_intron/np.sum(set2_counts_intron))
                )
                sreScores_intron *= frequency_ratio_intron**score_learning_rate

                psuedocount_denominator_exon = (np.sum(set1_counts_exon) 
                                                + np.sum(set2_counts_exon))
                set1_counts_exon += np.sum(set1_counts_exon)/psuedocount_denominator_exon
                set2_counts_exon += np.sum(set2_counts_exon)/psuedocount_denominator_exon

                frequency_ratio_exon = (
                    (set1_counts_exon/np.sum(set1_counts_exon)) /
                    (set2_counts_exon/np.sum(set2_counts_exon))
                )
                sreScores_exon *= frequency_ratio_exon**score_learning_rate

                sreScores3_intron = sreScores_intron
                sreScores3_exon   = sreScores_exon
                sreScores5_intron = sreScores_intron
                sreScores5_exon   = sreScores_exon
            else:
                if f1 >= held_f1:
                    held_f1               = np.copy(f1)
                    held_sreScores5_exon  = np.copy(sreScores5_exon)
                    held_sreScores3_exon  = np.copy(sreScores3_exon)
                    held_sreScores5_intron= np.copy(sreScores5_intron)
                    held_sreScores3_intron= np.copy(sreScores3_intron)
                else:
                    doneTime = 0
        
        sreScores5_exon   = np.copy(held_sreScores5_exon)
        sreScores3_exon   = np.copy(held_sreScores3_exon)
        sreScores5_intron = np.copy(held_sreScores5_intron)
        sreScores3_intron = np.copy(held_sreScores3_intron)

    # Filter test set
    lengthsOfGenes= np.array([len(str(genes[gene].seq)) for gene in testGenes])
    testGenes     = testGenes[lengthsOfGenes > sreEffect3_intron]

    notShortIntrons= []
    for gene in tqdm(testGenes, desc='Filtering Test Set'):
        trueThrees = np.nonzero(trueSeqs[gene] == B3)[0]
        trueFives  = np.nonzero(trueSeqs[gene] == B5)[0]
        
        n_fives  = np.sum(trueSeqs[gene] == B5)
        n_threes = np.sum(trueSeqs[gene] == B3)
        
        if n_fives != n_threes:
            notShortIntrons.append(False)
        elif np.min(trueThrees - trueFives + 1) < 25:
            notShortIntrons.append(False)
        else:
            notShortIntrons.append(True)

    notShortIntrons = np.array(notShortIntrons)
    testGenes       = testGenes[notShortIntrons]
    lengths         = np.array([len(str(genes[gene].seq)) for gene in testGenes])
    sequences       = [str(genes[gene].seq) for gene in testGenes]

    exonicSREs5s   = np.zeros((len(lengths), max(lengths)-kmer+1))
    exonicSREs3s   = np.zeros((len(lengths), max(lengths)-kmer+1))
    intronicSREs5s = np.zeros((len(lengths), max(lengths)-kmer+1))
    intronicSREs3s = np.zeros((len(lengths), max(lengths)-kmer+1))

    # ========== 並行計算 SRE 分數 (最終測試集) ==========
    print("Calculating SREs for Test Genes in parallel...")
    res_sre = Parallel(n_jobs=-1)(
        delayed(compute_sre_arrays)(
            sequences[g], lengths[g], kmer,
            sreScores5_exon, sreScores3_exon,
            sreScores5_intron, sreScores3_intron,
            debug_line=530
        )
        for g in range(len(sequences))
    )
    for g, (ex5, ex3, in5, in3) in enumerate(res_sre):
        exonicSREs5s[g, :len(ex5)]   = ex5
        exonicSREs3s[g, :len(ex3)]   = ex3
        intronicSREs5s[g, :len(in5)] = in5
        intronicSREs3s[g, :len(in3)] = in3

    pred_all = viterbi(
        sequences=sequences, transitions=transitions,
        pIL=pIL, pELS=pELS, pELF=pELF, pELM=pELM, pELL=pELL,
        exonicSREs5s=exonicSREs5s, exonicSREs3s=exonicSREs3s,
        intronicSREs5s=intronicSREs5s, intronicSREs3s=intronicSREs3s,
        k=kmer,
        sreEffect5_exon=sreEffect5_exon, sreEffect5_intron=sreEffect5_intron,
        sreEffect3_exon=sreEffect3_exon, sreEffect3_intron=sreEffect3_intron,
        meDir=maxEntDir
    )

    # Get the Sensitivity and Precision
    num_truePositives  = 0
    num_falsePositives = 0
    num_falseNegatives = 0

    for g, gene in tqdm(enumerate(testGenes), desc='Calculating Sensitivity and Precision'):
        L = lengths[g]
        predThrees = np.nonzero(pred_all[0][g,:L] == 3)[0]
        trueThrees = np.nonzero(trueSeqs[gene] == B3)[0]

        predFives  = np.nonzero(pred_all[0][g,:L] == 5)[0]
        trueFives  = np.nonzero(trueSeqs[gene] == B5)[0]
        
        if args.print_predictions:
            print(gene)
            print("\tAnnotated Fives:", trueFives, "Predicted Fives:", predFives)
            print("\tAnnotated Threes:", trueThrees,"Predicted Threes:", predThrees)
            
        num_truePositives  += (len(np.intersect1d(predThrees, trueThrees)) 
                               + len(np.intersect1d(predFives, trueFives)))
        num_falsePositives += (len(np.setdiff1d(predThrees, trueThrees)) 
                               + len(np.setdiff1d(predFives, trueFives)))
        num_falseNegatives += (len(np.setdiff1d(trueThrees, predThrees)) 
                               + len(np.setdiff1d(trueFives, predFives)))

    ssSens = num_truePositives / (num_truePositives + num_falseNegatives)
    ssPrec = num_truePositives / (num_truePositives + num_falsePositives)
    f1     = 2 / (1/ssSens + 1/ssPrec)
    print("Final Test Metrics", "Recall", ssSens, 
          "Precision", ssPrec, "f1", f1)

    # print_local_scores
    if args.print_local_scores:
        scored_sequences_5, scored_sequences_3 = score_sequences(
            sequences=sequences, 
            exonicSREs5s=exonicSREs5s, exonicSREs3s=exonicSREs3s,
            intronicSREs5s=intronicSREs5s, intronicSREs3s=intronicSREs3s,
            k=kmer,
            sreEffect5_exon=sreEffect5_exon, sreEffect5_intron=sreEffect5_intron,
            sreEffect3_exon=sreEffect3_exon, sreEffect3_intron=sreEffect3_intron,
            meDir=maxEntDir
        )
        for g, gene in tqdm(enumerate(testGenes), desc='Printing Local Scores'):
            L = lengths[g]
            predThrees = np.nonzero(pred_all[0][g,:L] == 3)[0]
            predFives  = np.nonzero(pred_all[0][g,:L] == 5)[0]
            
            print(gene, "Internal Exons:")
            for i, three in enumerate(predThrees[:-1]):
                five = predFives[i+1]
                score_val = (np.log2(scored_sequences_5[g,five])
                            + np.log2(scored_sequences_3[g,three])
                            + np.log2(pELM[five-three]))
                print("\t", score_val)
            
            print(gene, "Introns:")
            for i, three in enumerate(predThrees):
                five = predFives[i]
                score_val = (np.log2(scored_sequences_5[g,five])
                            + np.log2(scored_sequences_3[g,three])
                            + np.log2(pELM[-five + three]))
                print("\t", score_val)


if __name__ == "__main__":
    main()
