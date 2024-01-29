import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import re, os, sys
from collections import Counter
import math
import numpy as np
import re
from featureGenerator import *
from readToMatrix import *
def read_protein_sequences(file):
    if os.path.exists(file) == False:
        print('Error: file %s does not exist.' % file)
        sys.exit(1)
    with open(file) as f:
        records = f.read()
    if re.search('>', records) == None:
        print('Error: the input file %s seems not in FASTA format!' % file)
        sys.exit(1)
    records = records.split('>')[1:]
    fasta_sequences = []
    for fasta in records:
        array = fasta.split('\n')
        header, sequence = array[0].split()[0], re.sub('[^ACDEFGHIKLMNPQRSTVWY-]', '', ''.join(array[1:]).upper())
        header_array = header.split('|')
        name = header_array[0]
        #label = 'None' #header_array[1] if len(header_array) >= 1 else '0'
        #label_train = 'None' #header_array[2] if len(header_array) >= 2 else 'training'
        fasta_sequences.append([name, sequence])
    return fasta_sequences


def AAC(fastas, **kw):
    AA = 'ACDEFGHIKLMNPQRSTVWY'
    #AA = 'ARNDCQEGHILKMFPSTWYV'
    encodings = []
    header = []
    for i in AA:
        header.append(i)
    #encodings.append(header)

    for i in fastas:
        name, sequence = i[0], re.sub('-', '', i[1])
        count = Counter(sequence)
        for key in count:
            count[key] = count[key]/len(sequence)
        code = []
        for aa in AA:
            code.append(count[aa])
        encodings.append(code)
    return np.array(encodings, dtype=float)


def AAINDEX(fastas, props=None, **kw):
    AA = 'ARNDCQEGHILKMFPSTWYV'
    fileAAindex = 'data/AAindex.txt'
    with open(fileAAindex) as f:
        records = f.readlines()[1:]

    AAindex = []
    AAindexName = []
    for i in records:
        AAindex.append(i.rstrip().split()[1:] if i.rstrip() != '' else None)
        AAindexName.append(i.rstrip().split()[0] if i.rstrip() != '' else None)

    index = {}
    for i in range(len(AA)):
        index[AA[i]] = i

    #  use the user inputed properties
    if props:
        tmpIndexNames = []
        tmpIndex = []
        for p in props:
            if AAindexName.index(p) != -1:
                tmpIndexNames.append(p)
                tmpIndex.append(AAindex[AAindexName.index(p)])
        if len(tmpIndexNames) != 0:
            AAindexName = tmpIndexNames
            AAindex = tmpIndex
    
    header = []
    for idName in AAindexName:
        header.append(idName)
    
    encodings = []
    for i in fastas:
        name, sequence = i[0], i[1]
        code = []
        
        for j in AAindex:
            tmp = 0
            for aa in sequence:
                if aa == '-':
                    tmp = tmp + 0
                else:
                    tmp = tmp + float(j[index[aa]])
            code.append(tmp/len(sequence))
        encodings.append(code)
    return np.array(encodings, dtype=float)


def APAAC(fastas, lambdaValue=1, w=0.05, **kw):
    records = []
    records.append("#	A	R	N	D	C	Q	E	G	H	I	L	K	M	F	P	S	T	W	Y	V")
    records.append("Hydrophobicity	0.62	-2.53	-0.78	-0.9	0.29	-0.85	-0.74	0.48	-0.4	1.38	1.06	-1.5	0.64	1.19	0.12	-0.18	-0.05	0.81	0.26	1.08")
    records.append("Hydrophilicity	-0.5	3	0.2	3	-1	0.2	3	0	-0.5	-1.8	-1.8	3	-1.3	-2.5	0	0.3	-0.4	-3.4	-2.3	-1.5")
    records.append("SideChainMass	15	101	58	59	47	72	73	1	82	57	57	73	75	91	42	31	45	130	107	43")

    AA = ''.join(records[0].rstrip().split()[1:])
    AADict = {}
    for i in range(len(AA)):
        AADict[AA[i]] = i
    AAProperty = []
    AAPropertyNames = []
    for i in range(1, len(records) - 1):
        array = records[i].rstrip().split() if records[i].rstrip() != '' else None
        AAProperty.append([float(j) for j in array[1:]])
        AAPropertyNames.append(array[0])

    AAProperty1 = []
    for i in AAProperty:
        meanI = sum(i) / 20
        fenmu = math.sqrt(sum([(j - meanI) ** 2 for j in i]) / 20)
        AAProperty1.append([(j - meanI) / fenmu for j in i])

    encodings = []
    header = []
    for i in AA:
        header.append('Pc1.' + i)
    for j in range(1, lambdaValue + 1):
        for i in AAPropertyNames:
            header.append('Pc2.' + i + '.' + str(j))
    
    for i in fastas:
        name, sequence = i[0], re.sub('-', '', i[1])
        code = []
        theta = []

        for n in range(1, lambdaValue + 1):
            for j in range(len(AAProperty1)):
                theta.append(sum([AAProperty1[j][AADict[sequence[k]]] * AAProperty1[j][AADict[sequence[k + n]]] for k in
                                  range(len(sequence) - n)]) / (len(sequence) - n))
        myDict = {}
        for aa in AA:
            myDict[aa] = sequence.count(aa)

        code = code + [myDict[aa] / (1 + w * sum(theta)) for aa in AA]
        code = code + [w * value / (1 + w * sum(theta)) for value in theta]

        encodings.append(code)
    return np.array(encodings, dtype=float)

# ASA required SPINEX external program
# BINARY encoding required same protein sequence length

def BLOSUM62(fastas, **kw):
    AA = 'ARNDCQEGHILKMFPSTWYV'
    blosum62 = {
        'A': [4,  -1, -2, -2, 0,  -1, -1, 0, -2,  -1, -1, -1, -1, -2, -1, 1,  0,  -3, -2, 0],  # A
        'R': [-1, 5,  0,  -2, -3, 1,  0,  -2, 0,  -3, -2, 2,  -1, -3, -2, -1, -1, -3, -2, -3], # R
        'N': [-2, 0,  6,  1,  -3, 0,  0,  0,  1,  -3, -3, 0,  -2, -3, -2, 1,  0,  -4, -2, -3], # N
        'D': [-2, -2, 1,  6,  -3, 0,  2,  -1, -1, -3, -4, -1, -3, -3, -1, 0,  -1, -4, -3, -3], # D
        'C': [0,  -3, -3, -3, 9,  -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1], # C
        'Q': [-1, 1,  0,  0,  -3, 5,  2,  -2, 0,  -3, -2, 1,  0,  -3, -1, 0,  -1, -2, -1, -2], # Q
        'E': [-1, 0,  0,  2,  -4, 2,  5,  -2, 0,  -3, -3, 1,  -2, -3, -1, 0,  -1, -3, -2, -2], # E
        'G': [0,  -2, 0,  -1, -3, -2, -2, 6,  -2, -4, -4, -2, -3, -3, -2, 0,  -2, -2, -3, -3], # G
        'H': [-2, 0,  1,  -1, -3, 0,  0,  -2, 8,  -3, -3, -1, -2, -1, -2, -1, -2, -2, 2,  -3], # H
        'I': [-1, -3, -3, -3, -1, -3, -3, -4, -3, 4,  2,  -3, 1,  0,  -3, -2, -1, -3, -1, 3],  # I
        'L': [-1, -2, -3, -4, -1, -2, -3, -4, -3, 2,  4,  -2, 2,  0,  -3, -2, -1, -2, -1, 1],  # L
        'K': [-1, 2,  0,  -1, -3, 1,  1,  -2, -1, -3, -2, 5,  -1, -3, -1, 0,  -1, -3, -2, -2], # K
        'M': [-1, -1, -2, -3, -1, 0,  -2, -3, -2, 1,  2,  -1, 5,  0,  -2, -1, -1, -1, -1, 1],  # M
        'F': [-2, -3, -3, -3, -2, -3, -3, -3, -1, 0,  0,  -3, 0,  6,  -4, -2, -2, 1,  3,  -1], # F
        'P': [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7,  -1, -1, -4, -3, -2], # P
        'S': [1,  -1, 1,  0,  -1, 0,  0,  0,  -1, -2, -2, 0,  -1, -2, -1, 4,  1,  -3, -2, -2], # S
        'T': [0,  -1, 0,  -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1,  5,  -2, -2, 0],  # T
        'W': [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1,  -4, -3, -2, 11, 2,  -3], # W
        'Y': [-2, -2, -2, -3, -2, -1, -2, -3, 2,  -1, -1, -2, -1, 3,  -3, -2, -2, 2,  7,  -1], # Y
        'V': [0,  -3, -3, -3, -1, -2, -2, -3, -3, 3,  1,  -2, 1,  -1, -2, -2, 0,  -3, -1, 4],  # V
        '-': [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],  # -
    }
    encodings = []
    header = []
    for i in range(0,20):
        header.append('blosum62.F'+str(AA[i]))

    for i in fastas:
        name, sequence = i[0], i[1]
        code = np.asarray([0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0])
        for aa in sequence:
            code = code + np.asarray(blosum62[aa])
        encodings.append(list(code/len(sequence)))
    return np.array(encodings, dtype=float)

def generateGroupPairs(groupKey):
    gPair = {}
    for key1 in groupKey:
        for key2 in groupKey:
            gPair[key1 + '.' + key2] = 0
    return gPair


def CKSAAGP(fastas, gap=3, **kw):
    if gap < 0:
        print('Error: the gap should be equal or greater than zero' + '\n\n')
        return 0

    group = {
        'alphaticr': 'GAVLMI',
        'aromatic': 'FYW',
        'postivecharger': 'KRH',
        'negativecharger': 'DE',
        'uncharger': 'STCPNQ'
    }

    AA = 'ARNDCQEGHILKMFPSTWYV'

    groupKey = group.keys()

    index = {}
    for key in groupKey:
        for aa in group[key]:
            index[aa] = key

    gPairIndex = []
    for key1 in groupKey:
        for key2 in groupKey:
            gPairIndex.append(key1 + '.' + key2)

    encodings = []
    header = []
    for g in range(gap + 1):
        for p in gPairIndex:
            header.append(p + '.gap' + str(g))

    for i in fastas:
        name, sequence = i[0], re.sub('-', '', i[1])
        code = []
        for g in range(gap + 1):
            gPair = generateGroupPairs(groupKey)
            sum = 0
            for p1 in range(len(sequence)):
                p2 = p1 + g + 1
                if p2 < len(sequence) and sequence[p1] in AA and sequence[p2] in AA:
                    gPair[index[sequence[p1]] + '.' + index[sequence[p2]]] = gPair[index[sequence[p1]] + '.' + index[
                        sequence[p2]]] + 1
                    sum = sum + 1

            if sum == 0:
                for gp in gPairIndex:
                    code.append(0)
            else:
                for gp in gPairIndex:
                    code.append(gPair[gp] / sum)
        encodings.append(code)
    return np.array(encodings, dtype=float)

def CKSAAP(fastas, gap=3, **kw):
    if gap < 0:
        print('Error: the gap should be equal or greater than zero' + '\n\n')
        return 0

    AA = 'ACDEFGHIKLMNPQRSTVWY'
    encodings = []
    aaPairs = []
    for aa1 in AA:
        for aa2 in AA:
            aaPairs.append(aa1 + aa2)

    header = []
    for g in range(gap + 1):
        for aa in aaPairs:
            header.append(aa + '.gap' + str(g))

    for i in fastas:
        name, sequence = i[0], i[1]
        code = []
        for g in range(gap + 1):
            myDict = {}
            for pair in aaPairs:
                myDict[pair] = 0
            sum = 0
            for index1 in range(len(sequence)):
                index2 = index1 + g + 1
                if index1 < len(sequence) and index2 < len(sequence) and sequence[index1] in AA and sequence[
                    index2] in AA:
                    myDict[sequence[index1] + sequence[index2]] = myDict[sequence[index1] + sequence[index2]] + 1
                    sum = sum + 1
            for pair in aaPairs:
                code.append(myDict[pair] / sum)
        encodings.append(code)
    return np.array(encodings, dtype=float)

def Count(seq1, seq2):
    sum = 0
    for aa in seq1:
        sum = sum + seq2.count(aa)
    return sum

def AACPCP(fastas, **kw): 
    groups = {
        'charged':'DEKHR',
        'aliphatic':'ILV',
        'aromatic':'FHWY',
        'polar':'DERKQN',
        'neutral':'AGHPSTY',
        'hydrophobic':'CVLIMFW',
        'positively-charged':'HKR',
        'negatively-charged':'DE',
        'tiny':'ACDGST',
        'small':'EHILKMNPQV',
        'large':'FRWY'
    }


    property = (
    'charged', 'aliphatic', 'aromatic', 'polar',
    'neutral', 'hydrophobic', 'positively-charged', 'negatively-charged',
    'tiny', 'small', 'large')

    encodings = []
    header = property

    for i in fastas:
        name, sequence = i[0], re.sub('-', '', i[1])
        code = []
        for p in property:
            c = Count(groups[p], sequence) / len(sequence)
            code = code + [c]
        encodings.append(code)
    return np.array(encodings, dtype=float), list(header)

def CTDC(fastas, **kw):
    group1 = {
        'hydrophobicity_PRAM900101': 'RKEDQN',
        'hydrophobicity_ARGP820101': 'QSTNGDE',
        'hydrophobicity_ZIMJ680101': 'QNGSWTDERA',
        'hydrophobicity_PONP930101': 'KPDESNQT',
        'hydrophobicity_CASG920101': 'KDEQPSRNTG',
        'hydrophobicity_ENGD860101': 'RDKENQHYP',
        'hydrophobicity_FASG890101': 'KERSQD',
        'normwaalsvolume': 'GASTPDC',
        'polarity':        'LIFWCMVY',
        'polarizability':  'GASDT',
        'charge':          'KR',
        'secondarystruct': 'EALMQKRH',
        'solventaccess':   'ALFCGIVW'
    }
    group2 = {
        'hydrophobicity_PRAM900101': 'GASTPHY',
        'hydrophobicity_ARGP820101': 'RAHCKMV',
        'hydrophobicity_ZIMJ680101': 'HMCKV',
        'hydrophobicity_PONP930101': 'GRHA',
        'hydrophobicity_CASG920101': 'AHYMLV',
        'hydrophobicity_ENGD860101': 'SGTAW',
        'hydrophobicity_FASG890101': 'NTPG',
        'normwaalsvolume': 'NVEQIL',
        'polarity':        'PATGS',
        'polarizability':  'CPNVEQIL',
        'charge':          'ANCQGHILMFPSTWYV',
        'secondarystruct': 'VIYCWFT',
        'solventaccess':   'RKQEND'
    }
    group3 = {
        'hydrophobicity_PRAM900101': 'CLVIMFW',
        'hydrophobicity_ARGP820101': 'LYPFIW',
        'hydrophobicity_ZIMJ680101': 'LPFYI',
        'hydrophobicity_PONP930101': 'YMFWLCVI',
        'hydrophobicity_CASG920101': 'FIWC',
        'hydrophobicity_ENGD860101': 'CVLIMF',
        'hydrophobicity_FASG890101': 'AYHWVMFLIC',
        'normwaalsvolume': 'MHKFRYW',
        'polarity':        'HQRKNED',
        'polarizability':  'KMHFRYW',
        'charge':          'DE',
        'secondarystruct': 'GNPSD',
        'solventaccess':   'MSPTHY'
    }

    groups = [group1, group2, group3]
    property = (
    'hydrophobicity_PRAM900101', 'hydrophobicity_ARGP820101', 'hydrophobicity_ZIMJ680101', 'hydrophobicity_PONP930101',
    'hydrophobicity_CASG920101', 'hydrophobicity_ENGD860101', 'hydrophobicity_FASG890101', 'normwaalsvolume',
    'polarity', 'polarizability', 'charge', 'secondarystruct', 'solventaccess')

    encodings = []
    header = []
    for p in property:
        for g in range(1, len(groups) + 1):
            header.append(p + '.G' + str(g))

    for i in fastas:
        name, sequence = i[0], re.sub('-', '', i[1])
        code = []
        for p in property:
            c1 = Count(group1[p], sequence) / len(sequence)
            c2 = Count(group2[p], sequence) / len(sequence)
            c3 = 1 - c1 - c2
            code = code + [c1, c2, c3]
        encodings.append(code)
    return np.array(encodings, dtype=float)

def Count2(aaSet, sequence):
    number = 0
    for aa in sequence:
        if aa in aaSet:
            number = number + 1
    cutoffNums = [1, math.floor(0.25 * number), math.floor(0.50 * number), math.floor(0.75 * number), number]
    cutoffNums = [i if i >=1 else 1 for i in cutoffNums]

    code = []
    for cutoff in cutoffNums:
        myCount = 0
        for i in range(len(sequence)):
            if sequence[i] in aaSet:
                myCount += 1
                if myCount == cutoff:
                    code.append((i + 1) / len(sequence) * 100)
                    break
        if myCount == 0:
            code.append(0)
    return code


def CTDD(fastas, **kw):
    group1 = {
        'hydrophobicity_PRAM900101': 'RKEDQN',
        'hydrophobicity_ARGP820101': 'QSTNGDE',
        'hydrophobicity_ZIMJ680101': 'QNGSWTDERA',
        'hydrophobicity_PONP930101': 'KPDESNQT',
        'hydrophobicity_CASG920101': 'KDEQPSRNTG',
        'hydrophobicity_ENGD860101': 'RDKENQHYP',
        'hydrophobicity_FASG890101': 'KERSQD',
        'normwaalsvolume': 'GASTPDC',
        'polarity':        'LIFWCMVY',
        'polarizability':  'GASDT',
        'charge':          'KR',
        'secondarystruct': 'EALMQKRH',
        'solventaccess':   'ALFCGIVW'
    }
    group2 = {
        'hydrophobicity_PRAM900101': 'GASTPHY',
        'hydrophobicity_ARGP820101': 'RAHCKMV',
        'hydrophobicity_ZIMJ680101': 'HMCKV',
        'hydrophobicity_PONP930101': 'GRHA',
        'hydrophobicity_CASG920101': 'AHYMLV',
        'hydrophobicity_ENGD860101': 'SGTAW',
        'hydrophobicity_FASG890101': 'NTPG',
        'normwaalsvolume': 'NVEQIL',
        'polarity':        'PATGS',
        'polarizability':  'CPNVEQIL',
        'charge':          'ANCQGHILMFPSTWYV',
        'secondarystruct': 'VIYCWFT',
        'solventaccess':   'RKQEND'
    }
    group3 = {
        'hydrophobicity_PRAM900101': 'CLVIMFW',
        'hydrophobicity_ARGP820101': 'LYPFIW',
        'hydrophobicity_ZIMJ680101': 'LPFYI',
        'hydrophobicity_PONP930101': 'YMFWLCVI',
        'hydrophobicity_CASG920101': 'FIWC',
        'hydrophobicity_ENGD860101': 'CVLIMF',
        'hydrophobicity_FASG890101': 'AYHWVMFLIC',
        'normwaalsvolume': 'MHKFRYW',
        'polarity':        'HQRKNED',
        'polarizability':  'KMHFRYW',
        'charge':          'DE',
        'secondarystruct': 'GNPSD',
        'solventaccess':   'MSPTHY'
    }

    groups = [group1, group2, group3]
    property = (
    'hydrophobicity_PRAM900101', 'hydrophobicity_ARGP820101', 'hydrophobicity_ZIMJ680101', 'hydrophobicity_PONP930101',
    'hydrophobicity_CASG920101', 'hydrophobicity_ENGD860101', 'hydrophobicity_FASG890101', 'normwaalsvolume',
    'polarity', 'polarizability', 'charge', 'secondarystruct', 'solventaccess')


    encodings = []
    header = []
    for p in property:
        for g in ('1', '2', '3'):
            for d in ['0', '25', '50', '75', '100']:
                header.append(p + '.' + g + '.residue' + d)

    for i in fastas:
        name, sequence  = i[0], re.sub('-', '', i[1])
        code = []
        for p in property:
            code = code + Count2(group1[p], sequence) + Count2(group2[p], sequence) + Count2(group3[p], sequence)
        encodings.append(code)
    return np.array(encodings, dtype=float)

def CTDT(fastas, **kw):
    group1 = {
        'hydrophobicity_PRAM900101': 'RKEDQN',
        'hydrophobicity_ARGP820101': 'QSTNGDE',
        'hydrophobicity_ZIMJ680101': 'QNGSWTDERA',
        'hydrophobicity_PONP930101': 'KPDESNQT',
        'hydrophobicity_CASG920101': 'KDEQPSRNTG',
        'hydrophobicity_ENGD860101': 'RDKENQHYP',
        'hydrophobicity_FASG890101': 'KERSQD',
        'normwaalsvolume': 'GASTPDC',
        'polarity':        'LIFWCMVY',
        'polarizability':  'GASDT',
        'charge':          'KR',
        'secondarystruct': 'EALMQKRH',
        'solventaccess':   'ALFCGIVW'
    }
    group2 = {
        'hydrophobicity_PRAM900101': 'GASTPHY',
        'hydrophobicity_ARGP820101': 'RAHCKMV',
        'hydrophobicity_ZIMJ680101': 'HMCKV',
        'hydrophobicity_PONP930101': 'GRHA',
        'hydrophobicity_CASG920101': 'AHYMLV',
        'hydrophobicity_ENGD860101': 'SGTAW',
        'hydrophobicity_FASG890101': 'NTPG',
        'normwaalsvolume': 'NVEQIL',
        'polarity':        'PATGS',
        'polarizability':  'CPNVEQIL',
        'charge':          'ANCQGHILMFPSTWYV',
        'secondarystruct': 'VIYCWFT',
        'solventaccess':   'RKQEND'
    }
    group3 = {
        'hydrophobicity_PRAM900101': 'CLVIMFW',
        'hydrophobicity_ARGP820101': 'LYPFIW',
        'hydrophobicity_ZIMJ680101': 'LPFYI',
        'hydrophobicity_PONP930101': 'YMFWLCVI',
        'hydrophobicity_CASG920101': 'FIWC',
        'hydrophobicity_ENGD860101': 'CVLIMF',
        'hydrophobicity_FASG890101': 'AYHWVMFLIC',
        'normwaalsvolume': 'MHKFRYW',
        'polarity':        'HQRKNED',
        'polarizability':  'KMHFRYW',
        'charge':          'DE',
        'secondarystruct': 'GNPSD',
        'solventaccess':   'MSPTHY'
    }

    groups = [group1, group2, group3]
    property = (
    'hydrophobicity_PRAM900101', 'hydrophobicity_ARGP820101', 'hydrophobicity_ZIMJ680101', 'hydrophobicity_PONP930101',
    'hydrophobicity_CASG920101', 'hydrophobicity_ENGD860101', 'hydrophobicity_FASG890101', 'normwaalsvolume',
    'polarity', 'polarizability', 'charge', 'secondarystruct', 'solventaccess')

    encodings = []
    header = []
    for p in property:
        for tr in ('Tr1221', 'Tr1331', 'Tr2332'):
            header.append(p + '.' + tr)

    for i in fastas:
        name, sequence = i[0], re.sub('-', '', i[1])
        code = []
        aaPair = [sequence[j:j + 2] for j in range(len(sequence) - 1)]
        for p in property:
            c1221, c1331, c2332 = 0, 0, 0
            for pair in aaPair:
                if (pair[0] in group1[p] and pair[1] in group2[p]) or (pair[0] in group2[p] and pair[1] in group1[p]):
                    c1221 = c1221 + 1
                    continue
                if (pair[0] in group1[p] and pair[1] in group3[p]) or (pair[0] in group3[p] and pair[1] in group1[p]):
                    c1331 = c1331 + 1
                    continue
                if (pair[0] in group2[p] and pair[1] in group3[p]) or (pair[0] in group3[p] and pair[1] in group2[p]):
                    c2332 = c2332 + 1
            code = code + [c1221/len(aaPair), c1331/len(aaPair), c2332/len(aaPair)]
        encodings.append(code)
    return np.array(encodings, dtype=float)

def CalculateKSCTriad(sequence, gap, features, AADict):
    res = []
    for g in range(gap + 1):
        myDict = {}
        for f in features:
            myDict[f] = 0

        for i in range(len(sequence)):
            if i + g + 1 < len(sequence) and i + 2 * g + 2 < len(sequence):
                fea = AADict[sequence[i]] + '.' + AADict[sequence[i + g + 1]] + '.' + AADict[
                    sequence[i + 2 * g + 2]]
                myDict[fea] = myDict[fea] + 1

        maxValue, minValue = max(myDict.values()), min(myDict.values())
        for f in features:
            res.append((myDict[f] - minValue) / maxValue)

    return res

def CTriad(fastas, gap = 0, **kw):
    AAGroup = {
        'g1': 'AGV',
        'g2': 'ILFP',
        'g3': 'YMTS',
        'g4': 'HNQW',
        'g5': 'RK',
        'g6': 'DE',
        'g7': 'C'
    }

    myGroups = sorted(AAGroup.keys())

    AADict = {}
    for g in myGroups:
        for aa in AAGroup[g]:
            AADict[aa] = g

    features = [f1 + '.'+ f2 + '.' + f3 for f1 in myGroups for f2 in myGroups for f3 in myGroups]

    encodings = []
    header = []
    for f in features:
        header.append(f)

    for i in fastas:
        name, sequence = i[0], re.sub('-', '', i[1])
        code = []
        if len(sequence) < 3:
            print('Error: for "CTriad" encoding, the input fasta sequences should be greater than 3. \n\n')
            return 0
        code = code + CalculateKSCTriad(sequence, 0, features, AADict)
        encodings.append(code)

    return np.array(encodings, dtype=float)

def DDE(fastas, **kw):
    AA = 'ACDEFGHIKLMNPQRSTVWY'

    myCodons = {
        'A': 4,
        'C': 2,
        'D': 2,
        'E': 2,
        'F': 2,
        'G': 4,
        'H': 2,
        'I': 3,
        'K': 2,
        'L': 6,
        'M': 1,
        'N': 2,
        'P': 4,
        'Q': 2,
        'R': 6,
        'S': 6,
        'T': 4,
        'V': 4,
        'W': 1,
        'Y': 2
    }

    encodings = []
    diPeptides = ['DDE_'+aa1 + aa2 for aa1 in AA for aa2 in AA]
    header = [] + diPeptides


    myTM = []
    for pair in diPeptides:
        myTM.append((myCodons[pair[0]] / 61) * (myCodons[pair[1]] / 61))

    AADict = {}
    for i in range(len(AA)):
        AADict[AA[i]] = i

    for i in fastas:
        name, sequence = i[0], re.sub('-', '', i[1])
        code = []
        tmpCode = [0] * 400
        for j in range(len(sequence) - 2 + 1):
            tmpCode[AADict[sequence[j]] * 20 + AADict[sequence[j+1]]] = tmpCode[AADict[sequence[j]] * 20 + AADict[sequence[j+1]]] +1
        if sum(tmpCode) != 0:
            tmpCode = [i/sum(tmpCode) for i in tmpCode]

        myTV = []
        for j in range(len(myTM)):
            myTV.append(myTM[j] * (1-myTM[j]) / (len(sequence) - 1))

        for j in range(len(tmpCode)):
            tmpCode[j] = (tmpCode[j] - myTM[j]) / math.sqrt(myTV[j])

        code = code + tmpCode
        encodings.append(code)
    return np.array(encodings, dtype=float)

# Disorder (Disorder) Protein disorder information was first predicted using external VSL2
# DisorderB also required external program VSL2
# DisorderC also required external program VSL2

def DPC(fastas, gap, **kw):
    AA = 'ACDEFGHIKLMNPQRSTVWY'
    encodings = []
    diPeptides = [aa1 + aa2 for aa1 in AA for aa2 in AA]
    header = [] + diPeptides

    AADict = {}
    for i in range(len(AA)):
        AADict[AA[i]] = i

    for i in fastas:
        name, sequence = i[0], re.sub('-', '', i[1])
        code = []
        tmpCode = [0] * 400
        for j in range(len(sequence) - 2 + 1 - gap):
            tmpCode[AADict[sequence[j]] * 20 + AADict[sequence[j+gap+1]]] = tmpCode[AADict[sequence[j]] * 20 + AADict[sequence[j+gap+1]]] +1
        if sum(tmpCode) != 0:
            tmpCode = [i/sum(tmpCode) for i in tmpCode]
        code = code + tmpCode
        encodings.append(code)
    return np.array(encodings, dtype=float)

def EAAC(fastas, window=5, **kw):
    AA ='ACDEFGHIKLMNPQRSTVWY'
    #AA = 'ARNDCQEGHILKMFPSTWYV'
    encodings = []
    header = []
    for aa in AA:
        header.append('EACC.'+aa)

    for i in fastas:
        name, sequence = i[0], i[1]
        code = []
        for aa in AA:
            tmp = 0
            for j in range(len(sequence)):
                if j < len(sequence) and j + window <= len(sequence):
                    count = Counter(sequence[j:j+window])
                    for key in count:
                        count[key] = count[key] / len(sequence[j:j+window])
                    tmp = tmp + count[aa]
            code.append(tmp/len(sequence))
        encodings.append(code)
    return np.array(encodings, dtype=float)

def EGAAC(fastas, window=5, **kw):
    group = {
        'alphaticr': 'GAVLMI',
        'aromatic': 'FYW',
        'postivecharger': 'KRH',
        'negativecharger': 'DE',
        'uncharger': 'STCPNQ'
    }
    groupKey = group.keys()
    encodings = []
    header = []
    for w in range(1, len(fastas[0][1]) - window + 2):
        for g in groupKey:
            header.append('SW.' + str(w) + '.' + g)

    for i in fastas:
        name, sequence = i[0], i[1]
        code = []
        for key in groupKey:
            tmp=0
            for j in range(len(sequence)):
                if j + window <= len(sequence):
                    count = Counter(sequence[j:j + window])
                    myDict = {}
                    #for key in groupKey:
                    for aa in group[key]:
                        myDict[key] = myDict.get(key, 0) + count[aa]
                            
    
                    #for key in groupKey:
                    tmp = tmp + (myDict[key] / window)
            code.append(tmp/len(sequence))
        encodings.append(code)
    return np.array(encodings, dtype=float)

def Rvalue(aa1, aa2, AADict, Matrix):
    return sum([(Matrix[i][AADict[aa1]] - Matrix[i][AADict[aa2]]) ** 2 for i in range(len(Matrix))]) / len(Matrix)


def PAAC(fastas, lambdaValue=1, w=0.05, **kw):
    records = []
    records.append("#	A	R	N	D	C	Q	E	G	H	I	L	K	M	F	P	S	T	W	Y	V")
    records.append("Hydrophobicity	0.62	-2.53	-0.78	-0.9	0.29	-0.85	-0.74	0.48	-0.4	1.38	1.06	-1.5	0.64	1.19	0.12	-0.18	-0.05	0.81	0.26	1.08")
    records.append("Hydrophilicity	-0.5	3	0.2	3	-1	0.2	3	0	-0.5	-1.8	-1.8	3	-1.3	-2.5	0	0.3	-0.4	-3.4	-2.3	-1.5")
    records.append("SideChainMass	15	101	58	59	47	72	73	1	82	57	57	73	75	91	42	31	45	130	107	43")
    AA = ''.join(records[0].rstrip().split()[1:])
    AADict = {}
    for i in range(len(AA)):
        AADict[AA[i]] = i
    AAProperty = []
    AAPropertyNames = []
    for i in range(1, len(records)):
        array = records[i].rstrip().split() if records[i].rstrip() != '' else None
        AAProperty.append([float(j) for j in array[1:]])
        AAPropertyNames.append(array[0])

    AAProperty1 = []
    for i in AAProperty:
        meanI = sum(i) / 20
        fenmu = math.sqrt(sum([(j - meanI) ** 2 for j in i]) / 20)
        AAProperty1.append([(j - meanI) / fenmu for j in i])

    encodings = []
    header = []
    for aa in AA:
        header.append('Xc1.' + aa)
    for n in range(1, lambdaValue + 1):
        header.append('Xc2.lambda' + str(n))

    for i in fastas:
        name, sequence= i[0], re.sub('-', '', i[1])
        code = []
        theta = []
        for n in range(1, lambdaValue + 1):
            theta.append(
                sum([Rvalue(sequence[j], sequence[j + n], AADict, AAProperty1) for j in range(len(sequence) - n)]) / (
                    len(sequence) - n))
        myDict = {}
        for aa in AA:
            myDict[aa] = sequence.count(aa)
        code = code + [myDict[aa] / (1 + w * sum(theta)) for aa in AA]
        code = code + [(w * j) / (1 + w * sum(theta)) for j in theta]
        encodings.append(code)
    return np.array(encodings, dtype=float)

def TPC(fastas, **kw):
   # AA = kw['order'] if kw['order'] != None else 'ACDEFGHIKLMNPQRSTVWY'
    AA = 'ACDEFGHIKLMNPQRSTVWY'
    encodings = []
    #triPeptides = [aa1 + aa2 + aa3 for aa1 in AA for aa2 in AA for aa3 in AA]
    #header = ['#', 'label'] + triPeptides
    #encodings.append(header)

    AADict = {}
    for i in range(len(AA)):
        AADict[AA[i]] = i

    for i in fastas:
        name, sequence = i[0], re.sub('-', '', i[1])
        code = []
        tmpCode = [0] * 8000
        for j in range(len(sequence) - 3 + 1):
            tmpCode[AADict[sequence[j]] * 400 + AADict[sequence[j+1]]*20 + AADict[sequence[j+2]]] = tmpCode[AADict[sequence[j]] * 400 + AADict[sequence[j+1]]*20 + AADict[sequence[j+2]]] +1
        if sum(tmpCode) != 0:
            tmpCode = [i/sum(tmpCode) for i in tmpCode]
        code = code + tmpCode
        encodings.append(code)
    return np.array(encodings, dtype=float)

def reducedACID(seq):
    def fcount(string, substr):
       count = 0
       pos = 0
       while(True):
           pos = string.find(substr , pos)
           if pos > -1:
               count = count + 1
               pos += 1
           else:
               break
       return count

    for count, fasta in enumerate(seq):
        sub = "akn"
        subsub = [it1+it2 for it1 in sub for it2 in sub] 
        aalist = "ACDEFGHIKLMNPQRSTVWY"
        aasub = {}
        aasub["a"] = "DE"
        aasub["k"] = "KHR"
        aasub["n"] = "ACFGILMNPQSTVWY"
        
        seq1 = fasta[1]
        lenn=len(seq1)
        seq2 = seq1
        for key, value in aasub.items():
            for aa in value:
                seq2 = seq2.replace(aa,key)
        
        freq2 = {}
        for item in sub:
            freq2[item] = fcount(seq2, item)
        for item in subsub:
            freq2[item] = fcount(seq2, item)
            
        freq1 = {}
        for item in aalist:
            freq1[item] = fcount(seq1, item)
            
        feat = []
        for key, value in aasub.items():
            feat.append(freq2[key]/lenn)
            
        for item in aalist:
            for key, value in aasub.items():
                if item in value:
                    feat.append(freq1[item]/max(1,freq2[key]))
                    break
                    
        for item in subsub:
            feat.append(freq2[item]/(freq2[item[0]]+1))
        
        feat = np.array(feat)
        feat = feat.reshape(1,len(feat))
        if count == 0:
            allfeat = feat
        else:
            allfeat = np.vstack((allfeat, feat))
            
    return allfeat

def reducedPOLAR(seq):
    def fcount(string, substr):
       count = 0
       pos = 0
       while(True):
           pos = string.find(substr , pos)
           if pos > -1:
               count = count + 1
               pos += 1
           else:
               break
       return count

    for count, fasta in enumerate(seq):
        sub = "qwert"
        subsub = [it1+it2 for it1 in sub for it2 in sub] 
        aalist = "ACDEFGHIKLMNPQRSTVWY"
        aasub = {}
        aasub["q"] = "DE"
        aasub["w"] = "RHK"
        aasub["e"] = "WYF"
        aasub["r"] = "SCMNQT"
        aasub["t"] = "GAVLIP"
        
        seq1 = fasta[1]
        lenn=len(seq1)
        seq2 = seq1
        for key, value in aasub.items():
            for aa in value:
                seq2 = seq2.replace(aa,key)
        
        freq2 = {}
        for item in sub:
            freq2[item] = fcount(seq2, item)
        for item in subsub:
            freq2[item] = fcount(seq2, item)
            
        freq1 = {}
        for item in aalist:
            freq1[item] = fcount(seq1, item)
            
        feat = []
        for key, value in aasub.items():
            feat.append(freq2[key]/lenn)
            
        for item in aalist:
            for key, value in aasub.items():
                if item in value:
                    feat.append(freq1[item]/max(1,freq2[key]))
                    break
                    
        for item in subsub:
            feat.append(freq2[item]/(freq2[item[0]]+1))
        
        feat = np.array(feat)
        feat = feat.reshape(1,len(feat))
        if count == 0:
            allfeat = feat
        else:
            allfeat = np.vstack((allfeat, feat))
            
    return allfeat

def reducedSECOND(seq):
    def fcount(string, substr):
       count = 0
       pos = 0
       while(True):
           pos = string.find(substr , pos)
           if pos > -1:
               count = count + 1
               pos += 1
           else:
               break
       return count

    for count, fasta in enumerate(seq):
        sub = "qwe"
        subsub = [it1+it2 for it1 in sub for it2 in sub] 
        aalist = "ACDEFGHIKLMNPQRSTVWY"
        aasub = {}
        aasub["q"] = "EHALMQKR"
        aasub["w"] = "VTIYCWF"
        aasub["e"] = "GDNPS"
        
        seq1 = fasta[1]
        lenn=len(seq1)
        seq2 = seq1
        for key, value in aasub.items():
            for aa in value:
                seq2 = seq2.replace(aa,key)
        
        freq2 = {}
        for item in sub:
            freq2[item] = fcount(seq2, item)
        for item in subsub:
            freq2[item] = fcount(seq2, item)
            
        freq1 = {}
        for item in aalist:
            freq1[item] = fcount(seq1, item)
            
        feat = []
        for key, value in aasub.items():
            feat.append(freq2[key]/lenn)
            
        for item in aalist:
            for key, value in aasub.items():
                if item in value:
                    feat.append(freq1[item]/max(1,freq2[key]))
                    break
                    
        for item in subsub:
            feat.append(freq2[item]/(freq2[item[0]]+1))
        
        feat = np.array(feat)
        feat = feat.reshape(1,len(feat))
        if count == 0:
            allfeat = feat
        else:
            allfeat = np.vstack((allfeat, feat))
            
    return allfeat

def reducedCHARGE(seq):
    def fcount(string, substr):
       count = 0
       pos = 0
       while(True):
           pos = string.find(substr , pos)
           if pos > -1:
               count = count + 1
               pos += 1
           else:
               break
       return count

    for count, fasta in enumerate(seq):
        sub = "qwe"
        subsub = [it1+it2 for it1 in sub for it2 in sub] 
        aalist = "ACDEFGHIKLMNPQRSTVWY"
        aasub = {}
        aasub["q"] = "KR"
        aasub["w"] = "AVNCQGHILMFPSTWY"
        aasub["e"] = "DE"
        
        seq1 = fasta[1]
        lenn=len(seq1)
        seq2 = seq1
        for key, value in aasub.items():
            for aa in value:
                seq2 = seq2.replace(aa,key)
        
        freq2 = {}
        for item in sub:
            freq2[item] = fcount(seq2, item)
        for item in subsub:
            freq2[item] = fcount(seq2, item)
            
        freq1 = {}
        for item in aalist:
            freq1[item] = fcount(seq1, item)
            
        feat = []
        for key, value in aasub.items():
            feat.append(freq2[key]/lenn)
            
        for item in aalist:
            for key, value in aasub.items():
                if item in value:
                    feat.append(freq1[item]/max(1,freq2[key]))
                    break
                    
        for item in subsub:
            feat.append(freq2[item]/(freq2[item[0]]+1))
        
        feat = np.array(feat)
        feat = feat.reshape(1,len(feat))
        if count == 0:
            allfeat = feat
        else:
            allfeat = np.vstack((allfeat, feat))
            
    return allfeat

def reducedDHP(seq):
    def fcount(string, substr):
       count = 0
       pos = 0
       while(True):
           pos = string.find(substr , pos)
           if pos > -1:
               count = count + 1
               pos += 1
           else:
               break
       return count

    for count, fasta in enumerate(seq):
        sub = "qwer"
        subsub = [it1+it2 for it1 in sub for it2 in sub] 
        aalist = "ACDEFGHIKLMNPQRSTVWY"
        aasub = {} 
        aasub["q"] = "PALVIFWM"
        aasub["w"] = "QSTYCNG"
        aasub["e"] = "HKR"
        aasub["r"] = "DE"
        
        seq1 = fasta[1]
        lenn=len(seq1)
        seq2 = seq1
        for key, value in aasub.items():
            for aa in value:
                seq2 = seq2.replace(aa,key)
        
        freq2 = {}
        for item in sub:
            freq2[item] = fcount(seq2, item)
        for item in subsub:
            freq2[item] = fcount(seq2, item)
            
        freq1 = {}
        for item in aalist:
            freq1[item] = fcount(seq1, item)
            
        feat = []
        for key, value in aasub.items():
            feat.append(freq2[key]/lenn)
            
        for item in aalist:
            for key, value in aasub.items():
                if item in value:
                    feat.append(freq1[item]/max(1,freq2[key]))
                    break
                    
        for item in subsub:
            feat.append(freq2[item]/(freq2[item[0]]+1))
        
        feat = np.array(feat)
        feat = feat.reshape(1,len(feat))
        if count == 0:
            allfeat = feat
        else:
            allfeat = np.vstack((allfeat, feat))
            
    return allfeat

kw=['ACDEFGHIKLMNPQRSTVWY']
header = []
fasta = read_protein_sequences('Training_Pos.fasta')

feat1 = AAC(fasta)
data_csv = pd.DataFrame(data=feat1)
data_csv.to_csv('AAC.csv')
feat2 = APAAC(fasta, 0)
data_csv = pd.DataFrame(data=feat2)
data_csv.to_csv('APAAC.csv')
feat3= PAAC(fasta)
data_csv = pd.DataFrame(data=feat3)
data_csv.to_csv('PAAC.csv')
feat4 = AAINDEX(fasta)
data_csv = pd.DataFrame(data=feat4)
data_csv.to_csv('AAINDEX.csv')
feat5 = EAAC(fasta)
data_csv = pd.DataFrame(data=feat5)
data_csv.to_csv('EAAC.csv')
feat6 = APAAC(fasta)
data_csv = pd.DataFrame(data=feat6)
data_csv.to_csv('APAAC.csv')
