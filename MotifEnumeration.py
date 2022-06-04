"""
Jakub Susoł 274300
Zadanie 1 zaliczenie
Algorytmika

Algorytm MotifEnumeration
(strona 71 Bioinformatics Algorithms - An Active Learning Approach vol. I)
"""
import os
import sys
import json
import time


def readSequences(linestart, linestop, all=False):
    if not os.path.isfile('./compiled_sequences.fasta'):
        if not os.path.isdir('./sequences'):
            print("There seems to not be a 'sequences' folder present in current directory."
                  "This directory is essential for the program to work. It will be now created."
                  "Please move all sequence FASTA files to be worked on into this directory and run the program again")
            os.mkdir('./sequences')
            print('The program will stop in 10 seconds.')
            time.sleep(10)
            if sys.platform == 'win32':
                os.system('pause')
            elif sys.platform == 'linux' or sys.platform == 'linux2' or sys.platform == 'darwin':
                os.system('read -n1 -r -p "Press any key to continue..."')
        else:
            if not all:
                with open("compiled_sequences.fasta", 'w') as f:
                    for sequence in os.listdir('./sequences'):
                        with open(f'./sequences/{sequence}', 'rt') as s:
                            readseq = ''
                            for line in s.readlines()[linestart:linestop]:
                                if not line.startswith('>'):
                                    readseq += line.strip('\n')
                            readseq += '\n'
                            f.writelines(readseq)
                with open('./compiled_sequences.fasta', 'rt') as f:
                    DNA = f.readlines()
            else:
                with open("compiled_sequences.fasta", 'w') as f:
                    for sequence in os.listdir('./sequences'):
                        with open(f'./sequences/{sequence}', 'rt') as s:
                            readseq = ''
                            for line in s.readlines():
                                if not line.startswith('>'):
                                    readseq += line.strip('\n')
                            readseq += '\n'
                            f.writelines(readseq)
                with open('./compiled_sequences.fasta', 'rt') as f:
                    DNA = f.readlines()
    else:
        with open('./compiled_sequences.fasta', 'rt') as f:
            DNA = f.readlines()
    return DNA


# K-Mer is a (k,d)-motif if it appears in every line of DNA with at most d mismatches
def motifEnumeration(dna: list, k: int, d: int):
    patterns = dict()  # new patterns set
    refSeq = dna[0]
    patternExistsIn = dict()
    for compDna in dna[1::]:
        # for each k-mer pattern in dna
        for i in range(len(refSeq) - k):
            pattern = refSeq[i:i + k]
            patternExistsIn[pattern] = [(0, pattern)]
            print(f'Comparing {i + 1}/{len(refSeq) - k} in {dna.index(compDna)} of {len(dna) - 1}')
            # for each pattern' differing from pattern by at most d mismatches
            for j in range(len(compDna) - k):
                patternPrime = compDna[j:j + k]
                # check mismatches
                mismatches = 0
                for a, b in zip(pattern, patternPrime):
                    if a == b:
                        pass
                    else:
                        mismatches += 1
                # add index of DNA string the pattern' exists in and the pattern
                if mismatches <= d:
                    patternExistsIn[pattern].append((dna.index(compDna), patternPrime))
    # if pattern' exists in all lines of DNA
    for key, value in patternExistsIn.items():
        if len(value) == len(dna):
            patternlist = []
            for item in value:
                patternlist.append(item[1])
            # add pattern to patterns
            patterns[value[0][1]] = patternlist
    return patterns


def createJSON(DNA, k, d):
    motifdict = motifEnumeration(DNA, k, d)
    if not os.path.isdir('./motifs'):
        os.mkdir('./motifs')
    with open(f'./motifs/({k},{d})-motifs.json', 'w') as m:
        json.dump(motifdict, m)
    createdFile = f'({k},{d})-motifs.json'
    return motifdict, createdFile


if __name__ == '__main__':
    DNA = readSequences(0, 76, False)
    createJSON(DNA, 12, 3)
