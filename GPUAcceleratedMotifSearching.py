"""
This program takes plain algorithms programmed previously (found in UnitedMotifSearchUtils.py) and expands upon them
by vectorising the functions and running them in parallel on CUDA capable GPUs or hardware accelerators.
"""

## Imports
import os
import sys
import time
import numba
import numpy as np
import re


# Conversion dictionaries for converting nucleotides to numbers and numbers to nucleotides. The values are of importance
# and the dictionaries should mirror each other's inverse of key-value pairs.
conversionDict = {"A": 1, "T": 2, "G": 3, "C": 4}
unConversionDict = {1: "A", 2: "T", 3: "G", 4: "C"}

# Load data as plaintext and save all sequences into a separate file (same as in normal program)
# This is relatively fast compared to the rest of the algorithms, and I'm a little bit too lazy to change already
# proven to work code. If anyone wishes to optimize it, go for it, just make sure it works as intended :)
def readSequences(linestart: int, linestop: int, all=True):
    """
    Compiles sequences in the appropriate folder into one file for easy access and returns a list of strings,
    where each element is a one-line representative of the gene data from the FASTA files taken in order from the folder.

    :param linestart: Where to start reading the files from, set 0 if reading all
    :param linestop: Where to stop readinf from files at, set 0 if reading all
    :param all: Whether to read entire sequences or not (Default: True)
    :return: List of strings, each string is the genetic sequence from files in the appropriate folder in order.
    """
    if not os.path.isfile('./compiled_sequences.fasta'):
        if not os.path.isdir('./sequences'):
            print("There seems to not be a 'sequences' folder present in current directory."
                  "This directory is essential for the program to work. It will be now created."
                  "Please move all sequence FASTA files to be worked on into this directory and run the program again")
            os.mkdir('./sequences')
            print('The program will stop in 5 seconds.')
            time.sleep(5)
            if sys.platform == 'win32':
                os.system('pause')
            elif sys.platform == 'linux' or sys.platform == 'linux2' or sys.platform == 'darwin':
                os.system('read -n1 -r -p "Press any key to continue..."')
        else:
            if not all:
                with open("./compiled_sequences.fasta", 'w') as f:
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
        try:
            while True:
                choice = input("There are sequences that were compiled previously. Load from them? [Y/N]\n?>> ")
                if choice in ["Y", 'y', '']:
                    with open('./compiled_sequences.fasta', 'rt') as f:
                        DNA = f.readlines()
                    break
                elif choice in ["N", 'n']:
                    if not all:
                        with open("./compiled_sequences.fasta", 'w') as f:
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
                        break
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
                        break
                else:
                    raise ValueError
        except ValueError:
            print("The provided answer is invalid, try Y or N.\n")
    return DNA

# Vectorise the sequences read as plaintext
def vectoriseSequences(k: int, sequences: list):
    """
    Takes a list of sequences and returns an array of arrays, containing k-length pieces of sequences from beginning to
    end shifting by one nucleotide. Nucleotides are encoded numerically as follows:

    A -> 0
    T -> 1
    G -> 2
    C -> 3

    This is cruicial information for decoding the outcome sequences for further processing.

    :param k: (Integer) Defines the length of a k-mer in vectorised sequences
    :param sequences: (List) The list of sequences to vectorise
    :return: An array of arrays containing k-length numerically encoded pieces of given sequences in appropriate order.
    """

    numDNA = []
    for sequence in sequences:
        numSequence = []
        for nucleotide in sequence.strip('\n'):
            numSequence.append(conversionDict[nucleotide])
        numDNA.append(np.array(numSequence))
    numDNA = np.array(numDNA)

    return np.array([[numSequence[i:i+k] for i in range(len(numSequence)-k)] for numSequence in numDNA])

def vectorEnumerateMotifs(vectorisedSequences, d):
    pass


if __name__ == '__main__':
    DNA = readSequences(0, 0)
    vDNA = vectoriseSequences(15, DNA)
    print("Normal sequence:", DNA, len(DNA[0]))
    print("Vectorised sequence:", vDNA, len(vDNA), len(vDNA[0]), len(vDNA[0][0]))
