"""
This program takes plain algorithms programmed previously (found in UnitedMotifSearchUtils.py) and expands upon them
by vectorising the functions and running them in parallel on CUDA capable GPUs or hardware accelerators.
"""

## Imports
import os
import json
import sys
import time
import numpy as np
from numpy.typing import NDArray
from multiprocessing import Process, Pool, set_start_method

# Conversion dictionaries for converting nucleotides to numbers and numbers to nucleotides. The values are of importance
# and the dictionaries should mirror each other's inverse of key-value pairs.
conversionDict = {"A": 1, "T": 2, "G": 3, "C": 4, "N": 0}
unConversionDict = {1: "A", 2: "T", 3: "G", 4: "C", 0: "N"}


# Load data as plaintext and save all sequences into a separate file (same as in normal program)
# This is relatively fast compared to the rest of the algorithms, and I'm a little too lazy to change already
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

    A -> 1
    T -> 2
    G -> 3
    C -> 4

    This is crucial information for decoding the outcome sequences for further processing.

    :param k: (Integer) Defines the length of a k-mer in vectorised sequences
    :param sequences: (List) The list of sequences to vectorise
    :return: An array of arrays containing k-length numerically encoded pieces of given sequences in appropriate order.
    """

    numDNA = [] # Define starting list for vectorisation
    for sequence in sequences: # For each sequence in sequences
        numSequence = [] # Define a temporary list for appending conversion outcome
        for nucleotide in sequence.strip('\n'): # For each nucleotide in given sequence without newlines
            numSequence.append(conversionDict[nucleotide]) # Append the converted nucleotide number to temporary list
        numDNA.append(np.array(numSequence)) # Change the list to NumPy Array
    numDNA = np.array(numDNA) # Change whole converted sequences list to an array
    # EG. k=5 ['ATGCCGTAGTTAGGACT'] -> [1, 2, 3, 4, 4, 3, 2, 1, 3, 2, 2, 1, 3, 3, 1, 4, 2]

    # Return an array containing fragmented k-length pieces of sequences in appropriate order
    # EG. k=5 [1, 2, 3, 4, 4, 3, 2, 1, 3, 2, 2, 1, 3, 3, 1, 4, 2] ->
    # [
    #  [1, 2, 3, 4, 4],
    #  [2, 3, 4, 4, 3], [3, 4, 4, 3, 2], [4, 4, 3, 2, 1], [4, 3, 2, 1, 3], [3, 2, 1, 3, 2], [2, 1, 3, 2, 2],
    #  [1, 3, 2, 2, 1], [3, 2, 2, 1, 3], [2, 2, 1, 3, 3], [2, 1, 3, 3, 1], [1, 3, 3, 1, 4], [3, 3, 1, 4, 2]
    # ]
    return np.array([[numSequence[i:i + k] for i in range(len(numSequence) - (k-1))] for numSequence in numDNA])


def unvectorise(vectorToConvert): # Reverse the process of vectorisation for given vector
    return ''.join([unConversionDict[nucleotide] for nucleotide in vectorToConvert])


def makeSearchPatterns(refSeq: NDArray, splits: int):
    '''
    Return a partitioned array of patterns to send out ot subprocesses
    :param refSeq: Reference sequence to partition
    :param splits: How many resulting splits are to be returned
    :return: A partitioned array of patterns to send out ot subprocesses
    '''
    parts = len(refSeq) // splits
    return np.array_split(refSeq, parts)


def vectorEnumerateMotifs(vDNA: NDArray, searchPatterns: NDArray, d: int, subprocessID: int):
    '''
    One-by-one Brute-force motif enumeration comparing vector differences of given array elements
    :param vDNA: Vectorised DNA sequences
    :param searchPatterns: Patterns which will be searched through in the sequeences
    :param d: Number of mismatches that may occur in the motifs to be classified as proper
    :param subprocessID: ID of subprocess/rank (mainly for debugging)
    :return: A part of the final pattern dictionary
    '''
    print(f"[Worker {subprocessID}]: Initializing")
    # Initialize a dictionary for all patterns to append found motifs to
    patternsDict_part = {unvectorise(refPattern): [] for refPattern in searchPatterns}
    print(f"[Worker {subprocessID}]: Initialized, starting work. Displaying debug info every 100 patterns analysed.")
    sequence_id = 0 # Initialize the sequence counter
    for sequence in vDNA:
        sequence_id += 1 # For each sequence went through increment ID by 1
        # For each k-mer in refSeq
        for index, k_mer in enumerate(searchPatterns):
            start = time.perf_counter()
            # Print verbose progress every 100 k-mers
            if index % 100 == 0:
                print(
                    f"[Worker {subprocessID}]: Comparing {index + 1}/{len(searchPatterns)} in sequence {sequence_id}/{len(vDNA)}")
            # for k_mer' in sequence
            for k_merPrime in sequence:
                # Calculate the difference in k-mers
                patternDiff = k_merPrime - k_mer
                # If 0difference count is less or equal to mismatch threshold
                if np.count_nonzero(patternDiff) <= d:
                    # Append found proper k-mer to appropriate dictionary entry
                    patternsDict_part[unvectorise(searchPatterns[index])].append(unvectorise(k_merPrime))
            stop = time.perf_counter()
            print(f"[Worker {subprocessID}]: Comparing took {stop - start} seconds")
    # Return the resulting dictionary of patterns
    return patternsDict_part


def bigArraySubtractionMotifComparison(vDNA: NDArray, searchPatterns: NDArray, d: int, workerID: int):
    '''
    Faster method of brute-force motif enumeration utilizing array subtraction and result comparison
    :param vDNA: Vectorised DNA sequences
    :param searchPatterns: Patterns which will be searched through in the sequeences
    :param d: Number of mismatches that may occur in the motifs to be classified as proper
    :param workerID: ID of subprocess/rank (mainly for debugging)
    :return: A part of the final pattern dictionary
    '''
    # Initialize the dictionary for found proper patterns to be appended into
    patternDict_part = {''.join([unConversionDict[nuc] for nuc in refPattern]): [] for refPattern in searchPatterns}
    # For each sequence in vDNA
    for sequenceID, sequence in enumerate(vDNA):
        # For each pattern in sequence
        for patternIndex, pattern in enumerate(sequence):
            # Print verbose progress every 100 k-mers
            if patternIndex % 100 == 0 and patternIndex != 0:
                print(
                    f"[Worker {workerID}]: Comparing pattern {patternIndex + 1}/{len(sequence)} in sequence {sequenceID + 1}/{len(vDNA)}")
            elif patternIndex == 0:
                print(
                    f"[Worker {workerID}]: Comparing pattern {patternIndex + 1}/{len(sequence)} in sequence {sequenceID + 1}/{len(vDNA)}")
            # Create a pattern (motif) array of length same as reference sequence repeating the compared pattern (motif)
            patternarray = np.tile(pattern, [len(searchPatterns), 1])
            # Subtract the reference array and the constructed pattern (motif) array from one another
            comparisonarray = searchPatterns - patternarray
            # Iterate over the subtraction result and add only the results whose mismatches are <= d to dictionary
            for i, patternPrime in enumerate(comparisonarray):
                if np.count_nonzero(patternPrime) <= d:
                    patternDict_part[unvectorise(searchPatterns[i])].append(unvectorise(pattern))
    return patternDict_part


def createJSON(patternsDict, k, d):
    '''
    Write out program results in JSON format
    :param patternsDict: Dictionary to write out into the file
    :param k: k-length of searched motifs
    :param d: mismatch threshold of searched motifs
    :return: Pass-through dictionary of patterns and name of created file
    '''
    if not os.path.isdir('./motifs'):
        os.mkdir('./motifs')
    with open(f'./motifs/({k},{d})-motifs.json', 'w') as m:
        json.dump(patternsDict, m)
    createdFile = f'({k},{d})-motifs.json'
    return patternsDict, createdFile


if __name__ == '__main__':

    # PROGRAM ENGINE, TO BE REWRITTEN

    set_start_method("spawn")
    DNA = readSequences(0, 200, True)


    def PoolProcessing(workers, searchPatterns):
        foundPatterns = dict()
        with Pool(workers) as p:
            results = p.starmap(vectorEnumerateMotifs, [(vDNA, patterns, d, ID) for ID, patterns in
                                                        enumerate(searchPatterns)])
            for resultDict in results:
                foundPatterns.update(resultDict)
        createJSON(foundPatterns, k, d)


    def ArraySubtractionMultiprocessing(workers, searchPatterns):
        foundPatterns = dict()
        with Pool(workers) as p:
            results = p.starmap(bigArraySubtractionMotifComparison,
                                [(vDNA, searchPatterns, d, ID) for ID, searchPatterns in
                                 enumerate(searchPatterns)])
            for resultDict in results:
                foundPatterns.update(resultDict)
        createJSON(foundPatterns, k, d)


    initparam = False
    while 1:
        try:
            if not initparam:
                while 1:
                    try:
                        print("Input the desired length of mers.")
                        k = int(input("k = ? >> "))
                        print("Input the desired maximum number of mismatches.")
                        d = int(input("d = ? >> "))
                        initparam = True
                        break
                    except ValueError:
                        print("Invalid values have been input. Please use valid integers")

            print(f"Initializing and vectorising sequeces with k={k}, d={d}")
            vDNA = vectoriseSequences(k, DNA)
            print("Initialized! Continuing...\n")

            print("Choose the multiprocessing implementation to run the code with:\n"
                  "[1] Pool\n"
                  "[2] Array subtraction method")
            i1 = int(input("?>> "))
            if i1 == 1:
                print("Input the desired number of workers to initialize:\n"
                      "(The number must be less than 60, it's recommended to use the number of cores available in system)")
                workers = int(input("?>> "))
                print(
                    "Do you wish to search for patterns in the first sequence in the 'sequences' folder or to specify your own pattern(s)? [Y/N]")
                i2 = input("?>> ")
                if i2 in ["Y", "y", '']:
                    searchPatterns = makeSearchPatterns(vDNA[0], len(vDNA[0]) // workers)
                elif i2 in ["N", "n"]:
                    print("Please input desired patterns to search for, separated by spaces.")
                    initSearchPatterns = input("?>> ").upper().split()
                    if len(initSearchPatterns) == 1:
                        searchPatterns = np.array(
                            [[np.array([conversionDict[nuc] for nuc in pattern])] for pattern in initSearchPatterns])
                    else:
                        searchPatterns = np.array(
                            [np.array([conversionDict[nuc] for nuc in pattern]) for pattern in initSearchPatterns])
                    if len(searchPatterns) <= workers:
                        searchPatterns = np.tile(searchPatterns, [workers, 1])
                    else:
                        searchPatterns = np.array_split(searchPatterns, workers)
                PoolProcessing(workers, searchPatterns)
                break
            elif i1 == 2:
                print("Input the desired number of workers to initialize:\n"
                      "(The number must be less than 60, it's recommended to use the number of cores available in system)")
                workers = int(input("?>> "))
                print(
                    "Do you wish to search for patterns in the first sequence in the 'sequences' folder or to specify your own pattern(s)? [Y/N]")
                i2 = input("?>> ")
                if i2 in ["Y", "y", '']:
                    searchPatterns = makeSearchPatterns(vDNA[0], len(vDNA[0]) // workers)
                elif i2 in ["N", "n"]:
                    print("Please input desired patterns to search for, separated by spaces.")
                    initSearchPatterns = input("?>> ").upper().split()
                    if len(initSearchPatterns) == 1:
                        searchPatterns = np.array(
                            [[np.array([conversionDict[nuc] for nuc in pattern])] for pattern in initSearchPatterns])
                    else:
                        searchPatterns = np.array(
                            [np.array([conversionDict[nuc] for nuc in pattern]) for pattern in initSearchPatterns])
                    if len(searchPatterns) <= workers:
                        searchPatterns = np.tile(searchPatterns, [workers,1])
                    else:
                        searchPatterns = np.array_split(searchPatterns, workers)
                print(f"Initializing process on {workers} workers")
                ArraySubtractionMultiprocessing(workers, searchPatterns)
                break
            else:
                raise ValueError
        except ValueError:
            print("The input was incorrect, please use valid integers.")
