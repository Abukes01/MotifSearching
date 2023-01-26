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
from mpi4py import MPI
from itertools import cycle

# Conversion dictionaries for converting nucleotides to numbers and numbers to nucleotides. The values are of importance
# and the dictionaries should mirror each other's inverse of key-value pairs.
conversionDict: dict = {"A": 1, "T": 2, "G": 3, "C": 4, "N": 0, "Y": 0, "R": 0, "W": 0, "S": 0, "K": 0, "M": 0, "D": 0,
                        "V": 0, "H": 0, "B": 0}
unConversionDict: dict = {1: "A", 2: "T", 3: "G", 4: "C", 0: "N"}


# Load data as plaintext and save all sequences into a separate file (same as in normal program)
# This is relatively fast compared to the rest of the algorithms, and I'm a little too lazy to change already
# proven to work code. If anyone wishes to optimize it, go for it, just make sure it works as intended :)
def readSequences(linestart: int, linestop: int, all=True) -> (list, dict):
    """
    Compiles sequences in the appropriate folder into one file for easy access and returns a list of strings,
    where each element is a one-line representative of the gene data from the FASTA files taken in order from the folder.

    :param linestart: Where to start reading the files from, set 0 if reading all
    :param linestop: Where to stop reading from files at, set 0 if reading all
    :param all: Whether to read entire sequences or not (Default: True)
    :return: List of strings, each string is the genetic sequence from files in the appropriate folder in order.
    """

    def readFolder(ls=linestart, lstop=linestop, all=all):
        if not all:
            with open("./compiled_sequences.fasta", 'w') as f:
                for sequence in os.listdir('./sequences'):
                    with open(f'./sequences/{sequence}', 'rt') as s:
                        readseq = ''
                        for line in s.readlines()[ls:lstop]:
                            if line.startswith('>'):
                                readseq += line
                            else:
                                readseq += line.strip('\n')
                        readseq += '\n'
                        f.writelines(readseq)
        else:
            with open("./compiled_sequences.fasta", 'w') as f:
                for sequence in os.listdir('./sequences'):
                    with open(f'./sequences/{sequence}', 'rt') as s:
                        readseq = ''
                        for line in s.readlines():
                            if line.startswith('>'):
                                readseq += line
                            else:
                                readseq += line.strip('\n')
                        readseq += '\n'
                        f.writelines(readseq)
        del readseq

    def makeSeqenceDict(file: str, mode: str) -> dict:
        with open(file, mode) as f:
            headers, sequences = [], []
            for line in f.readlines():
                if line.startswith('>'):
                    headers.append(line)
                else:
                    sequences.append(line)
            seqdict = {header: sequence for header, sequence in zip(headers, sequences)}
            del headers, sequences
        return seqdict

    DNA = []
    seqDict = dict()
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
            readFolder()
            seqDict = makeSeqenceDict('./compiled_sequences.fasta', 'rt')
            DNA = [sequence.strip('\n') for sequence in seqDict.values()]
    else:
        try:
            while True:
                choice = input(
                    "There are sequences that were compiled previously. Load from them? (Select N if sequences were updated) [Y/N]\n?>> ")
                if choice in ["Y", 'y', '']:
                    seqDict = makeSeqenceDict('./compiled_sequences.fasta', 'rt')
                    DNA = [sequence.strip('\n') for sequence in seqDict.values()]
                    break
                elif choice in ["N", 'n']:
                    readFolder()
                    seqDict = makeSeqenceDict('./compiled_sequences.fasta', 'rt')
                    DNA = [sequence.strip('\n') for sequence in seqDict.values()]
                    break
                else:
                    raise ValueError
        except ValueError:
            print("The provided answer is invalid, try Y or N.\n")
    return DNA, seqDict


# Vectorise the sequences read as plaintext
def vectoriseSequences(k: int, sequences: list) -> NDArray:
    """
    Takes a list of sequences and returns an array of arrays, containing k-length pieces of sequences from beginning to
    end shifting right by one nucleotide. Nucleotides are encoded numerically as follows:

    A -> 1
    T -> 2
    G -> 3
    C -> 4
    Any ambiguity code -> 0

    This is crucial information for decoding the outcome sequences for further processing.

    :param k: (Integer) Defines the length of a k-mer in vectorised sequences
    :param sequences: (List) The list of sequences to vectorise
    :return: An array of arrays containing k-length numerically encoded pieces of given sequences in appropriate order.
    """

    numDNA = []  # Define starting list for vectorisation
    for sequence in sequences:  # For each sequence in sequences
        numSequence = []  # Define a temporary list for appending conversion outcome
        for nucleotide in sequence.strip('\n'):  # For each nucleotide in given sequence without newlines
            numSequence.append(conversionDict[nucleotide])  # Append the converted nucleotide number to temporary list
        numDNA.append(np.array(numSequence))  # Change the list to NumPy Array
    numDNA = np.array(numDNA)  # Change whole converted sequences list to an array
    # EG. k=5 ['ATGCCGTAGTTAGGACT'] -> [1, 2, 3, 4, 4, 3, 2, 1, 3, 2, 2, 1, 3, 3, 1, 4, 2]

    # Return an array containing fragmented k-length pieces of sequences in appropriate order
    # EG. k=5 [1, 2, 3, 4, 4, 3, 2, 1, 3, 2, 2, 1, 3, 3, 1, 4, 2] ->
    # [
    #  [1, 2, 3, 4, 4],
    #  [2, 3, 4, 4, 3], [3, 4, 4, 3, 2], [4, 4, 3, 2, 1], [4, 3, 2, 1, 3], [3, 2, 1, 3, 2], [2, 1, 3, 2, 2],
    #  [1, 3, 2, 2, 1], [3, 2, 2, 1, 3], [2, 2, 1, 3, 3], [2, 1, 3, 3, 1], [1, 3, 3, 1, 4], [3, 3, 1, 4, 2]
    # ]
    return np.array([[numSequence[i:i + k] for i in range(len(numSequence) - (k - 1))] for numSequence in numDNA])


def unvectorise(vectorToConvert) -> str:  # Reverse the process of vectorisation for given vector
    return ''.join([unConversionDict[nucleotide] for nucleotide in vectorToConvert])


def makeSearchPatterns(refSeq: NDArray, splits: int) -> NDArray:
    '''
    Return a partitioned array of patterns to send out ot subprocesses
    :param refSeq: Reference sequence to partition
    :param splits: How many resulting splits are to be returned
    :return: A partitioned array of patterns to send out ot subprocesses
    '''
    parts = len(refSeq) // splits
    return np.array_split(refSeq, parts)


def arraySubtractionMotifComparison(vDNA: NDArray, searchPatterns: NDArray, d: int, workerID: int) -> dict:
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
            if patternIndex % 100 == 0 or patternIndex == 0:
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


def createJSON(patternsDict: dict, k: int, d: int) -> (dict, str):
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


def programInit(lineStart: int, lineStop: int, k: int, all=False) -> (dict, dict):
    DNA, seqDict = readSequences(lineStart, lineStop, all)
    vectorDNA = vectoriseSequences(k, DNA)
    vSeqDict = {header: vSequence for header, vSequence in zip(seqDict.keys(), vectorDNA)}
    return seqDict, vSeqDict


def computeResults(sequenceID: str, vSequence: NDArray, vSearchPatterns: NDArray, mismatches: int,
                   workerID: str | int) -> dict:
    resultPatternDict = {sequenceID: {unvectorise(searchpattern): [] for searchpattern in vSearchPatterns}}
    for patternID, pattern in enumerate(vSequence):
        if patternID % 20 == 0 or patternID == 0:
            print(f"[{workerID = }] Comparing pattern {patternID + 1}/{len(vSequence)}")
        patternArray = np.tile(pattern, [len(vSearchPatterns), 1])
        subtractionResult = vSearchPatterns - patternArray
        for i, patternPrime in enumerate(subtractionResult):
            if np.count_nonzero(patternPrime) <= mismatches:
                resultPatternDict[sequenceID][unvectorise(vSearchPatterns[i])].append(unvectorise(pattern))
    return resultPatternDict


def MPIRun(seqDict: dict, vSeqDict: dict, vSearchPatterns: NDArray,k:int, mismatches: int) -> None:
    #                           OLD MULTITHREADED IMPLEMENTATION FOR REFERENCE
    #
    # def ArraySubtractionMultiprocessing(workers, searchPatterns):
    #     foundPatterns = dict()
    #     with Pool(workers) as p:
    #         results = p.starmap(bigArraySubtractionMotifComparison,
    #                             [(vDNA, searchPatterns, d, ID) for ID, searchPatterns in
    #                              enumerate(searchPatterns)])
    #         for resultDict in results:
    #             foundPatterns.update(resultDict)
    #     createJSON(foundPatterns, k, d)

    # Step 1: Initialize dataset to work on in a clever way, utilizing the maximum of allocated resources in effecient ways
    if len(list(vSeqDict.keys())) >= MPI.COMM_WORLD.size - 1:
        seqIDs = list(vSeqDict.keys())
        vSeqList = list(vSeqDict.values())
        searchPatternsList = [vSearchPatterns for _ in range(len(seqIDs))]
        mismatchesList = [mismatches for _ in range(len(seqIDs))]

        tasks = [params for params in zip(seqIDs, vSeqList, searchPatternsList,
                                          mismatchesList)]  # (Tasks should be tuples of the main compute function's parameters)
        tasksNumber = len(tasks)
        killTag = tasksNumber + 1
        resulsDict = {ID:{} for ID in seqIDs}
    elif len(list(vSeqDict.keys())) < MPI.COMM_WORLD.size - 1:
        remainingToFill = MPI.COMM_WORLD.size - 1 % len(list(vSeqDict.keys())) # How many free cores remain
        # Duplicate sequence IDs for free cores
        rawID = list(vSeqDict.keys())
        seqIDs = []
        for ID in rawID:
            if rawID.index(ID) < remainingToFill:
                seqIDs.append(ID)
                seqIDs.append(ID)
            else:
                seqIDs.append(ID)
        # make split sequences for free cores
        rawVSeq = list(vSeqDict.values())
        vSeqList = []
        for seq in rawVSeq:
            if rawVSeq.index(seq) < remainingToFill:
                splitseq = np.array_split(seq, 2)
                vSeqList.append(splitseq[0])
                vSeqList.append(splitseq[1])
            else:
                vSeqList.append(seq)
        searchPatternsList = [vSearchPatterns for _ in range(len(seqIDs))]
        mismatchesList = [mismatches for _ in range(len(seqIDs))]
        tasks = [params for params in zip(seqIDs, vSeqList, searchPatternsList,
                                          mismatchesList)]  # (Tasks should be tuples of the main compute function's parameters)
        tasksNumber = len(tasks)
        killTag = tasksNumber + 1
        resulsDict = {ID: {} for ID in seqIDs}

    def master(workers):
        # Status declaration
        status = MPI.Status()
        start = MPI.Wtime()
        print(f"Starting program with {size - 1} workers.")

        results: list[dict] = []
        dummyData: list[None] = [None]

        # Step 2: Send out first batch of tasks consisting of previously initialized parts of dataset to workers

        sent: int = 0
        for _ in range(workers):
            print(f"Sending task {_} to worker at rank {_ + 1}")
            comm.send(tasks[_], dest=_ + 1, tag=_)
            sent = _ + 1

        # Step 3: Receive complete computation results and send remaining tasks
        while sent < tasksNumber:
            result = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)  # assume worker outputs a dict
            workerID, workerTag = status.Get_source(), status.Get_tag()
            print(f"Received result from {workerID = } with {workerTag = }, appending to results list")
            results.append(result)
            print(f"Sending remaining task {sent} to worker {workerID}")
            comm.send(tasks[sent], dest=workerID, tag=sent)
            sent += 1

        # Step 4: Receive the remaining computation results and send the kill tag
        for _ in range(workers):
            result = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)  # assume worker outputs a dict
            workerID, workerTag = status.Get_source(), status.Get_tag()
            print(f"Received result from {workerID = } with {workerTag = }, appending to results list")
            results.append(result)
            print("All tasks have been received. Sending KILL tag.")
            comm.send(dummyData, dest=workerID, tag=killTag)

        # Step 5: Do proper operations on results and save.

        for res in results:
            seqID = str(*list(res.keys()))
            resulsDict[seqID].update(res[seqID])

        createJSON(resulsDict, k, mismatches)

        print(
            f"Master node received all completed tasks from all {workers} workers and processed results, terminating...")
        stop = MPI.Wtime()
        time = stop - start
        with open('./times.csv', 'a+') as t:
            t.write(f"{size}, {time}")


    def worker(worker_rank):
        # Initialize
        status = MPI.Status()
        completedTasks = 0
        print(f'Worker {worker_rank} initialized, waiting for jobs')
        # Get first tasks
        params = comm.recv(source=0, status=status)
        workerTag = status.Get_tag()
        print(f"Worker {worker_rank} received a task and started work.")
        # Start operations on received data, send results and receive remaining data, stop when killTag encountered
        while workerTag != killTag:
            # compute results and send them to master
            print(f"Worker {worker_rank} starting on work with tag {workerTag}")
            result = computeResults(*params, worker_rank)
            comm.send(result, dest=0, tag=workerTag)
            completedTasks += 1
            # receive next data to work on and its tag, start computation on it unless killTag encountered
            params = comm.recv(source=0, status=status)
            workerTag = status.Get_tag()
        print(f"Worker {worker_rank} encountered killTag with {completedTasks = }, terminating...")

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    if rank == 0:
        master(size - 1)
    else:
        worker(rank)


if __name__ == '__main__':
    # HERE GOES THE PROGRAM ENGINE
    # (AND HERE BE DRAGONS)
    pass
