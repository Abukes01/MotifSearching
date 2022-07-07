import os
import sys
import json
import time
import re
import random as rng


# ENUMERATION
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


def motifEnumeration(dna: list, k: int, d: int):
    patterns = dict()  # new patterns set
    refSeq = dna[0]
    patternExistsIn = dict()
    for compDna in dna:
        # for each k-mer pattern in dna
        for i in range(len(refSeq) - k):
            pattern = refSeq[i:i + k]
            patternExistsIn[pattern] = [(0, pattern)]
            print(f'Comparing {i + 1}/{len(refSeq) - k} in {dna.index(compDna) + 1} of {len(dna)}')
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


# DEFINITIONS
def loadDataOrCreate():
    motifsDirectoryList = os.listdir('./motifs')
    # print(motifsDirectoryList)
    motifslist = []
    if motifsDirectoryList:
        for file in motifsDirectoryList:
            match = re.match(r'(.+)\.json', file)
            if match is not None:
                motifslist.append(match.string)
            else:
                pass
        fileloadchoice = True
        while fileloadchoice:
            indices = [i for i in range(len(motifslist))]
            for i, file in zip(indices, motifslist):
                print(f'[{i}] - {file}')
            try:
                chosenfile = int(input('\nPlease choose a file to load data from:\n>>> '))
                with open(f'./motifs/{motifslist[chosenfile]}') as f:
                    motifsdict = json.load(f)
                fileloadchoice = False
                return motifsdict, motifslist[chosenfile]
            except ValueError:
                print(
                    'The number you chose is either not included on available list or Your input is not a '
                    'number.\nPlease input a number from the list of possible inputs.')
            except IndexError:
                print(
                    'The number you chose is either not included on available list or Your input is not a '
                    'number.\nPlease input a number from the list of possible inputs.\n')
    else:
        from MotifEnumeration import readSequences, createJSON
        while True:
            try:
                k = int(input('Input k integer (how long a mer is) for motif enumeration\nK = '))
                d = int(input('Input d integer (how many mismatches there are at maximum) for motif enumeration\nD = '))
                lines = int(input(
                    'Input how long should the sequences be in lines of fasta file read for motif enumeration or enter 0 for all.\nLines = '))
                break
            except ValueError:
                print("The number you input is not a valid integer, please try again.\n")
        print("The program will now create a motifs file and read it. Starting in 5 seconds")
        time.sleep(5)
        if lines != 0:
            motifsdict, createdFile = createJSON(readSequences(0, lines, False), k, d)
        else:
            motifsdict, createdFile = createJSON(readSequences(0, 76, True), k, d)
        return motifsdict, createdFile
    pass


def makeMotifsDictFromList(motifslist: list):
    motifsdict = dict()
    for motifs in motifslist:
        motifsdict[motifs[0]] = motifs
    return motifsdict


def defineMotifs(motifsDict):
    opDict = dict()
    for key, item in motifsDict.items():
        opDict[key] = {}
        opDict[key]["Motifs"] = item
        opDict[key]["Score"] = 0
        opDict[key]["Count"] = {"A": [0 for _ in range(len(item[0]))],
                                "C": [0 for _ in range(len(item[0]))],
                                "G": [0 for _ in range(len(item[0]))],
                                "T": [0 for _ in range(len(item[0]))]}
        opDict[key]["Profile"] = {"A": [],
                                  "C": [],
                                  "G": [],
                                  "T": []}
        opDict[key]["Consensus"] = ''
        listedSeqs = [list(motSeq) for motSeq in motifsDict[key]]

        # Count

        for nucleotide in range(len(listedSeqs[0])):
            for sequence in range(len(listedSeqs)):
                if listedSeqs[sequence][nucleotide] == 'T':
                    opDict[key]["Count"]["T"][nucleotide] += 1
                elif listedSeqs[sequence][nucleotide] == 'G':
                    opDict[key]["Count"]["G"][nucleotide] += 1
                elif listedSeqs[sequence][nucleotide] == 'C':
                    opDict[key]["Count"]["C"][nucleotide] += 1
                elif listedSeqs[sequence][nucleotide] == 'A':
                    opDict[key]["Count"]["A"][nucleotide] += 1

        # Profile

        for opKey, opItem in opDict[key]["Count"].items():
            for value in opItem:
                percent = value / len(item)
                opDict[key]["Profile"][opKey].append(percent)

        # Consensus

        pKeys = [pKey for pKey in opDict[key]["Profile"].keys()]
        pItems = [pItem for pItem in opDict[key]["Profile"].values()]
        for nucIndex in range(len(pItems[0])):
            percents = []
            for nucleotide in range(len(pKeys)):
                percents.append(pItems[nucleotide][nucIndex])
            opDict[key]["Consensus"] += pKeys[percents.index(max(percents))]

        # Score

        listedConsensus = list(opDict[key]["Consensus"])
        for sequence in listedSeqs:
            for cNuc, sNuc in zip(listedConsensus, sequence):
                if cNuc != sNuc:
                    opDict[key]["Score"] += 1
    return opDict


def makeComparisonJSON(motifsDict, file):
    opDict = defineMotifs(motifsDict)
    if not os.path.isdir('./motifs/definitions'):
        os.mkdir('./motifs/definitions')
    with open(f'./motifs/definitions/({file})_definitions.json', 'w') as d:
        json.dump(opDict, d)
    return opDict


# MEDIANSTR
def patternToNumber(pattern: str):
    sym2num = {"A": 0, "C": 1, "G": 2, "T": 3}
    if not pattern:
        return 0
    symbol = pattern[-1]
    prefix = pattern[:-1]
    return 4 * patternToNumber(prefix) + sym2num[symbol]


def numberToPattern(index: int, k: int):
    def genstr(index, k):
        num2sym = {0: "A", 1: "C", 2: "G", 3: "T"}
        pattern = ''
        if k == 1:
            return pattern.join(num2sym[index % 4])
        prefixIndex = index // 4
        r = index % 4
        pattern += num2sym[r]
        pattern += genstr(prefixIndex, k - 1)
        return pattern

    pattern = ''.join(reversed(list(genstr(index, k))))
    return pattern


def distancePatternString(pattern: str, dna: list):
    def hammingDistance(text, compare):
        distance = 0
        for a, b in zip(text, compare):
            if a != b:
                distance += 1
        return distance

    k = len(pattern)
    distance = 0
    for sequence in dna:
        hammingDist = 1e100
        for i in range(len(sequence) - k):
            patternprime = sequence[i:i + k]
            if hammingDist > hammingDistance(pattern, patternprime):
                hammingDist = hammingDistance(pattern, patternprime)
        distance += hammingDist
    return distance


def medianString(dna: list, k: int):
    distance = 1e100
    median = ''
    for i in range(4 ** k):
        print(f'Comparing pattern {i + 1}/{4 ** k}')
        pattern = numberToPattern(i, k)
        patternDist = distancePatternString(pattern, dna)
        if distance > patternDist:
            distance = patternDist
            median = pattern
    return median, print(median)


# GREEDY
def count_profile(motifs: list, succession=True):
    # Uwzględniam poprawkę zasady sukcesji Laplace'a jako parametr opcjonalny, z podstawy True
    listedMotifStrings = [list(strToList) for strToList in motifs]
    count = {"A": [0 for _ in range(len(listedMotifStrings[0]))],
             "C": [0 for _ in range(len(listedMotifStrings[0]))],
             "G": [0 for _ in range(len(listedMotifStrings[0]))],
             "T": [0 for _ in range(len(listedMotifStrings[0]))]}
    profile = {"A": [],
               "C": [],
               "G": [],
               "T": []}

    # Count
    for row in range(len(listedMotifStrings)):
        for col in range(len(listedMotifStrings[row])):
            if listedMotifStrings[row][col] == "A":
                count["A"][col] += 1
            elif listedMotifStrings[row][col] == "G":
                count["G"][col] += 1
            elif listedMotifStrings[row][col] == "C":
                count["C"][col] += 1
            elif listedMotifStrings[row][col] == "T":
                count["T"][col] += 1

    # Profile
    if succession:
        for key, item in count.items():
            for value in item:
                percent = (value + 1) / (2 * len(item))
                profile[key].append(percent)
    else:
        for key, item in count.items():
            for value in item:
                percent = value / len(item)
                profile[key].append(percent)
    return count, profile


def score(count: dict, lengthOfMotifsMatrix: int):
    score = 0
    cKeys = [key for key in count.keys()]
    cItems = [item for item in count.values()]
    for col in range(len(cItems[0])):
        cValsList = [cItems[row][col] for row in range(len(cKeys))]
        score += lengthOfMotifsMatrix - max(cValsList)
    return score


def profileMostProbableKMer(dna: str, k: int, profile: dict):
    patterns, probabilities = [], []
    for i in range(len(dna) - k):
        pattern = dna[i:i + k]
        patterns.append(pattern)
        probability = 1
        for nucleotide in range(len(pattern)):
            probability *= profile[pattern[nucleotide]][nucleotide]
        probabilities.append(probability)
    return patterns[probabilities.index(max(probabilities))]


def greedyMotifSearch(dna: list, k: int, t: int, succession=True):
    bestMotifs = [rawDna[0:k] for rawDna in dna]
    bmCount, bmProfile = count_profile(bestMotifs, succession)
    for kmer in range(len(dna[0]) - k):
        # print(f'Comparing k-mer {kmer+1}/{len(dna[0]) - k}')
        motifs = [dna[0][kmer:kmer + k]]
        for i in range(1, t):
            count, profile = count_profile(motifs, succession)
            motifs.append(profileMostProbableKMer(dna[i], k, profile))
        motifsCount, motifsProfile = count_profile(motifs)
        if score(motifsCount, len(motifs)) < score(bmCount, len(bestMotifs)):
            bestMotifs = motifs
    return {bestMotifs[0]: bestMotifs}


# RANDOM
def randomizedMotifSearch(dna: list, k: int):
    # randomly select k-mers
    motifs = []
    for sequence in dna:
        i = rng.randint(0, len(sequence) - k)
        motifs.append(sequence[i:i + k])
    bestMotifs = motifs
    # perform the search
    while True:
        count, profile = count_profile(motifs, False)
        bmCount, bmProfile = count_profile(bestMotifs, False)
        motifs = []
        for sequence in dna:
            motifs.append(profileMostProbableKMer(sequence, k, profile))
        mCount, mProfile = count_profile(motifs, False)
        if score(mCount, len(motifs)) < score(bmCount, len(bestMotifs)):
            bestMotifs = motifs
        else:
            return bestMotifs


if __name__ == '__main__':
    class ChoiceException(Exception):
        pass


    readall = True
    linestart = 0
    linestop = 0
    while True:
        try:
            allLines = input("Read all lines? [Y/N, default: Y]\n?>> ")
            if allLines in ["y", "Y", ""]:
                print("Reading all sequence data...")
                break
            elif allLines in ["n", "N"]:
                readall = False
                linestart = int(input("Please input the line to start reading from in the sequences:\n>>> "))
                linestop = int(input("Please input the line to stop reading on in the sequences:\n>>> "))
                if linestart>=0 and linestop>0:
                    print(f'Reading sequence data from lines {linestart} to {linestop}...')
                    break
            else:
                raise ChoiceException
        except ValueError:
            print("You must provide valid integer numbers from 0 to the amount of lines your sequence files have.\n")
        except ChoiceException:
            print("You must provide a Y/N answer or leave blank for default.\n")

    DNA = readSequences(linestart, linestop, readall)

    while True:
        try:
            k = int(input("Please input the length of motifs you are searching for:\n?>> "))
            d = int(input("Please input the maximum number of mismatches in the motifs you are searching for:\n?>> "))
            if k and d:
                print(f"The algorithms will search for {k}-length motifs with at most {d} mutations/mismatches")
                break
        except ValueError:
            print("Please input a valid integer.")

    run = True
    while run:
        try:
            print('Choose what algorithm you want to use:\n'
                  '[1] Enumeration and definition\n'
                  '[2] Median String\n'
                  '[3] Greedy search\n'
                  '[4] Random Search')
            choice1 = int(input("?>> "))
            if choice1 == 1:
                print("Running Motif Enumeration and Motif Definition algorithms, this may take a moment.")
                createJSON(DNA, k, d)
                motifsDict, file = loadDataOrCreate()
                makeComparisonJSON(motifsDict, file)
                print("Results saved in motifs/definitions/ folder.")
                run = False
            elif choice1 == 2:
                print("Running Median String algorithm, this may take a moment.")
                with open('./medianStringSequences.txt', 'w') as result:
                    medstr = medianString(DNA, k)[0]
                    result.write(medstr)
                run = False
            elif choice1 == 3:
                print("Running Greedy Motif Search algorithm, this may take a moment.")
                bestMotifsNoSuccession = greedyMotifSearch(DNA, k, len(DNA), False)
                bestMotifsWithSuccession = greedyMotifSearch(DNA, k, len(DNA))
                makeComparisonJSON(bestMotifsNoSuccession, 'greedy_NoSuccession')
                makeComparisonJSON(bestMotifsWithSuccession, 'greedy_Succession')
                print("Results saved in motifs/definitions/ folder.")
                run = False
            elif choice1 == 4:
                print("Running Randomized Motif Search algorithm, this may take a moment.")
                while True:
                    try:
                        iterations = int(input("Please input the number of iterations to run:\n?>> "))
                        if iterations <= 0:
                            raise ValueError
                        else:
                            break
                    except ValueError:
                        print("Please input a valid, non-negative and non-zero integer.\n")
                with open('./RandomSearchResults.json', 'w') as result:
                    searchResults = [randomizedMotifSearch(DNA, k) for _ in range(iterations)]
                    resultsDict = {searchResults[i][0]: searchResults[i] for i in range(len(searchResults))}
                    json.dump(resultsDict, result)
                run = False
            else:
                raise ChoiceException
        except ChoiceException:
            print("Please choose a valid option from the listed options.\n")
