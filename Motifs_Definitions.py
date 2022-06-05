"""
Algorytmika

Algorytmy obliczające Score, Count, Profile i Consensus dla motywów uzyskanych od MotifEnumeration.
(Bioinformatics Algorithms - An Active Learning Approach vol. I)
"""
import os
import json
import re
import time


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


if __name__ == '__main__':
    import time

    start = time.time()
    motifsDict, file = loadDataOrCreate()
    makeComparisonJSON(motifsDict, file)
    end = time.time()
    print(f"The script took {end - start} s to finish")
