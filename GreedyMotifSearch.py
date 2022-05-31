"""
Jakub Susoł 274300
Zadanie 1 zaliczenie
Algorytmika

Algorytm GreedyMotifSearch
(Bioinformatics Algorithms - An Active Learning Approach vol. I, strona 85)
"""


# Ze względu na sposób implementacji algorytmów obliczających Score i Profile w poprzedniej części, redefiniuję
# te algorytmy pod zadaną funkcję GreedyMotifSearch


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
        print(f'Comparing k-mer {kmer+1}/{len(dna[0]) - k}')
        motifs = [dna[0][kmer:kmer + k]]
        for i in range(1, t):
            count, profile = count_profile(motifs, succession)
            motifs.append(profileMostProbableKMer(dna[i], k, profile))
        motifsCount, motifsProfile = count_profile(motifs)
        if score(motifsCount, len(motifs)) < score(bmCount, len(bestMotifs)):
            bestMotifs = motifs
    return {bestMotifs[0]:bestMotifs}


if __name__ == '__main__':
    from MotifEnumeration import readSequences
    from Motifs_Definitions import defineMotifs
    import pprint as pp

    DNA = readSequences()
    bestMotifsNoSuccession = greedyMotifSearch(DNA, 12, len(DNA), False)
    bestMotifsWithSuccession = greedyMotifSearch(DNA, 12, len(DNA))
    pp.pprint(defineMotifs(bestMotifsNoSuccession))
    pp.pprint(defineMotifs(bestMotifsWithSuccession))

