"""
Jakub Susoł 274300
Zadanie 1 zaliczenie
Algorytmika

Algorytm RandomMotifSearch
(Bioinformatics Algorithms - An Active Learning Approach vol. I, strona 93)
"""
from GreedyMotifSearch import count_profile, score, profileMostProbableKMer
from MotifEnumeration import readSequences
import random as rng

DNA = readSequences(0, 76, False)


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
    import time
    start = time.time()
    randomizedMotifSearch(DNA, 12)
    stop = time.time()
    print(f"This script ran in {stop-start} s")
